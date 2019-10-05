import json
from pathlib import Path

import numpy as np
from keras import datasets

import fed_learn

args = fed_learn.get_args()

fed_learn.set_working_GPU(str(args.gpu))

experiment_folder_path = Path(__file__).resolve().parent / "experiments" / args.name
experiment = fed_learn.Experiment(experiment_folder_path, args.overwrite_experiment)
experiment.serialize_args(args)

tf_scalar_logger = experiment.create_scalar_logger()

client_train_params = {"epochs": args.client_epochs, "batch_size": args.batch_size}


def model_fn():
    return fed_learn.create_model((32, 32, 3), 10, init_with_imagenet=False, learning_rate=args.learning_rate)


weight_summarizer = fed_learn.FedAvg()
server = fed_learn.Server(model_fn,
                          weight_summarizer,
                          args.clients,
                          args.fraction)

weight_path = args.weights_file
if weight_path is not None:
    server.load_model_weights(weight_path)

server.update_client_train_params(client_train_params)
server.create_clients()

(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
data_handler = fed_learn.DataHandler(x_train, y_train, x_test, y_test, fed_learn.CifarProcessor(), args.debug)
data_handler.assign_data_to_clients(server.clients, args.data_sampling_technique)
x_test, y_test = data_handler.preprocess(data_handler.x_test, data_handler.y_test)

for epoch in range(args.global_epochs):
    print("Global Epoch {0} is starting".format(epoch))
    server.init_for_new_epoch()
    selected_clients = server.select_clients()

    fed_learn.print_selected_clients(selected_clients)

    for client in selected_clients:
        print("Client {0} is starting the training".format(client.id))

        server.send_model(client)
        hist = client.edge_train(server.get_client_train_param_dict())
        server.epoch_losses.append(hist.history["loss"][-1])

        server.receive_results(client)

    server.summarize_weights()

    epoch_mean_loss = np.mean(server.epoch_losses)
    server.global_train_losses.append(epoch_mean_loss)
    tf_scalar_logger.log_scalar("train_loss/client_mean_loss", server.global_train_losses[-1], epoch)
    print("Loss (client mean): {0}".format(server.global_train_losses[-1]))

    global_test_results = server.test_global_model(x_test, y_test)
    print("--- Global test ---")
    test_loss = global_test_results["loss"]
    test_acc = global_test_results["acc"]
    print("{0}: {1}".format("Loss", test_loss))
    print("{0}: {1}".format("Accuracy", test_acc))
    tf_scalar_logger.log_scalar("test_loss/global_loss", test_loss, epoch)
    tf_scalar_logger.log_scalar("test_acc/global_acc", test_acc, epoch)

    with open(str(experiment.train_hist_path), 'w') as f:
        json.dump(server.global_test_metrics_dict, f)

    # TODO: save only when a condition is fulfilled (validation loss gets better, etc...)
    server.save_model_weights(experiment.global_weight_path)

    print("_" * 30)
