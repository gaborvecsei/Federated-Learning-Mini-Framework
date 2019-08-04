import json

import numpy as np

import fed_learn

args = fed_learn.get_args()

nb_clients = args.clients
nb_global_epochs = args.global_epochs
debug = args.debug

client_train_params = {"epochs": args.client_epochs, "batch_size": args.batch_size}


def model_fn():
    return fed_learn.create_model((32, 32, 3), 10, init_with_imagenet=False, learning_rate=args.learning_rate)


weight_summarizer = fed_learn.FedAvg()
server = fed_learn.Server(model_fn, nb_clients, weight_summarizer, debug)
server.update_client_train_params(client_train_params)

for epoch in range(nb_global_epochs):
    print("Global Epoch {0} is starting".format(epoch))
    server.init_for_new_epoch()
    server.create_clients()

    for client in server.clients:
        print("Client {0} is starting the training".format(client.id))

        server.send_model(client)
        server.send_train_data(client)

        hist = client.edge_train(server.get_client_train_param_dict())
        server.epoch_losses.append(hist.history["loss"][-1])

        server.receive_results(client)

    server.summarize_weights()

    epoch_mean_loss = np.mean(server.epoch_losses)
    server.global_train_losses.append(epoch_mean_loss)
    print("Loss (client mean): {0}".format(server.global_train_losses[-1]))

    global_test_results = server.test_global_model()
    print("Global test|")
    for metric_name, value in global_test_results.items():
        print("_" * 10)
        print("{0}: {1}".format(metric_name, value))

    with open("fed_learn_global_test_results.json", 'w') as f:
        json.dump(server.global_test_metrics_dict, f)

    print("_" * 30)
