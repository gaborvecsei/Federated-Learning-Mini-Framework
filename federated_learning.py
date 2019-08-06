import json
from pathlib import Path

import numpy as np

import fed_learn

args = fed_learn.get_args()

EXPERIMENT_FOLDER_PATH = Path(__file__).resolve().parent / "experiments" / args.name
EXPERIMENT_FOLDER_PATH.mkdir(parents=True, exist_ok=args.overwrite_experiment)

args_json_path = EXPERIMENT_FOLDER_PATH / "args.json"
fed_learn.save_args_as_json(args, EXPERIMENT_FOLDER_PATH / args_json_path)

train_hist_path = EXPERIMENT_FOLDER_PATH / "fed_learn_global_test_results.json"

client_train_params = {"epochs": args.client_epochs, "batch_size": args.batch_size}


def model_fn():
    return fed_learn.create_model((32, 32, 3), 10, init_with_imagenet=False, learning_rate=args.learning_rate)


weight_summarizer = fed_learn.FedAvg()
server = fed_learn.Server(model_fn, weight_summarizer, args.clients, args.fraction, args.debug)
server.update_client_train_params(client_train_params)
server.create_clients()
server.send_train_data()

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
    print("Loss (client mean): {0}".format(server.global_train_losses[-1]))

    global_test_results = server.test_global_model()
    print("--- Global test ---")
    for metric_name, value in global_test_results.items():
        print("{0}: {1}".format(metric_name, value))

    with open(str(train_hist_path), 'w') as f:
        json.dump(server.global_test_metrics_dict, f)

    print("_" * 30)
