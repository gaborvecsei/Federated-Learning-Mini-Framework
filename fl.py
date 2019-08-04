import argparse
import json

import numpy as np

import fed_learn

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--global-epochs", help="Number of global (server) epochs", type=int, default=5,
                    required=False)
parser.add_argument("-c", "--clients", help="Number of clients", type=int, default=10, required=False)
parser.add_argument("-d", "--debug", help="Debugging", action="store_true", required=False)
args = parser.parse_args()

nb_clients = args.clients
nb_epochs = args.global_epochs
debug = args.debug


def model_fn():
    return fed_learn.create_model((32, 32, 3), 10, init_with_imagenet=False)


weight_summarizer = fed_learn.FedAvg()
server = fed_learn.Server(model_fn, nb_clients, weight_summarizer, debug)

for epoch in range(nb_epochs):
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
    for metric_name, value in global_test_results.items():
        print("Global test|")
        print("_" * 10)
        print("{0}: {1}".format(metric_name, value))

    print("_" * 30)

with open("fed_learn_global_test_results.json", 'w') as f:
    json.dump(server.global_test_metrics_dict, f)
