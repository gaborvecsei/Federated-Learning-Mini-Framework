import fed_learn

NB_CLIENTS = 10
NB_EPOCHS = 3


def model_fn():
    return fed_learn.create_model((32, 32, 3), 10)


weight_summarizer = fed_learn.FedAvg()
server = fed_learn.Server(model_fn, NB_CLIENTS, weight_summarizer)

for epoch in range(NB_EPOCHS):
    print("Global Epoch {0} is starting".format(epoch))
    server.create_clients()
    server.init_for_new_epoch()

    loss = []

    for client in server.clients:
        print("Client {0} is starting the training".format({client.id}))

        server.send_model(client)
        server.send_train_data(client)

        hist = client.edge_train(server.get_client_train_param_dict())
        loss.append(hist.history["loss"])

        server.receive_results(client)

    server.summarize_weights()
    print("-" * 30)

    # TODO: test the base model with the aggregated weights
