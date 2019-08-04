import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--global-epochs", help="Number of global (server) epochs", type=int, default=10,
                        required=False)
    parser.add_argument("-c", "--clients", help="Number of clients", type=int, default=100, required=False)
    parser.add_argument("-f", "--fraction", help="Client fraction to use", type=float, default=0.2,
                        required=False)
    parser.add_argument("-d", "--debug", help="Debugging", action="store_true", required=False)

    parser.add_argument("-lr", "--learning-rate", help="Learning rate", type=float, default=0.01, required=False)
    parser.add_argument("-b", "--batch-size", help="Batch Size", type=int, default=32, required=False)
    parser.add_argument("-ce", "--client-epochs", help="Number of epochs for the clients", type=int, default=5,
                        required=False)
    args = parser.parse_args()
    return args
