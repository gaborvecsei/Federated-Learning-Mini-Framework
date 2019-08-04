# Federated Learning Example

## Cifar10 experiments

Training on Cifar10 with IID data where we had 100 clients and for each round (global epoch) we used only
10% of them (selected randomly). Every client fitted 1 epoch on "their part" of the data with the batch size of 64.

`python fl.py -e 200 -c 100 -f 0.1 -lr 0.2 -b 64 -ce 1`

<img src="art/fl_3_clients_accuracy.png" width="250">
