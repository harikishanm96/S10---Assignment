# S10---Assignment

## Model summary

![image](https://github.com/harikishanm96/S10---Assignment/assets/53985105/29fa06c2-32e9-4450-bae2-21edf04e873f)

## LR finder

net_exp = copy.deepcopy(net)

optimizer = torch.optim.Adam(net_exp.parameters(), lr=0.001, weight_decay=0.01)

criterion = nn.CrossEntropyLoss()

lr_finder = LRFinder(net_exp, optimizer, criterion, device=device)

lr_finder.range_test(trainloader, end_lr=0.1, num_iter=200)

lr_finder.plot()

Suggested LR: 1.32E-03

![image](https://github.com/harikishanm96/S10---Assignment/assets/53985105/29364eb7-a134-40bd-a255-4a59485176e6)


## Logs

Epoch: 0,Loss=1.25 Batch_id=97 Accuracy=46.52: 100%|██████████| 98/98 [00:31<00:00,  3.13it/s]
Test set: Average loss: 0.0022, Accuracy: 5988/10000 (59.88%)

Epoch: 1,Loss=0.94 Batch_id=97 Accuracy=61.53: 100%|██████████| 98/98 [00:24<00:00,  3.96it/s]
Test set: Average loss: 0.0019, Accuracy: 6703/10000 (67.03%)

Epoch: 2,Loss=0.83 Batch_id=97 Accuracy=68.81: 100%|██████████| 98/98 [00:25<00:00,  3.87it/s]
Test set: Average loss: 0.0018, Accuracy: 6969/10000 (69.69%)

Epoch: 3,Loss=0.80 Batch_id=97 Accuracy=72.33: 100%|██████████| 98/98 [00:24<00:00,  4.07it/s]
Test set: Average loss: 0.0015, Accuracy: 7272/10000 (72.72%)

Epoch: 4,Loss=0.78 Batch_id=97 Accuracy=74.97: 100%|██████████| 98/98 [00:24<00:00,  3.94it/s]
Test set: Average loss: 0.0015, Accuracy: 7403/10000 (74.03%)

Epoch: 5,Loss=0.71 Batch_id=97 Accuracy=78.00: 100%|██████████| 98/98 [00:24<00:00,  4.02it/s]
Test set: Average loss: 0.0012, Accuracy: 8032/10000 (80.32%)

Epoch: 6,Loss=0.46 Batch_id=97 Accuracy=79.89: 100%|██████████| 98/98 [00:24<00:00,  4.02it/s]
Test set: Average loss: 0.0010, Accuracy: 8294/10000 (82.94%)

Epoch: 7,Loss=0.48 Batch_id=97 Accuracy=81.67: 100%|██████████| 98/98 [00:23<00:00,  4.09it/s]
Test set: Average loss: 0.0010, Accuracy: 8396/10000 (83.96%)

Epoch: 8,Loss=0.47 Batch_id=97 Accuracy=83.13: 100%|██████████| 98/98 [00:24<00:00,  3.97it/s]
Test set: Average loss: 0.0010, Accuracy: 8302/10000 (83.02%)

Epoch: 9,Loss=0.47 Batch_id=97 Accuracy=84.82: 100%|██████████| 98/98 [00:24<00:00,  4.06it/s]
Test set: Average loss: 0.0009, Accuracy: 8457/10000 (84.57%)

Epoch: 10,Loss=0.41 Batch_id=97 Accuracy=86.27: 100%|██████████| 98/98 [00:23<00:00,  4.09it/s]
Test set: Average loss: 0.0009, Accuracy: 8473/10000 (84.73%)

Epoch: 11,Loss=0.48 Batch_id=97 Accuracy=86.78: 100%|██████████| 98/98 [00:23<00:00,  4.19it/s]
Test set: Average loss: 0.0008, Accuracy: 8720/10000 (87.20%)

Epoch: 12,Loss=0.41 Batch_id=97 Accuracy=87.34: 100%|██████████| 98/98 [00:24<00:00,  4.05it/s]
Test set: Average loss: 0.0007, Accuracy: 8734/10000 (87.34%)

Epoch: 13,Loss=0.39 Batch_id=97 Accuracy=88.72: 100%|██████████| 98/98 [00:24<00:00,  4.01it/s]
Test set: Average loss: 0.0007, Accuracy: 8814/10000 (88.14%)

Epoch: 14,Loss=0.27 Batch_id=97 Accuracy=89.39: 100%|██████████| 98/98 [00:25<00:00,  3.88it/s]
Test set: Average loss: 0.0006, Accuracy: 8920/10000 (89.20%)

Epoch: 15,Loss=0.35 Batch_id=97 Accuracy=90.48: 100%|██████████| 98/98 [00:24<00:00,  4.00it/s]
Test set: Average loss: 0.0007, Accuracy: 8824/10000 (88.24%)

Epoch: 16,Loss=0.20 Batch_id=97 Accuracy=90.97: 100%|██████████| 98/98 [00:24<00:00,  3.99it/s]
Test set: Average loss: 0.0008, Accuracy: 8746/10000 (87.46%)

Epoch: 17,Loss=0.25 Batch_id=97 Accuracy=91.58: 100%|██████████| 98/98 [00:24<00:00,  3.97it/s]
Test set: Average loss: 0.0006, Accuracy: 8972/10000 (89.72%)

Epoch: 18,Loss=0.20 Batch_id=97 Accuracy=92.35: 100%|██████████| 98/98 [00:24<00:00,  4.04it/s]
Test set: Average loss: 0.0006, Accuracy: 8993/10000 (89.93%)

Epoch: 19,Loss=0.18 Batch_id=97 Accuracy=92.75: 100%|██████████| 98/98 [00:24<00:00,  4.00it/s]
Test set: Average loss: 0.0006, Accuracy: 9058/10000 (90.58%)

Epoch: 20,Loss=0.14 Batch_id=97 Accuracy=93.72: 100%|██████████| 98/98 [00:24<00:00,  4.04it/s]
Test set: Average loss: 0.0005, Accuracy: 9098/10000 (90.98%)

Epoch: 21,Loss=0.16 Batch_id=97 Accuracy=94.36: 100%|██████████| 98/98 [00:24<00:00,  4.02it/s]
Test set: Average loss: 0.0005, Accuracy: 9131/10000 (91.31%)

Epoch: 22,Loss=0.16 Batch_id=97 Accuracy=94.79: 100%|██████████| 98/98 [00:24<00:00,  4.06it/s]
Test set: Average loss: 0.0005, Accuracy: 9142/10000 (91.42%)

Epoch: 23,Loss=0.14 Batch_id=97 Accuracy=95.24: 100%|██████████| 98/98 [00:24<00:00,  4.03it/s]
Test set: Average loss: 0.0005, Accuracy: 9155/10000 (91.55%)
