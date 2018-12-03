Quickstart
==========

This quickstart guide will give a minimal code example using Olympic. This example is also available as a Jupyter
notebook at olympic-pytorch/notebooks/Quickstart.ipynb

First make all of the necessary imports.::

    from torch import nn, optim
    from torch.utils.data import DataLoader
    import torch.nn.functional as F
    from torchvision import transforms, datasets
    from multiprocessing import cpu_count

    import olympic

Create datasets.::

    transform = transforms.Compose([
       transforms.ToTensor(),
       transforms.Normalize((0.1307,), (0.3081,))
    ])

    train = datasets.MNIST('', train=True, transform=transform, download=True)
    test = datasets.MNIST('', train=False, transform=transform, download=True)

    train_loader = DataLoader(train, batch_size=128, num_workers=cpu_count())
    test_loader = DataLoader(test, batch_size=128, num_workers=cpu_count())

Define network.::

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
            self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
            self.conv2_drop = nn.Dropout2d()
            self.fc1 = nn.Linear(320, 50)
            self.fc2 = nn.Linear(50, 10)

        def forward(self, x):
            x = F.relu(F.max_pool2d(self.conv1(x), 2))
            x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
            x = x.view(-1, 320)
            x = F.relu(self.fc1(x))
            x = F.dropout(x, training=self.training)
            x = self.fc2(x)
            return F.log_softmax(x, dim=1)

Instantiate network, loss and optimiser.::

    model = Net()
    optimiser = optim.SGD(model.parameters(), lr=0.1)
    loss_fn = nn.CrossEntropyLoss()

Create desired callbacks.::

    callbacks = [
        # Evaluates every epoch on val_loader
        olympic.callbacks.Evaluate(val_loader),
        # Saves model with best val_accuracy
        olympic.callbacks.ModelCheckpoint('model.pt', save_best_only=True, monitor='val_accuracy'),
        # Logs all metrics
        olympic.callbacks.CSVLogger('log.csv')
    ]

Call ``olympic.fit``::

    olympic.fit(
        model,
        optimiser,
        loss_fn,
        dataloader=train_loader,
        epochs=10,
        metrics=['accuracy'],
        callbacks=callbacks
    )

You should see this output.::

    Begin training...

    Epoch 1:  26%|██▌       | 122/469 [00:03<00:09, 35.70it/s, loss=0.515, accuracy=0.867]

The network will train for 10 epochs. The current directory will contain both ``model.pt`` and ``log.csv`` which
should look something like this.::

    epoch,accuracy,loss,val_accuracy,val_loss
    1,0.7888348436389482,0.6585237751605668,0.9437,0.1692712503015995
    2,0.9093039267945985,0.3049919113421491,0.9712,0.08768766190297901
    3,0.9272832267235251,0.24685336495322713,0.9745,0.07711423026025295
    4,0.9375388681592041,0.21396846514044285,0.9777,0.06789233392337338
    5,0.9416588930348259,0.19915449465595203,0.9815,0.0603904211839661
    6,0.9476168265813789,0.18155415136136735,0.9822,0.05375468297088519
    7,0.9493048152096659,0.1694526430894571,0.984,0.04907846948835067
    8,0.953008395522388,0.16376275851377356,0.9852,0.04469430861719884
    9,0.9561122956645345,0.15457178367329621,0.9859,0.043301032841484996
    10,0.9554237739872068,0.1532330308109522,0.9869,0.0410145413863007
