import unittest
from sklearn.datasets import make_classification
from torch.utils.data import DataLoader, TensorDataset
from torch import nn, optim
from torch import Tensor
import os

from olympic import callbacks as cbks
from olympic import fit

from .utils import mkdir, rmdir


__PATH__ = os.path.dirname(os.path.realpath(__file__))


class TestIntegrations(unittest.TestCase):
    def test_classification(self):
        mkdir(__PATH__ + '/tmp/')

        x, y = make_classification(128, n_classes=3, n_features=5, n_informative=5, n_redundant=0, n_repeated=0)
        data = TensorDataset(Tensor(x), Tensor(y).long())
        dataloader = DataLoader(data, batch_size=32)
        model = nn.Sequential(nn.Linear(5, 3))

        callbacks = [
            cbks.Evaluate(dataloader),
            cbks.CSVLogger(__PATH__ + '/tmp/log.csv'),
            cbks.ModelCheckpoint(__PATH__ + '/tmp/model.pt')
        ]

        history = fit(
            model=model,
            loss_fn=nn.CrossEntropyLoss(),
            optimiser=optim.SGD(model.parameters(), lr=0.1),
            epochs=1,
            dataloader=dataloader,
            metrics=['accuracy'],
            callbacks=callbacks
        )

        history_metrics = ['loss', 'accuracy', 'val_loss', 'val_accuracy']
        self.assertTrue(all(m in history[0] for m in history_metrics))

        rmdir(__PATH__ + '/tmp/')
