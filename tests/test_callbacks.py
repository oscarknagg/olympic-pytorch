import unittest
from unittest.mock import patch, MagicMock
from sklearn.datasets import make_classification
from torch.utils.data import DataLoader, TensorDataset
from torch import nn, optim
from torch import Tensor

from olympic import fit
from olympic.callbacks import *


__PATH__ = os.path.dirname(os.path.realpath(__file__))


def constant_loss_update_fn(model, optimiser, loss_fn, x, y, epoch, **kwargs):
    """Performs no calculations but just automatically returns a pre-determined loss"""
    return Tensor([1]), y


class TestCallbacks(unittest.TestCase):
    def test_reduce_lr_on_plateau(self):
        x, y = make_classification(128, n_classes=3, n_features=5, n_informative=5, n_redundant=0, n_repeated=0)
        data = TensorDataset(Tensor(x), Tensor(y).long())
        dataloader = DataLoader(data, batch_size=32)
        model = nn.Sequential(nn.Linear(5, 3))

        reduce_lr_on_plateau = ReduceLROnPlateau(monitor='loss', verbose=True, patience=1)

        with patch.object(reduce_lr_on_plateau, '_reduce_lr') as mock_reduce_lr:
            fit(
                model=model,
                loss_fn=nn.CrossEntropyLoss(),
                optimiser=optim.SGD(model.parameters(), lr=0.1),
                epochs=2,
                dataloader=dataloader,
                metrics=['accuracy'],
                callbacks=[reduce_lr_on_plateau],
                update_fn=constant_loss_update_fn,
            )
            mock_reduce_lr.assert_called_once()
