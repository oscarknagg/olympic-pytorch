import unittest
from sklearn.datasets import make_classification
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
from torch import Tensor

from olympic.eval import evaluate


class TestEval(unittest.TestCase):
    def test_eval_with_passed_callable(self):
        x, y = make_classification(128, n_classes=3, n_features=5, n_informative=5, n_redundant=0, n_repeated=0)
        data = TensorDataset(Tensor(x), Tensor(y).long())
        dataloader = DataLoader(data, batch_size=32)
        model = nn.Sequential(nn.Linear(5, 3))

        def custom_metric(y_true, y_pred):
            return 0.5

        # Run eval function with custom callable
        metrics = evaluate(model, dataloader, metrics=['accuracy', custom_metric])
