import pytest
import numpy as np
import torch

from olympic import metrics


classification_metrics = [
    metrics.accuracy,
    metrics.top_k_accuracy
]


@pytest.mark.parametrize('metric', classification_metrics)
def test_classification_metrics(metric):
    y = torch.Tensor(np.random.randint(0, 7, size=(6,))).long()
    y_pred = torch.Tensor(np.random.random((6, 7)))
    output = metric(y, y_pred)
    assert isinstance(output, float)
