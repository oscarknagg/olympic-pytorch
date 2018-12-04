import torch
import torch.nn.functional as F


def accuracy(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Calculates categorical accuracy.

    # Arguments:
        y_true: Ground truth categories. Must have shape [batch_size,]
        y_pred: Prediction probabilities or logits of shape [batch_size, num_categories]
    """
    return torch.eq(y_pred.argmax(dim=-1), y_true).sum().item() / y_pred.shape[0]


def top_k_accuracy(y_true: torch.Tensor, y_pred: torch.Tensor, top_k: int = 5) -> float:
    """Computes the precision@k for the specified values of k

    # Arguments:
        y_true: Ground truth categories. Must have shape [batch_size,]
        y_pred: Prediction probabilities or logits of shape [batch_size, num_categories]
        top_k:
    """
    batch_size = y_true.size(0)

    _, pred = y_pred.topk(top_k, 1, True, True)
    pred = pred.t()
    correct = pred.eq(y_true.view(1, -1).expand_as(pred))

    correct_k = correct[:top_k].view(-1).float().sum(0, keepdim=True)
    return correct_k.mul_(1. / batch_size).item()


def mean_absolute_error(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Calculates mean absolute error.

    # Arguments:
        y_true: Ground truth values. Must have shape [batch_size, *] where * is any additional
            number of dimensions
        y_pred: Predicted vales of shape [batch_size,*] where * is any additional number
            of dimensions. Must have the same size as y
    """
    return F.l1_loss(y_pred, y_true, reduction='elementwise_mean').item()


def mean_squared_error(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Calculates mean squared error.

    # Arguments:
        y_true: Ground truth values. Must have shape [batch_size, *] where * is any additional
            number of dimensions
        y_pred: Predicted vales of shape [batch_size,*] where * is any additional number
            of dimensions. Must have the same size as y
    """
    return F.mse_loss(y_pred, y_true, reduction='elementwise_mean').item()


NAMED_METRICS = {
    'accuracy': accuracy,
    'top_k_accuracy': top_k_accuracy,
    'mean_absolute_error': mean_absolute_error,
    'mae': mean_absolute_error,
    'mean_squared_error': mean_squared_error,
    'mse': mean_squared_error,
}
