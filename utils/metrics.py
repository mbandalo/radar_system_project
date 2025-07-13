import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def entropy(probs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Computes entropy of a probability distribution.
    Args:
        probs: Probability distributions (..., num_classes).
    Returns:
        Entropy values.
    """
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().cpu().numpy()
    probs = np.clip(probs, eps, 1.0)
    return -np.sum(probs * np.log(probs), axis=-1)

def info_gain(prior: np.ndarray, posterior: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Computes information gain (reduction in entropy).
    Args:
        prior: Prior probability distributions.
        posterior: Posterior probability distributions.
    Returns:
        Information gain values.
    """
    return entropy(prior, eps) - entropy(posterior, eps)

def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, average: str = 'macro') -> dict:
    """
    Computes common classification metrics.
    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        average: Aggregation method ('micro', 'macro', 'weighted', 'binary', or None).
    Returns:
        A dictionary of metrics.
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    if y_true.ndim == 1: # Single-label case
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
            'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
            'f1': f1_score(y_true, y_pred, average=average, zero_division=0)
        }

    # Multi-label case
    n_classes = y_true.shape[1]
    per_class = {}
    for c in range(n_classes):
        true_c = y_true[:, c]
        pred_c = y_pred[:, c]
        per_class[c] = {
            'accuracy': accuracy_score(true_c, pred_c),
            'precision': precision_score(true_c, pred_c, average='binary', zero_division=0),
            'recall': recall_score(true_c, pred_c, average='binary', zero_division=0),
            'f1': f1_score(true_c, pred_c, average='binary', zero_division=0)
        }
    if average is None:
        return per_class

    return { # Aggregate for requested average
        'accuracy': np.mean([per_class[c]['accuracy'] for c in per_class]),
        'precision': np.mean([per_class[c]['precision'] for c in per_class]),
        'recall': np.mean([per_class[c]['recall'] for c in per_class]),
        'f1': np.mean([per_class[c]['f1'] for c in per_class])
    }