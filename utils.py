import torch

def get_accuracy_from_logits(logits, labels):
    predictions = torch.argmax(logits, dim=1)
    return (predictions == labels).float().mean()