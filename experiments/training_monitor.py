# experiments/training_monitor.py

def compute_gradient_norm(model):
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.data.norm(2).item() ** 2
    return total ** 0.5


def compute_weight_norm(model):
    total = 0.0
    for p in model.parameters():
        total += p.data.norm(2).item() ** 2
    return total ** 0.5
