from typing import List
from pathlib import Path
import numpy as np
import torch


def save_checkpoint(model, model_name, epoch, checkpoint_dir, cfg):
    checkpoint_dir = Path(checkpoint_dir)
    if checkpoint_dir.is_file():
        raise ValueError(f"{checkpoint_dir} is a file, must be a directory")
    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    state = {
        "state_dict": model.state_dict(),
        "epoch": epoch,
        "cfg": cfg,
    }
    torch.save(state, checkpoint_dir / f"{model_name}_best_{epoch}.pth")


def evaluate_accuracy(model, val_loader, output_method=None):
    correct = 0
    total = 0
    model.eval()
    if output_method is not None:
        predict = output_method
    else:
        predict = model

    with torch.no_grad():
        for data in val_loader:
            images, labels = data
            device = next(model.parameters()).device
            images, labels = images.to(device), labels.to(device)
            outputs = predict(images)
            predicted = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
    return accuracy


def erdos_renyi_kernel(
    parameters: List[torch.nn.Parameter],
    sparsity: float,
    excluded_params=None
):
    assert sparsity > 0 and sparsity < 1
    epsilon = 1.0
    is_epsilon_valid = False
    # # The following loop will terminate worst case when all masks are in the
    # custom_sparsity_map. This should probably never happen though, since once
    # we have a single variable or more with the same constant, we have a valid
    # epsilon. Note that for each iteration we add at least one variable to the
    # custom_sparsity_map and therefore this while loop should terminate.
    # dense_params = set(excluded_params)
    dense_params = set()
    if excluded_params is not None:
        dense_params = set(excluded_params)
    # default_ones = sum([p.numel() for p in excluded_params])
    raw_probabilities = {}
    while not is_epsilon_valid:
        # We will start with all layers and try to find right epsilon. However if
        # any probablity exceeds 1, we will make that layer dense and repeat the
        # process (finding epsilon) with the non-dense layers.
        # We want the total number of connections to be the same. Let say we have
        # for layers with N_1, ..., N_4 parameters each. Let say after some
        # iterations probability of some dense layers (3, 4) exceeded 1 and
        # therefore we added them to the dense_layers set. Those layers will not
        # scale with erdos_renyi, however we need to count them so that target
        # paratemeter count is achieved. See below.
        # eps * (p_1 * N_1 + p_2 * N_2) + (N_3 + N_4) =
        #    (1 - default_sparsity) * (N_1 + N_2 + N_3 + N_4)
        # eps * (p_1 * N_1 + p_2 * N_2) =
        #    (1 - default_sparsity) * (N_1 + N_2) - default_sparsity * (N_3 + N_4)
        # eps = rhs / (\sum_i p_i * N_i) = rhs / divisor.

        divisor = 0
        rhs = 0
        raw_probabilities = {}
        # for name, mask in masking.mask_dict.items():
        for param in parameters:
            n_param = param.numel()
            n_ones = int(n_param * (1 - sparsity))
            if param in dense_params:
                rhs -= n_param
            else:
                rhs += n_ones
                # Erdos-Renyi probability: epsilon * (n_in + n_out) / (n_in * n_out).
                # raw_probabilities[param] = np.sum(param.shape) / n_param
                raw_probabilities[param] = np.sum(param.shape[:2]) / n_param
                # Note that raw_probabilities[mask] * n_param gives the individual
                # elements of the divisor.
                divisor += raw_probabilities[param] * n_param
        # By multipliying individual probabilites with epsilon, we should get the
        # number of parameters per layer correctly.
        epsilon = rhs / divisor
        # If epsilon * raw_probabilities[mask.name] > 1. We set the sparsities of that
        # mask to 0., so they become part of dense_layers sets.
        max_prob = np.max(list(raw_probabilities.values()))
        max_prob_one = max_prob * epsilon
        if max_prob_one > 1:
            is_epsilon_valid = False
            for mask_name, mask_raw_prob in raw_probabilities.items():
                if mask_raw_prob >= max_prob:
                    dense_params.add(mask_name)
        else:
            is_epsilon_valid = True

    prob_dict = {}

    for param in parameters:
        if param in dense_params:
            prob_dict[param] = 1.0
        else:
            prob_dict[param] = epsilon * raw_probabilities[param]

    return prob_dict
