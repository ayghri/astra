"""
Licensed under CC BY-NC 4.0 (see LICENSE or https://creativecommons.org/licenses/by-nc/4.0/)
Non-commercial use only; contact us for commercial licensing.
"""

import os
import sys
import torch
from tqdm import tqdm
from lm_eval.tasks import TaskManager
from lm_eval.models import huggingface
from lm_eval.evaluator import simple_evaluate


def evaluate_accuracy(model, data_loader, device="cuda"):
    """
    Evaluates the accuracy of a model on a given data loader.
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


def get_model_sparsity(model, eps=1e-8):
    total_params = 0
    nnz = 0
    for p in model.parameters():
        total_params += p.numel()
        nnz += (p.abs() <= eps).sum().item()
    return (nnz / total_params) if total_params > 0 else 0


def evaluate_ppl_hf(hf_model, tokenizer, silent=False):
    # Store original stdout and stderr

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    if silent:
        devnull = open(os.devnull, "w")
        sys.stdout = devnull
        sys.stderr = devnull

    hf_model = huggingface.HFLM(hf_model, tokenizer=tokenizer)

    task_manager = TaskManager()
    with torch.no_grad():
        results = simple_evaluate(
            model=hf_model,
            tasks=["wikitext"],
            num_fewshot=0,
            task_manager=task_manager,
            log_samples=False,
            batch_size=2,
        )
    if silent:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        devnull.close()

    return results["results"]
