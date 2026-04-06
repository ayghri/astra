import torch
from torch import nn
import math
import transformers


def sparsegpt_prune(weights, hessian, blocksize, sparsity, prune_n, prune_m, percdamp):

    W = weights.float().clone()
    H = hessian.clone()

    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0
    rows, cols = W.shape

    Losses = torch.zeros(rows).to(H)

    H.diagonal().add_(percdamp * H.diagonal().mean())

    H_inv = torch.cholesky_inverse(torch.linalg.cholesky(H))
    C = torch.linalg.cholesky(H_inv, upper=True)

    mask = None

    for i1 in range(0, cols, blocksize):
        i2 = min(i1 + blocksize, cols)
        count = i2 - i1

        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Losses1 = torch.zeros_like(W1)
        C_sub = C[i1:i2, i1:i2]

        if prune_n == 0:
            if mask is not None:
                mask1 = mask[:, i1:i2]
            else:
                tmp = W1**2 / (torch.diag(C_sub).reshape((1, -1))) ** 2
                thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * sparsity)]
                mask1 = tmp <= thresh
        else:
            mask1 = torch.zeros_like(W1) == 1

        for i in range(count):
            w = W1[:, i]
            d = C_sub[i, i]

            if prune_n != 0 and i % prune_m == 0:
                tmp = (
                    W1[:, i : (i + prune_m)] ** 2
                    / (torch.diag(C_sub)[i : (i + prune_m)].reshape((1, -1))) ** 2
                )
                mask1.scatter_(
                    1,
                    i + torch.topk(tmp, prune_n, dim=1, largest=False)[1],
                    True,
                )

            q = w.clone()
            q[mask1[:, i]] = 0

            Q1[:, i] = q
            Losses1[:, i] = (w - q) ** 2 / d**2

            err1 = (w - q) / d
            W1[:, i:] -= err1.unsqueeze(1).matmul(C_sub[i, i:].unsqueeze(0))
            Err1[:, i] = err1

        W[:, i1:i2] = Q1
        Losses += torch.sum(Losses1, 1) / 2

        W[:, i2:] -= Err1.matmul(C[i1:i2, i2:])

    return W


## SparseGPT: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
class SparseGPT:
    def __init__(self, layer):
        self.layer = layer
        self.device = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.device)
        self.nsamples = 0

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(
            self.layer, transformers.Conv1D
        ):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    def fasterprune(self, sparsity, prune_n=0, prune_m=0, blocksize=128, percdamp=0.01):
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        W = sparsegpt_prune(
            W,
            self.H,
            blocksize,
            sparsity=sparsity,
            prune_n=prune_n,
            prune_m=prune_m,
            percdamp=percdamp,
        )

        torch.cuda.synchronize()
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(
            self.layer.weight.data.dtype
        )

    def free(self):
        self.H = None
        torch.cuda.empty_cache()


def find_layers(module, layers=[nn.Linear], name=""):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    found_layers = {}
    for name1, child in module.named_children():
        found_layers.update(
            find_layers(
                child,
                layers=layers,
                name=name + "." + name1 if name != "" else name1,
            )
        )
    return found_layers


# @torch.no_grad()
# def prune_sparsegpt(args, model, tokenizer, dev, prune_n=0, prune_m=0):
#     ## SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
#     print("Starting ...")
#     dataloader, _ = get_loaders(
#         "c4",
#         nsamples=args.nsamples,
#         seed=args.seed,
#         seqlen=model.seqlen,
#         tokenizer=tokenizer,
#     )

#     use_cache = model.config.use_cache
#     model.config.use_cache = False
#     decoder_layers = model.model.layers

#     if "model.embed_tokens" in model.hf_device_map:
#         dev = model.hf_device_map["model.embed_tokens"]

#     dtype = next(iter(model.parameters())).dtype
#     inputs = torch.zeros(
#         (args.nsamples, model.seqlen, model.config.hidden_size),
#         dtype=dtype,
#         device=dev,
#     )
#     cache = {"i": 0, "attention_mask": None, "position_ids": None}

#     class Catcher(nn.Module):
#         def __init__(self, module):
#             super().__init__()
#             self.module = module

#         def forward(self, inp, **kwargs):
#             inputs[cache["i"]] = inp
#             cache["i"] += 1
#             cache["attention_mask"] = kwargs["attention_mask"]
#             cache["position_ids"] = kwargs["position_ids"]
#             raise ValueError

#     decoder_layers[0] = Catcher(decoder_layers[0])
#     for batch in dataloader:
#         try:
#             model(batch[0].to(dev))
#         except ValueError:
#             pass
#     decoder_layers[0] = decoder_layers[0].module
#     torch.cuda.empty_cache()

#     outputs = torch.zeros_like(inputs)
#     attention_mask = cache["attention_mask"]
#     position_ids = cache["position_ids"]

#     print("Ready.")

#     for i in range(len(decoder_layers)):
#         layer = decoder_layers[i]
#         if f"model.layers.{i}" in model.hf_device_map:
#             dev = model.hf_device_map[f"model.layers.{i}"]
#             print(f"layer {i} device {dev}")
#             inputs, outputs, attention_mask, position_ids = (
#                 inputs.to(dev),
#                 outputs.to(dev),
#                 attention_mask.to(dev),
#                 position_ids.to(dev),
#             )

#         subset = find_layers(layer)

#         gpts = {}
#         handles = []

#         def add_batch(name):
#             def tmp(_, inp, out):
#                 gpts[name].add_batch(inp[0].data, out.data)

#             return tmp

#         for name in subset:
#             gpts[name] = SparseGPT(subset[name])
#             handles.append(subset[name].register_forward_hook(add_batch(name)))

#         for j in range(args.nsamples):
#             outputs[j] = layer(
#                 inputs[j].unsqueeze(0),
#                 attention_mask=attention_mask,
#                 position_ids=position_ids,
#             )[0]
#         for h in handles:
#             h.remove()

#         for name in gpts:
#             print(i, name)
#             print("Pruning ...")

#             gpts[name].fasterprune(
#                 args.sparsity_ratio,
#                 prune_n=prune_n,
#                 prune_m=prune_m,
#                 percdamp=0.01,
#                 blocksize=128,
#             )
#             gpts[name].free()

#         for j in range(args.nsamples):
#             outputs[j] = layer(
#                 inputs[j].unsqueeze(0),
#                 attention_mask=attention_mask,
#                 position_ids=position_ids,
#             )[0]

#         decoder_layers[i] = layer
#         torch.cuda.empty_cache()

#         inputs, outputs = outputs, inputs

#     model.config.use_cache = use_cache
#     torch.cuda.empty_cache()
