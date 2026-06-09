"""
Licensed under CC BY-NC 4.0 (see LICENSE or https://creativecommons.org/licenses/by-nc/4.0/)
Non-commercial use only; contact us for commercial licensing.
"""

import os
from typing import List, Dict
import hashlib
import re
from omegaconf import OmegaConf, ListConfig
from hydra import initialize, compose
from torch import optim
from torch.optim import lr_scheduler


from adasoft.controllers import AlphaController
from adasoft.controllers import EMAController
from adasoft.controllers import LambdaController


def _filter_kwargs(target, kwargs: Dict, exclude=None):
    import inspect

    if exclude is None:
        exclude = ()
    opt_args = list(inspect.signature(target).parameters.keys())
    return {
        k: v for k, v in kwargs.items() if k in opt_args and k not in exclude
    }


def get_envs(env_names: List[str], ignore=False) -> Dict[str, str]:
    env_values = {}
    for env_n in env_names:
        env_val = os.environ.get(env_n, None)
        if not ignore and env_val is None:
            raise ValueError(f"env variable {env_n} is not set")
        env_values[env_n] = env_val
    return env_values


def hash_config(cfg):
    """Generate a unique hash for a given configuration dictionary."""
    cfg_str = OmegaConf.to_yaml(cfg, resolve=True)
    cfg_hash = hashlib.md5(cfg_str.encode()).hexdigest()[:8]
    return cfg_hash


def get_model(model_cfg, **kwargs):
    name = model_cfg.name
    if name.lower().startswith("wideresnet"):
        from adasoft.models.wideresnet import get_wideresnet

        call_func = get_wideresnet
    elif name.lower().startswith("resnet"):
        from adasoft.models.resnets import get_resnet

        call_func = get_resnet
    else:
        raise ValueError(f"Unknown model name: {name}")

    filtered_kwargs = _filter_kwargs(call_func, model_cfg)
    filtered_kwargs.update(kwargs)

    return call_func(**filtered_kwargs)


def get_optimizer(optimizer_cfg, model_params) -> optim.Optimizer:
    name = str(getattr(optimizer_cfg, "name", "")).lower()
    cls_map = {
        "sgd": optim.SGD,
        "adamw": optim.AdamW,
    }
    opt_cls = cls_map.get(name, None)
    if opt_cls is None:
        raise ValueError(
            f"Unknown optimizer {getattr(optimizer_cfg, 'name', name)}"
        )
    kwargs = _filter_kwargs(opt_cls, optimizer_cfg)

    return opt_cls(model_params, **kwargs)


def get_lr_scheduler(lr_scheduler_cfg, optimizer) -> lr_scheduler.LRScheduler:
    name = str(getattr(lr_scheduler_cfg, "name", "")).lower()

    cls_map = {
        "cosine": lr_scheduler.CosineAnnealingLR,
        "multistep": lr_scheduler.MultiStepLR,
    }
    sch_cls = cls_map.get(name, None)
    if sch_cls is None:
        raise ValueError(
            f"Unknown lr scheduler name: {name}, allowed: {list(cls_map.keys())}"
        )
    kwargs = {k: v for k, v in lr_scheduler_cfg.items()}
    if name == "multistep":
        # we create
        ratio = lr_scheduler_cfg.step_ratio
        num_epochs = lr_scheduler_cfg.num_epochs
        if isinstance(ratio, float):
            step = int(num_epochs * ratio)
            offset_ratio = lr_scheduler_cfg.offset_ratio  # not used: negative
            offset = int(offset_ratio * num_epochs)
            if offset < 0:
                offset = step
            kwargs["milestones"] = list(range(offset, num_epochs, step))
        elif isinstance(ratio, ListConfig) or isinstance(ratio, list):
            kwargs["milestones"] = [int(r * num_epochs) for r in ratio]
        else:
            raise ValueError("step_ratio must be float or list of floats")

    kwargs = _filter_kwargs(sch_cls, kwargs)
    return sch_cls(optimizer, **kwargs)


def _match_rules(name, include, exclude):
    for pattern in exclude:
        if re.match(pattern, name):
            return False
    for pattern in include:
        if re.match(pattern, name):
            return True
    return False


def get_sparsity_specs(specs_cfg, named_parameters, default_exclude):
    # default_cls = specs_cfg._class_
    specs = {}
    named_parameters = list(named_parameters)
    for k, s_cfg in specs_cfg.items():
        if k == "_class_":
            continue
        specs[k] = []
        # spec_cls = specs.get("_class_", default_cls)
        exclude = s_cfg.exclude + default_exclude
        include = s_cfg.include
        num_parameters = len(named_parameters)
        for _ in range(num_parameters):
            param_name, param = named_parameters.pop(0)
            if _match_rules(param_name, include, exclude):
                # instantiate(
                #     {
                #         "_target_": spec_cls,
                #         **{
                #             "param": param,
                #             "block_size": tuple(s_cfg.block_size),
                #             "group_size": tuple(s_cfg.group_size),
                #             "name": param_name,
                #         },
                #     }
                specs[k].append(
                    BlockGroupSpec(
                        param=param,
                        block_size=tuple(s_cfg.block_size),
                        group_size=tuple(s_cfg.group_size),
                        name=param_name,
                    )
                )
            else:
                named_parameters.append((param_name, param))
    return specs


def get_sparsity_groups(
    coupling_cfg, name_to_specs: Dict[str, List], default_sparsity: float
) -> List[SpecCoupler]:
    groups = []
    # for group_name, group_cfg in coupling_cfg.items():
    global_cfg = coupling_cfg.get("global", None)
    if global_cfg is not None:
        coupled_specs = []
        coupled_orders = []
        group_sparsity = global_cfg.get("sparsity", default_sparsity)
        for spec_name, order in global_cfg.specs.items():
            coupled_specs += name_to_specs[spec_name]
            coupled_orders += [tuple(order) for _ in name_to_specs[spec_name]]

        groups.append(
            SpecCoupler(
                specs=coupled_specs,
                orders=coupled_orders,
                sparsity=group_sparsity,
                name="global",
            )
        )
    indiv_cfg = coupling_cfg.get("indiv", None)
    if indiv_cfg is not None:
        group_sparsity = indiv_cfg.get("sparsity", default_sparsity)
        for spec_name, order in indiv_cfg.specs.items():
            for spec in name_to_specs[spec_name]:
                groups.append(
                    SpecCoupler(
                        specs=[spec],
                        orders=[()],
                        sparsity=group_sparsity,
                        name="indiv",
                    )
                )

    return groups


def get_alphas(alpha_cfg, name_to_specs: Dict[str, List]) -> AlphaController:
    alphas = AlphaController(default=alpha_cfg.default)
    for name, alp in alpha_cfg.items():
        if name == "default":
            continue
        for s in name_to_specs[name]:
            alphas.set(s, alp)
    return alphas


def get_lambdas(lambda_cfg, device) -> LambdaController:
    return LambdaController(
        device=device, **{k: v for k, v in lambda_cfg.items()}
    )


def get_ema(ema_cfg) -> EMAController:
    return EMAController(**{k: v for k, v in ema_cfg.items()})


def init_wandb(wandb_cfg, cfg):
    import wandb  # type: ignore

    return wandb.init(
        project=wandb_cfg.project,
        entity=wandb_cfg.get("entity", None),
        config=OmegaConf.to_container(cfg, resolve=True),
        mode=wandb_cfg.mode,
        group=wandb_cfg.group,
    )


def print_wandb_info(config_name, config_path):
    """Prints wandb project and entity"""

    with initialize(version_base=None, config_path=config_path):
        cfg = compose(config_name=config_name)
        project = cfg.wandb.project
        entity = cfg.wandb.entity
        print(f"{entity}/{project}")
