"""
Licensed under CC BY-NC 4.0 (see LICENSE or https://creativecommons.org/licenses/by-nc/4.0/)
Non-commercial use only; contact us for commercial licensing.
"""
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb


def validate_parameter_ranges(params):
    """Validate that parameter values stay within specified ranges."""
    for param_name, param_config in params.items():
        if "min" in param_config and "max" in param_config:
            min_val = param_config["min"]
            max_val = param_config["max"]
            if min_val >= max_val:
                raise ValueError(
                    f"Invalid range for {param_name}: min ({min_val}) must be less than max ({max_val})"
                )

            # For log_uniform distributions, ensure min > 0
            if (
                param_config.get("distribution") == "log_uniform"
                and min_val <= 0
            ):
                raise ValueError(
                    f"Invalid range for {param_name}: min must be greater than 0 for log_uniform distribution"
                )


@hydra.main(config_path="config", config_name="sweep", version_base="1.2")
def launch_sweep(cfg: DictConfig):
    # Ensure wandb is logged in
    check_wandb_login()

    # Convert OmegaConf to regular Python dict
    sweep_config = OmegaConf.to_container(cfg, resolve=True)

    # Remove hydra-specific fields
    sweep_config.pop("wandb", None)
    sweep_config.pop("defaults", None)

    # Validate parameter ranges
    if "parameters" in sweep_config:
        validate_parameter_ranges(sweep_config["parameters"])

        # Create new parameters dict with experiment namespace
        new_params = {}
        for k, v in sweep_config["parameters"].items():
            if "distribution" in v:
                new_params[f"experiment.{k}"] = {
                    "min": float(v["min"]),
                    "max": float(v["max"]),
                    "distribution": v["distribution"],
                }
            else:
                new_params[f"experiment.{k}"] = v

        sweep_config["parameters"] = new_params

    print("Sweep configuration:")
    print(OmegaConf.to_yaml(sweep_config))

    # Initialize wandb sweep
    sweep_id = wandb.sweep(
        sweep=sweep_config,
        project=cfg.wandb.project,
        entity=cfg.wandb.get("entity", None),
    )

    print(f"Created sweep with ID: {sweep_id}")
    print("You can now run the sweep with:")
    print(f"wandb agent {cfg.wandb.entity}/{cfg.wandb.project}/{sweep_id}")


if __name__ == "__main__":
    launch_sweep()
