import os
import wandb


def initialize_wandb_logger(**configs):
    os.environ["WANDB_API_KEY"] = configs["wandb_api_key"]
    wandb.init(
        project=configs["wandb_project_name"],
        name=configs["results_file_name"],
    )
    # log configurations
    wandb.config.update(configs)
