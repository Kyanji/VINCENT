import wandb
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint


def set_dashboard(config):
    wandb.login()
    wandb.init(
        # Set entity to specify your username or team name
        # ex: entity="carey",
        # Set the project where this run will be logged
        project="IDS_Transformers",
        # Track hyperparameters and run metadata
        config={
            # "learning_rate": 0.02,
            "UseRGBEncoding": config["SETTINGS"]["UseRGBEncoding"],
            "Dataset": config["SETTINGS"]["Dataset"],
            "UseScale0_1": config["SETTINGS"]["UseScale0_1"],
            "UseScale-1_1": config["SETTINGS"]["UseScale-1_1"],
            "Resize": config["SETTINGS"]["Resize"],
            "ResizeShape": config["SETTINGS"]["ResizeShape"],
            "Model": config["MODEL"]["Model"],
            "Lr": config["MODEL"]["Lr"],
            "Decay": config["MODEL"]["Decay"],
            "EarlyStop": config["MODEL"]["EarlyStop"],
            "Patience": config["MODEL"]["Patience"],
            "BatchSize": config["MODEL"]["BatchSize"],
            "Epochs": config["MODEL"]["Epochs"],

        },
    )
    wandb_callbacks = [
        WandbMetricsLogger(log_freq=1),
        # WandbModelCheckpoint(filepath="tmp/my_model_{epoch:01d}"),
    ]
    return [wandb_callbacks], wandb
