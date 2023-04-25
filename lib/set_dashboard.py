import wandb
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint


def set_dashboard(config, task="MODEL"):
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
            # vit
            "HiddenDim": config["VIT_SETTINGS"]["HiddenDim"],
            "PatchSize": config["VIT_SETTINGS"]["PatchSize"],
            "NumLayer": config["VIT_SETTINGS"]["NumLayer"],
            "NumHeads": config["VIT_SETTINGS"]["NumHeads"],
            "MlpDim": config["VIT_SETTINGS"]["MlpDim"],

        },
    )
    wandb_callbacks = [
        WandbMetricsLogger(log_freq=1),
        # WandbModelCheckpoint(filepath="tmp/my_model_{epoch:01d}"),
    ]
    return [wandb_callbacks], wandb


def set_dashboard_distiller(config, param, run_id, id, task="MODEL"):
    wandb.login()
    wandb.init(
        # Set entity to specify your username or team name
        # ex: entity="carey",
        # Set the project where this run will be logged
        project="IDS_Transformers",
        # Track hyperparameters and run metadata
        config={
            # "learning_rate": 0.02,
            "Dataset": config["SETTINGS"]["Dataset"],
            "Model": config["DISTILLATION"]["model"],

            "Patience": config["MODEL"]["Patience"],
            "Epochs": config["DISTILLATION"]["Epochs"],
            # vit
            "LAB": config["DISTILLATION"]["LAB"],
            "ALPHA": param["A"],
            "Temp": param["T"],
            "kernel": param["kernel"],
            "filter": param["filter"],
            "filter2": param["filter2"],
            "batch": param["batch"],
            "dropout1": param["dropout1"],
            "dropout2": param["dropout2"],
            "learning_rate": param["learning_rate"],
            "RUN_ID": run_id,
            "ID": id,

        },
    )
    wandb_callbacks = [
        WandbMetricsLogger(log_freq=1),
        # WandbModelCheckpoint(filepath="tmp/my_model_{epoch:01d}"),
    ]
    return [wandb_callbacks], wandb
