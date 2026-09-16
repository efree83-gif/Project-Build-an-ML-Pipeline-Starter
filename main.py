import json
import mlflow
import tempfile
import os
import wandb
import hydra
from omegaconf import DictConfig

_steps = [
    "download",
    "basic_cleaning",
    "data_check",
    "data_split",
    "train_random_forest",
    # NOTE: We do not include this in the steps so it is not run by mistake.
    # You first need to promote a model export to "prod" before you can run this,
    # then you need to run this step explicitly
#    "test_regression_model"
]


# This automatically reads in the configuration
@hydra.main(version_base=None, config_name='config', config_path='.')
def go(config: DictConfig):

    # Setup the wandb experiment. All runs will be grouped under this name
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # Ensure Python can locate wandb_utils inside the components directory for all steps
    os.environ["PYTHONPATH"] = os.path.abspath("components")

    # Steps to execute
    steps_par = config['main']['steps']
    active_steps = steps_par.split(",") if steps_par != "all" else _steps

    # Move to a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:

        if "download" in active_steps:
            # Download file and load in W&B
            _ = mlflow.run(
                f"{config['main']['components_repository']}/get_data",
                "main",
                env_manager="local",  # <-- Changed from "conda" to "local"
                parameters={
                    "sample": config["etl"]["sample"],
                    "artifact_name": "sample.csv",
                    "artifact_type": "raw_data",
                    "artifact_description": "Raw_file_as_downloaded"
                },
            )

        if "basic_cleaning" in active_steps:
            _ = mlflow.run(
                "src/basic_cleaning",
                "main",
                env_manager="local",
                parameters={
                    "input_artifact": "sample.csv:latest",
                    "output_artifact": "clean_sample.csv",
                    "output_type": "clean_sample",
                    "output_description": "Cleaned_data_sample",
                    "min_price": config["etl"]["min_price"],
                    "max_price": config["etl"]["max_price"]
                },
            )

        if "data_check" in active_steps:
            _ = mlflow.run(
                "src/data_check",
                "main",
                env_manager="local",
                parameters={
                    "csv": "clean_sample.csv:latest",
                    "ref": "clean_sample.csv:reference",
                    "kl_threshold": config["data_check"]["kl_threshold"],
                    "min_price": config["etl"]["min_price"],
                    "max_price": config["etl"]["max_price"],
                },
            )

        if "data_split" in active_steps:
            os.environ["PYTHONPATH"] = os.path.abspath("components")  # <-- Use absolute path here
            _ = mlflow.run(
                f"{config['main']['components_repository']}/train_val_test_split",
                "main",
                env_manager="local",
                parameters={
                    "input": "clean_sample.csv:latest",
                    "test_size": config["modeling"]["test_size"],
                    "random_seed": config["modeling"]["random_seed"],
                    "stratify_by": config["modeling"]["stratify_by"]
                },
            )

        if "train_random_forest" in active_steps:
            # Write rf_config.json directly inside the component folder to avoid space-in-path errors on Windows
            rf_config_path = os.path.join("src", "train_random_forest", "rf_config.json")
            with open(rf_config_path, "w+") as fp:
                json.dump(dict(config["modeling"]["random_forest"].items()), fp)

            _ = mlflow.run(
                "src/train_random_forest",
                "main",
                env_manager="local",
                parameters={
                    "trainval_artifact": "trainval_data.csv:latest",
                    "val_size": config["modeling"]["val_size"],
                    "random_seed": config["modeling"]["random_seed"],
                    "stratify_by": config["modeling"]["stratify_by"],
                    "max_tfidf_features": config["modeling"]["max_tfidf_features"],
                    "output_artifact": "model_export",
                    "rf_config": "rf_config.json",
                },
            )

        if "test_regression_model" in active_steps:
            # Ensure Python can locate wandb_utils inside the components directory
            os.environ["PYTHONPATH"] = os.path.abspath("components")

            _ = mlflow.run(
                f"{config['main']['components_repository']}/test_regression_model",
                "main",
                env_manager="local",
                parameters={
                    "mlflow_model": "model_export:prod",
                    "test_dataset": "test_data.csv:latest",
                },
            )


if __name__ == "__main__":
    go()
    