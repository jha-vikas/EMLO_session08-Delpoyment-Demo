from typing import Tuple, Dict
import mlflow, os
import lightning as L
import torch
import hydra
from omegaconf import DictConfig

import mlflow.pytorch
from mlflow import MlflowClient
from lightning.pytorch.loggers.mlflow import MLFlowLogger

from gold import utils

from pathlib import Path
import os

log = utils.get_pylogger(__name__)


@utils.task_wrapper
def train(cfg: DictConfig) -> Tuple[dict, dict]:
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: L.LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: L.LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating loggers...")
    logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = utils.instantiate_callbacks(cfg.get("callbacks"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: L.Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "trainer": trainer,
        "callbacks": callbacks,
        "logger": logger,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    if cfg.get("compile"):
        model = torch.compile(model)

    ckpt_path = None
    ckpt_fol = cfg.get("ckpt_fol")
    if ckpt_fol:
        try:
            ckpt_path = max(Path(ckpt_fol).rglob("*.ckpt"), key=os.path.getctime)
        except:
            print("No checkpoint found. Continuing with default staring weights.")

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

    train_metrics = trainer.callback_metrics

    log.info("Scripting Model ...")

    scripted_model = model.to_torchscript(method="script")
    torch.jit.save(scripted_model, f"{cfg.model_ckpt_path}")

    log.info(f"Saving traced model to {cfg.model_ckpt_path}")


    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = ""
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")

        for logger_ in logger:
            if isinstance(logger_, MLFlowLogger):
                ckpt = torch.load(ckpt_path)
                model.load_state_dict(ckpt["state_dict"])
                os.environ['MLFLOW_RUN_ID'] = logger_.run_id
                os.environ['MLFLOW_EXPERIMENT_ID'] = logger_.experiment_id
                os.environ['MLFLOW_EXPERIMENT_NAME'] = logger_._experiment_name
                os.environ['MLFLOW_TRACKING_URI'] = logger_._tracking_uri
                mlflow.pytorch.log_model(model, "model")
                break


    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig):
    # train the model
    # print(cfg)
    metric_dict, _ = train(cfg)

    # this will be used by hydra later for optimization
    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = utils.get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
