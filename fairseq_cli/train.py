#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Train a new model on one or across multiple GPUs.
"""

import argparse
import logging
import math
import os
import sys
import copy
import pdb
import json
from typing import Dict, Optional, Any, List, Tuple, Callable

# We need to setup root logger before importing any fairseq libraries.
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "DEBUG").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("fairseq_cli.train")
logger.setLevel(logging.DEBUG)

import numpy as np
import torch
from fairseq import (
    checkpoint_utils,
    options,
    quantization_utils,
    tasks,
    utils,
)
from fairseq.data import iterators, data_utils
from fairseq.data.plasma_utils import PlasmaStore
from fairseq.dataclass.configs import FairseqConfig
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.distributed import fsdp_enable_wrap, fsdp_wrap, utils as distributed_utils
from fairseq.file_io import PathManager
from fairseq.logging import meters, metrics, progress_bar
from fairseq.model_parallel.megatron_trainer import MegatronTrainer
from fairseq.trainer import Trainer
from omegaconf import DictConfig, OmegaConf


def main(cfg: FairseqConfig) -> None:
    if isinstance(cfg, argparse.Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)

    utils.import_user_module(cfg.common)

    if distributed_utils.is_master(cfg.distributed_training) and "job_logging_cfg" in cfg:
        # make hydra logging work with ddp (see # see https://github.com/facebookresearch/hydra/issues/1126)
        logging.config.dictConfig(OmegaConf.to_container(cfg.job_logging_cfg))

    assert (
        cfg.dataset.max_tokens is not None or cfg.dataset.batch_size is not None
    ), "Must specify batch size either with --max-tokens or --batch-size"
    metrics.reset()

    if cfg.common.log_file is not None:
        handler = logging.FileHandler(filename=cfg.common.log_file)
        logger.addHandler(handler)

    np.random.seed(cfg.common.seed)
    utils.set_torch_seed(cfg.common.seed)

    if distributed_utils.is_master(cfg.distributed_training):
        checkpoint_utils.verify_checkpoint_directory(cfg.checkpoint.save_dir)

    # Print args
    logger.info(cfg)

    if cfg.checkpoint.write_checkpoints_asynchronously:
        try:
            import iopath  # noqa: F401
        except ImportError:
            logging.exception(
                "Asynchronous checkpoint writing is specified but iopath is "
                "not installed: `pip install iopath`"
            )
            return

    # Setup task, e.g., translation, language modeling, etc.
    task = tasks.setup_task(cfg.task)

    # added by Eachan: we add the lang2id and id2lang into args (actually lang_token to id)
    if getattr(cfg.task, "LS_reorder", None) \
            or getattr(cfg.task, "LS_dropout", None)\
            or getattr(cfg.task, "LS_epoch", None):
        # assert type(cfg.model.lang_pairs) == type([])
        cfg.model.lang2id = {}
        cfg.model.id2lang = {}
        for lang_pair in cfg.model.lang_pairs.split(','):
            src_lang, tgt_lang = lang_pair.split('-')
            src_lang_token = '__{}__'.format(src_lang)
            tgt_lang_token = '__{}__'.format(tgt_lang)
            # 如何获取每个语言在词表中的id？
            dict = list(task.dicts.items())[0][1]

            cfg.model.lang2id[src_lang_token] = dict.index(src_lang_token)
            cfg.model.lang2id[tgt_lang_token] = dict.index(tgt_lang_token)

            cfg.model.id2lang[dict.index(src_lang_token)] = src_lang
            cfg.model.id2lang[dict.index(tgt_lang_token)] = tgt_lang
        cfg.task.lang2id = cfg.model.lang2id
        cfg.task.id2lang = cfg.model.id2lang

    assert cfg.criterion, "Please specify criterion to train a model"

    # Build model and criterion
    if cfg.distributed_training.ddp_backend == "fully_sharded":  # False
        with fsdp_enable_wrap(cfg.distributed_training):
            model = fsdp_wrap(task.build_model(cfg.model))
    else:
        model = task.build_model(cfg.model)
    criterion = task.build_criterion(cfg.criterion)
    logger.info(model)
    logger.info("task: {}".format(task.__class__.__name__))
    logger.info("model: {}".format(model.__class__.__name__))
    logger.info("criterion: {}".format(criterion.__class__.__name__))
    logger.info(
        "num. shared model params: {:,} (num. trained: {:,})".format(
            sum(p.numel() for p in model.parameters() if not getattr(p, "expert", False)),
            sum(p.numel() for p in model.parameters() if not getattr(p, "expert", False) and p.requires_grad)
        )
    )

    logger.info(
        "num. expert model params: {} (num. trained: {})".format(
            sum(p.numel() for p in model.parameters() if getattr(p, "expert", False)),
            sum(p.numel() for p in model.parameters() if getattr(p, "expert", False) and p.requires_grad),
        )
    )

    # Load valid dataset (we load training data below, based on the latest checkpoint)
    # We load the valid dataset AFTER building the model
    data_utils.raise_if_valid_subsets_unintentionally_ignored(cfg)
    if cfg.dataset.combine_valid_subsets:
        task.load_dataset("valid", combine=True, epoch=1)
    else:
        for valid_sub_split in cfg.dataset.valid_subset.split(","):
            task.load_dataset(valid_sub_split, combine=False, epoch=1)

    # (optionally) Configure quantization
    if cfg.common.quantization_config_path is not None:
        quantizer = quantization_utils.Quantizer(
            config_path=cfg.common.quantization_config_path,
            max_epoch=cfg.optimization.max_epoch,
            max_update=cfg.optimization.max_update,
        )
    else:
        quantizer = None

    # Build trainer
    if cfg.common.model_parallel_size == 1:
        trainer = Trainer(cfg, task, model, criterion, quantizer)
    else:
        trainer = MegatronTrainer(cfg, task, model, criterion)
    logger.info(
        "training on {} devices (GPUs/TPUs)".format(
            cfg.distributed_training.distributed_world_size
        )
    )
    logger.info(
        "max tokens per device = {} and max sentences per device = {}".format(
            cfg.dataset.max_tokens,
            cfg.dataset.batch_size,
        )
    )

    # Load the latest checkpoint if one is available and restore the
    # corresponding train iterator
    extra_state, epoch_itr = checkpoint_utils.load_checkpoint(
        cfg.checkpoint,
        trainer,
        # don't cache epoch iterators for sharded datasets
        disable_iterator_cache=task.has_sharded_data("train"),
    )

    # added by Eachan: init the LS_best_models  =======================Online Distillation for MNMT=====================
    logger.debug(f"online_distillation_MNMT: {getattr(cfg.task, 'online_distillation_MNMT', False)}")
    if getattr(cfg.task, "online_distillation_MNMT", False):
        init_LS_best_models(cfg, trainer, task, epoch_itr)

    if cfg.common.tpu:
        import torch_xla.core.xla_model as xm
        xm.rendezvous("load_checkpoint")  # wait for all workers

    max_epoch = cfg.optimization.max_epoch or math.inf
    lr = trainer.get_lr()

    train_meter = meters.StopwatchMeter()
    train_meter.start()

    while epoch_itr.next_epoch_idx <= max_epoch:
        if lr <= cfg.optimization.stop_min_lr:
            logger.info(
                f"stopping training because current learning rate ({lr}) is smaller "
                "than or equal to minimum learning rate "
                f"(--stop-min-lr={cfg.optimization.stop_min_lr})"
            )
            break

        # *********************************************************************************
        # train for one epoch
        valid_losses, should_stop = train(cfg, trainer, task, epoch_itr)
        if should_stop:
            break

        # only use first validation loss to update the learning rate
        lr = trainer.lr_step(epoch_itr.epoch, valid_losses[0])

        # Added by Eachan: test code
        epoch_itr = trainer.get_train_iterator(
            epoch_itr.next_epoch_idx,
            # sharded data: get train iterator for next epoch
            load_dataset=task.has_sharded_data("train"),
            # don't cache epoch iterators for sharded datasets
            disable_iterator_cache=task.has_sharded_data("train"),
        )
    train_meter.stop()
    logger.info("done training in {:.1f} seconds".format(train_meter.sum))

    # ioPath implementation to wait for all asynchronous file writes to complete.
    if cfg.checkpoint.write_checkpoints_asynchronously:
        logger.info(
            "ioPath PathManager waiting for all asynchronous checkpoint "
            "writes to finish."
        )
        PathManager.async_close()
        logger.info("ioPath PathManager finished waiting.")


def should_stop_early(cfg: DictConfig, valid_loss: float) -> bool:
    # skip check if no validation was done in the current epoch
    if valid_loss is None:
        return False
    if cfg.checkpoint.patience <= 0:
        return False

    def is_better(a, b):
        return a > b if cfg.checkpoint.maximize_best_checkpoint_metric else a < b

    prev_best = getattr(should_stop_early, "best", None)
    if prev_best is None or is_better(valid_loss, prev_best):
        should_stop_early.best = valid_loss
        should_stop_early.num_runs = 0
        return False
    else:
        should_stop_early.num_runs += 1
        if should_stop_early.num_runs >= cfg.checkpoint.patience:
            logger.info(
                "early stop since valid performance hasn't improved for last {} runs".format(
                    cfg.checkpoint.patience
                )
            )
            return True
        else:
            return False


@metrics.aggregate("train")
def train(
    cfg: DictConfig, trainer: Trainer, task: tasks.FairseqTask, epoch_itr
) -> Tuple[List[Optional[float]], bool]:
    """Train the model for one epoch and return validation losses."""
    # Initialize data iterator
    # epoch_itr: iterators.EpochBatchIterator
    itr = epoch_itr.next_epoch_itr(
        fix_batches_to_gpus=cfg.distributed_training.fix_batches_to_gpus,
        shuffle=(epoch_itr.next_epoch_idx > cfg.dataset.curriculum),
    )
    # 这个itr是个iterators.CountingIterator
    update_freq = (
        cfg.optimization.update_freq[epoch_itr.epoch - 1]
        if epoch_itr.epoch <= len(cfg.optimization.update_freq)
        else cfg.optimization.update_freq[-1]
    )
    # 这个GroupIterator的作用好像就是将原来的一次返回一个epoch变成一次返回一个list，
    # 这个list中包含update_freq个epoch的数据
    itr = iterators.GroupedIterator(itr, update_freq)
    if cfg.common.tpu:
        itr = utils.tpu_data_loader(itr)
    progress = progress_bar.progress_bar(
        itr,
        log_format=cfg.common.log_format,
        log_file=cfg.common.log_file,
        log_interval=cfg.common.log_interval,
        epoch=epoch_itr.epoch,
        tensorboard_logdir=(
            cfg.common.tensorboard_logdir
            if distributed_utils.is_master(cfg.distributed_training)
            else None
        ),
        default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
        wandb_project=(
            cfg.common.wandb_project
            if distributed_utils.is_master(cfg.distributed_training)
            else None
        ),
        wandb_run_name=os.environ.get(
            "WANDB_NAME", os.path.basename(cfg.checkpoint.save_dir)
        ),
        azureml_logging=(
            cfg.common.azureml_logging
            if distributed_utils.is_master(cfg.distributed_training)
            else False
        ),
    )
    progress.update_config(_flatten_config(cfg))

    trainer.begin_epoch(epoch_itr.epoch)

    valid_subsets = cfg.dataset.valid_subset.split(",")
    should_stop = False
    num_updates = trainer.get_num_updates()
    logger.info("Start iterating over samples")
    for i, samples in enumerate(progress):

        with metrics.aggregate("train_inner"), torch.autograd.profiler.record_function(
            "train_step-%d" % i
        ):
            # =====================key code line================
            log_output = trainer.train_step(samples, epoch=epoch_itr.epoch, step=i)

        if log_output is not None:  # not OOM, overflow, ...
            # log mid-epoch stats
            num_updates = trainer.get_num_updates()
            if num_updates % cfg.common.log_interval == 0:
                stats = get_training_stats(metrics.get_smoothed_values("train_inner"))
                progress.log(stats, tag="train_inner", step=num_updates)

                # reset mid-epoch stats after each log interval
                # the end-of-epoch stats will still be preserved
                metrics.reset_meters("train_inner")

        end_of_epoch = not itr.has_next()

        # test code:
        # if i > 3:
        #     end_of_epoch = True

        # valid
        valid_losses, should_stop = validate_and_save(
            cfg, trainer, task, epoch_itr, valid_subsets, end_of_epoch
        )

        # test code:
        # if i > 3:
        #     break

        if should_stop:
            break

    # log end-of-epoch stats
    logger.info("end of epoch {} (average epoch stats below)".format(epoch_itr.epoch))
    stats = get_training_stats(metrics.get_smoothed_values("train"))
    progress.print(stats, tag="train", step=num_updates)

    # reset epoch-level meters
    metrics.reset_meters("train")
    return valid_losses, should_stop


def _flatten_config(cfg: DictConfig):
    config = OmegaConf.to_container(cfg)
    # remove any legacy Namespaces and replace with a single "args"
    namespace = None
    for k, v in list(config.items()):
        if isinstance(v, argparse.Namespace):
            namespace = v
            del config[k]
    if namespace is not None:
        config["args"] = vars(namespace)
    return config


def validate_and_save(
    cfg: DictConfig,
    trainer: Trainer,
    task: tasks.FairseqTask,
    epoch_itr,
    valid_subsets: List[str],
    end_of_epoch: bool,
) -> Tuple[List[Optional[float]], bool]:
    num_updates = trainer.get_num_updates()
    max_update = cfg.optimization.max_update or math.inf

    # Stopping conditions (and an additional one based on validation loss later
    # on)
    should_stop = False
    if num_updates >= max_update:
        should_stop = True
        logger.info(
            f"Stopping training due to "
            f"num_updates: {num_updates} >= max_update: {max_update}"
        )

    training_time_hours = trainer.cumulative_training_time() / (60 * 60)
    if (
        cfg.optimization.stop_time_hours > 0
        and training_time_hours > cfg.optimization.stop_time_hours
    ):
        should_stop = True
        logger.info(
            f"Stopping training due to "
            f"cumulative_training_time: {training_time_hours} > "
            f"stop_time_hours: {cfg.optimization.stop_time_hours} hour(s)"
        )

    do_save = (
        (end_of_epoch and epoch_itr.epoch % cfg.checkpoint.save_interval == 0)
        or should_stop
        or (
            cfg.checkpoint.save_interval_updates > 0
            and num_updates > 0
            and num_updates % cfg.checkpoint.save_interval_updates == 0
            and num_updates >= cfg.dataset.validate_after_updates
        )
    )

    do_validate = (
        (not end_of_epoch and do_save)  # validate during mid-epoch saves
        or (end_of_epoch and epoch_itr.epoch % cfg.dataset.validate_interval == 0)
        or should_stop
        or (
            cfg.dataset.validate_interval_updates > 0
            and num_updates > 0
            and num_updates % cfg.dataset.validate_interval_updates == 0
        )
    ) and not cfg.dataset.disable_validation and num_updates >= cfg.dataset.validate_after_updates

    # Validate
    valid_losses = [None]
    if do_validate:
        if getattr(cfg.task, "LS_epoch", None):
            valid_losses, LS_valid_loss = validate(cfg, trainer, task, epoch_itr, valid_subsets)

            if getattr(cfg.task, "online_distillation_MNMT", False):
                update_LS_best_models(cfg, trainer, task, LS_valid_loss, epoch_itr, end_of_epoch, valid_losses[0])

        else:
            valid_losses = validate(cfg, trainer, task, epoch_itr, valid_subsets)

    should_stop |= should_stop_early(cfg, valid_losses[0])

    # Save checkpoint
    if do_save or should_stop:
        if getattr(cfg.task, "LS_epoch", None):
            checkpoint_utils.save_checkpoint(
                cfg.checkpoint, trainer, epoch_itr, valid_losses[0], LS_valid_loss=LS_valid_loss
            )
        else:
            checkpoint_utils.save_checkpoint(
                cfg.checkpoint, trainer, epoch_itr, valid_losses[0]
            )

    return valid_losses, should_stop


def get_training_stats(stats: Dict[str, Any]) -> Dict[str, Any]:
    stats["wall"] = round(metrics.get_meter("default", "wall").elapsed_time, 0)
    return stats


def validate(
    cfg: DictConfig,
    trainer: Trainer,
    task: tasks.FairseqTask,
    epoch_itr,
    subsets: List[str],
) -> List[Optional[float]]:
    """Evaluate the model on the validation set(s) and return the losses."""

    if cfg.dataset.fixed_validation_seed is not None:
        # set fixed seed for every validation
        utils.set_torch_seed(cfg.dataset.fixed_validation_seed)

    trainer.begin_valid_epoch(epoch_itr.epoch)

    # Added  by Eachan: create a dict (named 'LS_valid_losses') to record valid loss cumulative sum and cumulative size
    #     for each language.
    if getattr(cfg.task, "LS_epoch", None):
        LS_valid_cumulative_loss = {}
        for lang_pair in cfg.model.lang_pairs.split(','):
            src_lang, tgt_lang = lang_pair.split('-')
            LS_valid_cumulative_loss[src_lang + '-' + tgt_lang] = {
                'cumulative_loss': 0.0,
                'cumulative_size': 0
            }

    valid_losses = []
    for subset in subsets:  # ['valid']
        logger.info('begin validation on "{}" subset'.format(subset))

        # Initialize data iterator
        itr = trainer.get_valid_iterator(subset).next_epoch_itr(
            shuffle=False, set_dataset_epoch=False  # use a fixed valid set
        )
        if cfg.common.tpu:
            itr = utils.tpu_data_loader(itr)
        progress = progress_bar.progress_bar(
            itr,
            log_format=cfg.common.log_format,
            log_interval=cfg.common.log_interval,
            epoch=epoch_itr.epoch,
            prefix=f"valid on '{subset}' subset",
            tensorboard_logdir=(
                cfg.common.tensorboard_logdir
                if distributed_utils.is_master(cfg.distributed_training)
                else None
            ),
            default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
            wandb_project=(
                cfg.common.wandb_project
                if distributed_utils.is_master(cfg.distributed_training)
                else None
            ),
            wandb_run_name=os.environ.get(
                "WANDB_NAME", os.path.basename(cfg.checkpoint.save_dir)
            ),
        )

        # create a new root metrics aggregator so validation metrics
        # don't pollute other aggregators (e.g., train meters)
        with metrics.aggregate(new_root=True) as agg:
            for i, sample in enumerate(progress):
                if cfg.dataset.max_valid_steps is not None and i > cfg.dataset.max_valid_steps:
                    break

                # modified by eachan: receive three variables.
                if getattr(cfg.task, "LS_epoch", None) and (sample is not None and len(sample) > 0):
                    logging_output, loss_, sample_size = trainer.valid_step(sample)

                    # add loss and size into LS_valid_cumulative_loss:
                    for lang_pair in cfg.model.lang_pairs.split(','):
                        src_lang, tgt_lang = lang_pair.split('-')
                        LS_valid_cumulative_loss[src_lang + '-' + tgt_lang]["cumulative_loss"] += round(
                            logging_output[src_lang + '-' + tgt_lang + '_loss'].item() / math.log(2), 3)
                        LS_valid_cumulative_loss[src_lang + '-' + tgt_lang]["cumulative_size"] += logging_output[
                            src_lang + '-' + tgt_lang + '_size']
                else:
                    trainer.valid_step(sample)


        # added by eachan: calculate the average valid loss for each language.
        if getattr(cfg.task, "LS_epoch", None):
            # print("eachan print:")
            # print(agg.get_smoothed_values())
            LS_valid_loss = {}
            for lang, item in LS_valid_cumulative_loss.items():

                LS_valid_loss[lang] = item['cumulative_loss'] / item['cumulative_size']

        # log validation stats
        stats = get_valid_stats(cfg, trainer, agg.get_smoothed_values())

        if hasattr(task, "post_validate"):
            task.post_validate(trainer.get_model(), stats, agg)

        progress.print(stats, tag=subset, step=trainer.get_num_updates())

        valid_losses.append(stats[cfg.checkpoint.best_checkpoint_metric])

    # Added by Eachan:
    if getattr(cfg.task, "LS_epoch", None):
        return valid_losses, LS_valid_loss
    else:
        return valid_losses


def get_valid_stats(
    cfg: DictConfig, trainer: Trainer, stats: Dict[str, Any]
) -> Dict[str, Any]:
    stats["num_updates"] = trainer.get_num_updates()
    if hasattr(checkpoint_utils.save_checkpoint, "best"):
        key = "best_{0}".format(cfg.checkpoint.best_checkpoint_metric)
        best_function = max if cfg.checkpoint.maximize_best_checkpoint_metric else min
        stats[key] = best_function(
            checkpoint_utils.save_checkpoint.best,
            stats[cfg.checkpoint.best_checkpoint_metric],
        )
    return stats


def init_LS_best_models(cfg, trainer, task, epoch_iter):
    """
    At the beginning of training, we
    :param cfg:
    :param trainer:
    :param task:
    :param epoch_iter:
    :return:
    """
    LS_best_models = {}

    # 1. Check the Mode is 'M2O' or 'O2M' or 'M2M':
    src_lang_set = set()
    tgt_lang_set = set()
    for lang_pair in cfg.model.lang_pairs.split(','):
        src_lang, tgt_lang = lang_pair.split('-')
        src_lang_set.add(src_lang)
        tgt_lang_set.add(tgt_lang)
    if len(src_lang_set) > 1 and len(tgt_lang_set) == 1:
        MNMT_mode = 'M2O'
    elif len(src_lang_set) == 1 and len(tgt_lang_set) > 1:
        MNMT_mode = 'O2M'
    else:
        MNMT_mode = 'M2M'
    cfg.task.MNMT_mode = MNMT_mode
    cfg.task.src_lang_set = src_lang_set
    cfg.task.tgt_lang_set = tgt_lang_set
    logger.debug(f"cfg.task.MNMT_mode: {cfg.task.MNMT_mode}")
    logger.debug(f"cfg.task.src_lang_set: {cfg.task.src_lang_set}")
    logger.debug(f"cfg.task.tgt_lang_set: {cfg.task.tgt_lang_set}")

    # 2. if 'M2O': load language-specific best model for each source language;
    #    if 'O2M' or 'M2M': load language-specific best model for each target language
    # logger.debug(f"whether to use EMA: {cfg.task.EMA}")
    if cfg.task.EMA:
        initial_copy_model = task.build_model(cfg.model)
        initial_copy_model.load_state_dict(trainer.model.state_dict())
        initial_copy_model.to(device=trainer.device)
        initial_copy_model.half()
        for p in initial_copy_model.parameters():
            p.requires_grad = False
        initial_copy_model.eval()
        trainer.EMA_model = initial_copy_model

    # Read loss curve (history) of each language.
    LS_valid_loss_history = None
    if epoch_iter.epoch > 1:
        f = open(cfg.checkpoint.save_dir + '/LS_valid_loss_history.json')
        LS_valid_loss_history = json.load(f)
        f.close()

    # Load the best checkpoint up to the current epoch for each source language
    if MNMT_mode == 'M2O':
        for src_lang in src_lang_set:
            LS_best_epoch = -1
            LS_best_loss = 10000000
            LS_prev_best_model = None

            if LS_valid_loss_history is not None:
                # Find the epoch of language-specific best checkpoint
                LS_best_loss = min(LS_valid_loss_history[f"{src_lang}-{list(tgt_lang_set)[0]}"])
                LS_best_epoch = 1 + LS_valid_loss_history[f"{src_lang}-{list(tgt_lang_set)[0]}"].index(LS_best_loss)
                # for epoch, loss in enumerate(LS_valid_loss_history[f"{src_lang}-{list(tgt_lang_set)[0]}"]):
                #     if loss < LS_best_loss:
                #         LS_best_loss = loss
                #         LS_best_epoch = epoch + 1

                # load language-specific best checkpoint
                LS_prev_best_model = task.build_model(cfg.model)
                if getattr(cfg.task, "single_teacher_MNMT"):
                    # load overall best checkpoint
                    tmp_state = torch.load(f"{cfg.checkpoint.save_dir}/checkpoint_best.pt")
                else:
                    # load language-specific best checkpoint
                    tmp_state = torch.load(f"{cfg.checkpoint.save_dir}/checkpoint{LS_best_epoch}.pt")
                LS_prev_best_model.load_state_dict(tmp_state["model"], strict=True, model_cfg=cfg.model)
                del tmp_state
                LS_prev_best_model.half()
                for p in LS_prev_best_model.parameters():  # No parameter updating
                    p.requires_grad = False
                LS_prev_best_model.to(device=trainer.device)
                LS_prev_best_model.eval()

            LS_best_models[src_lang] = {
                'epoch': LS_best_epoch,
                'valid_loss': LS_best_loss,
                'model': LS_prev_best_model
            }
    else:
        for tgt_lang in tgt_lang_set:
            # read previous best model for current language
            LS_best_epoch = -1
            LS_best_loss = 10000000
            LS_prev_best_model = None

            if LS_valid_loss_history is not None:
                LS_best_loss = min(LS_valid_loss_history[f"{list(src_lang_set)[0]}-{tgt_lang}"])
                LS_best_epoch = 1 + LS_valid_loss_history[f"{list(src_lang_set)[0]}-{tgt_lang}"].index(LS_best_loss)
                # for epoch, loss in enumerate(LS_valid_loss_history[f"{list(src_lang_set)[0]}-{tgt_lang}"]):
                #     if loss < LS_best_loss:
                #         LS_best_loss = loss
                #         LS_best_epoch = epoch + 1

                LS_prev_best_model = task.build_model(cfg.model)
                if getattr(cfg.task, "single_teacher_MNMT"):
                    # load overall best checkpoint
                    tmp_state = torch.load(f"{cfg.checkpoint.save_dir}/checkpoint_best.pt")
                else:
                    # load language-specific best checkpoint
                    tmp_state = torch.load(f"{cfg.checkpoint.save_dir}/checkpoint{LS_best_epoch}.pt")
                LS_prev_best_model.load_state_dict(tmp_state["model"], strict=True, model_cfg=cfg.model)
                del tmp_state
                LS_prev_best_model.half()
                for p in LS_prev_best_model.parameters():
                    p.requires_grad = False
                LS_prev_best_model.to(device=trainer.device)
                LS_prev_best_model.eval()

            LS_best_models[tgt_lang] = {
                'epoch': LS_best_epoch,
                'valid_loss': LS_best_loss,  # this is a very big value
                'model': LS_prev_best_model  # initial_copy_model
            }

        os.system("nvidia-smi")

    # 3. add the LS_best_models into trainer and task
    trainer.LS_best_models = LS_best_models
    task.LS_best_models = LS_best_models
    logger.debug(f"LS_best_models: {LS_best_models}")


def update_LS_best_models(cfg, trainer, task, LS_valid_loss, epoch_itr, end_of_epoch, valid_loss):
    """
    update Language-specific best checkpoint
    :param cfg:
    :param trainer:
    :param task:
    :param LS_valid_loss: {'eng-aze': 3.5, 'eng-tur': 4.2, ....}
    :param epoch_itr:
    :param end_of_epoch:
    :param valid_loss:
    :return:
    """
    if not end_of_epoch:
        return

    """
    Added by Eachan: update the LS_best_models
    I meet a embarrassing thing that:
      The key of LS_valid_loss is language-pair. However, the key of LS_best_models is language...
    Therefore, I have to do a transformation.
    """
    LS_best_models = trainer.LS_best_models
    current_model_copy = None

    # 1. update best checkpoint for each language
    for lang in LS_best_models.keys():
        # 1.1 get the language-pair and corresponding valid loss
        if cfg.task.MNMT_mode == "M2O":
            lang_pair = lang + '-' + list(cfg.task.tgt_lang_set)[0]
            valid_loss_for_lang = LS_valid_loss[lang_pair]
        elif cfg.task.MNMT_mode == 'O2M':
            lang_pair = list(cfg.task.src_lang_set)[0] + '-' + lang
            valid_loss_for_lang = LS_valid_loss[lang_pair]
        else:
            """
            For M2M, we save best checkpoint for each target language.
              Therefore, the keys of LS_best_models are target language.
              We use the average valid loss over all source languages to the certain target language
              as the valid loss of each target language.
            """
            lang_pairs = [i for i in cfg.model.lang_pairs.split(',') if i.split('-')[1] == lang]
            valid_loss_for_lang = sum([LS_valid_loss[i] for i in lang_pairs]) / len(lang_pairs)
            # valid_loss_for_lang = sum([LS_valid_loss[src_lang + '-' + lang]
            #                            for src_lang in cfg.task.src_lang_set]) \
            #                       / len(cfg.task.src_lang_set)

        # 1.2 update the language-specific teacher model
        if getattr(cfg.task, "single_teacher_MNMT"):
            logger.debug(f"Single Teacher Distillation | "
                         f"prev_best:{getattr(checkpoint_utils.save_checkpoint, 'best', valid_loss)},"
                         f"current: {valid_loss}")

            update_switch = epoch_itr.epoch == 1 or valid_loss < getattr(checkpoint_utils.save_checkpoint, "best",
                                                                         valid_loss)
        else:
            update_switch = epoch_itr.epoch == 1 or LS_best_models[lang]['valid_loss'] > valid_loss_for_lang
        if update_switch:
            LS_best_models[lang]['valid_loss'] = valid_loss_for_lang
            LS_best_models[lang]['epoch'] = epoch_itr.epoch

            # copy the current model and freeze:
            if current_model_copy is None:
                # current_model_copy = copy.deepcopy(trainer.model)
                """
                I tried to use copy.deepcopy, but meet a exception when I use multi-GPU.
                Therefore, we create a new model, and load the original model's parameters.
                """
                current_model_copy = task.build_model(cfg.model)

                if cfg.task.EMA:
                    current_model_copy.load_state_dict(trainer.EMA_model.state_dict())
                else:
                    current_model_copy.load_state_dict(trainer.model.state_dict())
                current_model_copy.to(device=trainer.device)
                current_model_copy.half()

                for p in current_model_copy.parameters():
                    p.requires_grad = False
                current_model_copy.eval()

            LS_best_models[lang]['model'] = current_model_copy

            logger.debug(f"best {lang} model achieve best.")

    torch.cuda.empty_cache()

    # 2. save LS_valid_loss_history in "LS_valid_loss_history.json"
    if trainer.is_data_parallel_master:
        # 2.1 read LS_valid_loss.json as LS_valid_loss_history
        LS_valid_loss_history = {}
        for lang_pair in LS_valid_loss.keys():
            LS_valid_loss_history[lang_pair] = []
        LS_valid_loss_history["all"] = []

        if epoch_itr.epoch > 1:
            f = open(cfg.checkpoint.save_dir + "/LS_valid_loss_history.json")
            LS_valid_loss_history = json.load(f)
            f.close()

        for lang_pair in LS_valid_loss_history.keys():
            assert len(LS_valid_loss_history[lang_pair]) == epoch_itr.epoch - 1

        # 2.2 update LS_valid_loss_history
        for lang_pair, loss in LS_valid_loss.items():
            if isinstance(loss, torch.Tensor):
                LS_valid_loss_history[lang_pair].append(loss.item())
            else:
                LS_valid_loss_history[lang_pair].append(loss)
        LS_valid_loss_history["all"].append(valid_loss)

        # 2.3 write into LS_valid_loss.json
        with open(cfg.checkpoint.save_dir + "/LS_valid_loss_history.json", 'w+') as f:
            json.dump(LS_valid_loss_history, f)


def cli_main(
    modify_parser: Optional[Callable[[argparse.ArgumentParser], None]] = None
) -> None:
    parser = options.get_training_parser()
    args = options.parse_args_and_arch(parser, modify_parser=modify_parser)

    cfg = convert_namespace_to_omegaconf(args)

    if cfg.common.use_plasma_view:
        server = PlasmaStore(path=cfg.common.plasma_path)
        logger.info(f"Started plasma server pid {server.server.pid} {cfg.common.plasma_path}")

    if args.profile:
        with torch.cuda.profiler.profile():
            with torch.autograd.profiler.emit_nvtx():
                distributed_utils.call_main(cfg, main)
    else:
        distributed_utils.call_main(cfg, main)

    # if cfg.common.use_plasma_view:
    #     server.server.kill()


if __name__ == "__main__":
    cli_main()
