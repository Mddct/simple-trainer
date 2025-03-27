import abc
import os
import time
from contextlib import nullcontext

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.multiprocessing as mp
import torch.nn as nn
from torch.distributed.checkpoint.state_dict import (get_state_dict,
                                                     set_state_dict)
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from utils import if_print_model, init_distributed, wrap_cuda_model
from writer import create_default_writer

import torch.distributed as dist


class TrainState(Stateful):
    """This is a useful wrapper for checkpointing the Application State. Since this object is compliant
    with the Stateful protocol, DCP will automatically call state_dict/load_stat_dict as needed in the
    dcp.save/load APIs.

    Note: We take advantage of this wrapper to hande calling distributed state dict methods on the model
    and optimizer.
    """

    def __init__(self, model: torch.nn.Module, config, optimizer=None):
        self.model = wrap_cuda_model(model, config)
        self.optimizer = optimizer
        self._step = 0

    def state_dict(self):
        # this line automatically manages FSDP FQN's, as well as sets the default state dict type to FSDP.SHARDED_STATE_DICT
        model_state_dict, optimizer_state_dict = get_state_dict(
            self.model, self.optimizer)
        return {"model": model_state_dict, "optim": optimizer_state_dict}

    def load_state_dict(self, state_dict):
        # sets our state dicts on the model and optimizer, now that we've loaded
        set_state_dict(self.model,
                       self.optimizer,
                       model_state_dict=state_dict["model"],
                       optim_state_dict=state_dict["optim"])

    @classmethod
    def train_step(self, batch: dict):
        pass

    @classmethod
    def eval_step(self, batch: dict):
        pass

    @property
    def step(self):
        return self._step

    @classmethod
    def reduce_train_metrics(self, metrics: dict):
        pass

    @classmethod
    def reduce_eval_metrics(self, metrics: dict):
        pass

    @property
    def model(self):
        return self._model


class CheckpointManager:

    def __init__(self,
                 checkpoint_dir,
                 collection,
                 async_checkpoint: bool = False,
                 max_saves=None):
        self.async_checkpoint = async_checkpoint
        self.checkpoint_future = None
        self.checkpoint_dir = checkpoint_dir
        self.collection = collection

    def save(self, model: TrainState):

        step = model.step
        dir = f"{self.checkpoint_dir}/step_{step}"

        state_dict = {self.collection: model}
        if self.async_checkpoint and self.checkpoint_future is not None:
            self.checkpoint_future.result()
        if self.async_checkpoint:
            self.checkpoint_future = dcp.async_save(state_dict,
                                                    checkpoint_id=dir)
        else:
            dcp.save(state_dict, checkpoint_id=dir)

    def restore(self, model: TrainState, checkpoint_dir):
        state_dict = {self.collection: model}
        dcp.load(state_dict=state_dict, checkpoint_id=checkpoint_dir)


class Trainer:

    def __init__(self,
                 config,
                 state: TrainState,
                 dataloader_or_dataset,
                 eval_datalader_or_dataset=None):
        self.config = config
        self.world_size, self.rank, self.local_rank = init_distributed(config)
        self.writer = create_default_writer(config.tensorboard_dir,
                                            just_logging=self.rank != 0,
                                            asynchronous=True)
        self.train_iter = iter(dataloader_or_dataset)
        self.eval_iter = None
        if eval_datalader_or_dataset is not None:
            self.eval_iter = iter(eval_datalader_or_dataset)
        self.train_state = state
        self.checkpoint_manager = CheckpointManager(
            config.model_dir,
            "model",
            config.async_checkpoint,
        )

    def train(self):
        if self.config.checkpoint_dir != '':
            self.checkpoint_manager.restore(self.train_state,
                                            self.config.checkpoint_dir)
        steps_offset = self.train_state.step
        if steps_offset != 0:
            # TODO: skip first n in train iter
            pass
        train_metrics_last_t = time.time()

        train_metrics = []

        self.train_state.model.train()
        for step, batch in zip(range(steps_offset, self.config.max_steps),
                               self.train_iter):
            if (step + 1) % self.config.accum_grad == 0:
                self.train_state.model.set_requires_gradient_sync(True)
            else:
                self.train_state.model.set_requires_gradient_sync(False)
            metrics = self.train_state.train_step(batch)
            train_metrics.append(metrics)
            if (step + 1) % self.config.log_interval == 0:
                m = self.train_state.reduce_train_metrics(train_metrics)
                summary = {f"train/{k}": v for k, v in m.items()}
                summary['steps_per_second'] = self.config.log_interval / (
                    time.time() - train_metrics_last_t)
                self.writer.write_scalars(step + 1, summary)
                train_metrics = []
                train_metrics_last_t = time.time()

            if (step + steps_offset + 1
                ) % self.config.save_interval == 0 and steps_offset != 0 and (
                    step + 1) % self.config.accum_grad == 0:

                self.checkpoint_manager.save(self.train_state)
                dist.barrier()

            if (step + steps_offset + 1
                ) % self.config.eval_interval == 0 and steps_offset != 0 and (
                    step + 1
                ) % self.config.accum_grad == 0 and self.eval_iter is not None:
                eval_metrics_last = time.time()
                self.train_state.model.eval()
                with torch.no_grad():
                    for (i, eval_batch) in enumerate(self.eval):
                        eval_metrics = self.train_state.eval_step(batch)
                        if (i + 1) % self.config.log_interval == 0:
                            m = self.train_state.reduce_train_metrics(
                                eval_metrics)
                            summary = {f"eval/{k}": v for k, v in m.items()}
                            summary[
                                'steps_per_second'] = self.config.log_interval / (
                                    time.time() - eval_metrics_last)
                            self.writer.write_scalars(step + 1, summary)
                            eval_metrics = []
                            eval_metrics_last = time.time()
                    dist.barrier()
                self.train_state.model.train()
