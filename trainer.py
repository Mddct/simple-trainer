import abc
import time
from contextlib import nullcontext

import torch

from utils import init_distributed, if_print_model, wrap_cuda_model
from writer import create_default_writer

import os

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
import torch.multiprocessing as mp
import torch.nn as nn

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


class TrainState(Stateful):
    """This is a useful wrapper for checkpointing the Application State. Since this object is compliant
    with the Stateful protocol, DCP will automatically call state_dict/load_stat_dict as needed in the
    dcp.save/load APIs.

    Note: We take advantage of this wrapper to hande calling distributed state dict methods on the model
    and optimizer.
    """

    def __init__(self, model, optimizer=None, tag='F5'):
        self.model = model
        self.optimizer = optimizer
        self.tag
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

    @property
    def step(self):
        return self._step

    @classmethod
    def reduce_train_metrics(self, metrics: dict):
        pass

    @property
    def model(self):
        return self._model

    def save(self, checkpoint_dir):
        state_dict = {self.tag: self}
        dcp.save(state_dict, checkpoint_id=checkpoint_dir)
        
    def restore(self, checkpoint_dir)
        state_dict = {"F5": self}
        dcp.load(
            state_dict=state_dict,
            checkpoint_id=checkpoint_dir,
        )
        self._step = self.optimizer.s

class Trainer:

    def __init__(self,
                 config,
                 state: TrainState,
                 dataloader_or_dataset,
                 eval_datalader_or_dataset=None):
        self.config = config

        self.world_size, self.rank, self.local_rank = init_distributed(config)
        if_print_model(
            config,
            state.model,
        )
        self.model  = wrap_cuda_model(self.train_state.model, config)

        self.writer = create_default_writer(config.tensorboard_dir,
                                            just_logging=self.rank != 0,
                                            asynchronous=True)
        self.train_iter = iter(dataloader_or_dataset)

        self.train_state = state

        # TODO: pytorch native dcp
        self.checkpoint_manager = None

    def train(self):
        if self.config.checkpoint_dir != '':
            self.train_state.restore(self.config.checkpoint_dir)
        steps_offset = self.train_state.step
        if steps_offset != 0:
            # TODO: skip first n in train iter
            return
        train_metrics_last_t = time.time()

        if isinstance(self.train_state._model,
                      torch.nn.parallel.DistributedDataParallel):
            model_context = self.model.join
        else:
            model_context = nullcontext

        train_metrics_last_t = time.time()
        train_metrics = []
        with model_context():
            for step, batch in zip(range(steps_offset, self.config.max_steps),
                                   self.train_iter):

                # Disable gradient synchronizations across DDP processes.
                # Within this context, gradients will be accumulated on module
                # variables, which will later be synchronized.
                if (step + 1) % self.config.accum_grad != 0:
                    context = self.model.no_sync
                # Used for single gpu training and DDP gradient synchronization
                # processes.
                else:
                    context = nullcontext

                with context():
                    metrics = self.train_state.train_step(batch)
                train_metrics.append(metrics)
                if (step + 1) % self.config.log_interval == 0:
                    m = self.train_state.reduce_train_metrics(train_metrics)
                    summary = {k: v for k, v in m.items()}
                    summary['steps_per_second'] = self.config.log_interval / (
                        time.time() - train_metrics_last_t)
                    self.writer.write_scalars(step + 1, summary)
                    train_metrics = []
                    train_metrics_last_t = time.time()

                if (
                        step + steps_offset + 1
                ) % self.config.save_interval == 0 and steps_offset != 0 and (
                        step + 1) % self.config.accum_grad == 0:
                    import torch.distributed as dist
                    self.train_state.save(self.config.model_dir)
                    dist.barrier()
                    
                if (
                        step + steps_offset + 1
                ) % self.config.eval_interval == 0 and steps_offset != 0 and (
                        step + 1) % self.config.accum_grad == 0:
                    # TODO: run evaluation
                    # import torch.distributed as dist
                    # self.train_state.save()
                    # dist.barrier()
