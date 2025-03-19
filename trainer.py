import abc
import time
from contextlib import nullcontext

import torch

from writer import create_default_writer


class TrainState(abc.ABC):

    def __init__(self, model):
        super().__init__()
        self._model = model
        # TODO: resume from ckpt
        self._step = 0

    @classmethod
    def train_step(self, batch: dict):
        pass

    @property
    def model(self):
        return self._model

    @property
    def step(self):
        return self._step

    @classmethod
    def resume(self, path: str):
        pass

    @classmethod
    def reduce_train_metrics(self, metrics: dict):
        pass


class Trainer:

    def __init__(self,
                 config,
                 state: TrainState,
                 dataloader_or_dataset,
                 eval_datalader_or_dataset=None):
        self.config = config
        # TODO: distributed
        rank = 0
        self.writer = create_default_writer(config.tensorboard_dir,
                                            just_logging=rank != 0,
                                            asynchronous=True)
        self.train_iter = iter(dataloader_or_dataset)

        self.train_state = state

    def train(self, resume: str):
        # TODO: restore from TrainState
        steps_offset = self.train_state.step
        if steps_offset != 0:
            # TODO: skip first n in train iter
            return
        train_metrics_last_t = time.time()

        if isinstance(self.train_state._model,
                      torch.nn.parallel.DistributedDataParallel):
            model_context = self.train_state.model.join
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
                if self.config.train_engine in [
                        "torch_ddp", "torch_fsdp"
                ] and (step + 1) % self.config.accum_grad != 0:
                    context = self.train_state.model.no_sync
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
                    self.train_state.save()
                    dist.barrier()
