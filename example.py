import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Optional

import torch
from absl import app, flags
from ml_collections import config_flags

from writer import create_default_writer

FLAGS = flags.FLAGS

_CONFIG = config_flags.DEFINE_config_file('config')
from trainer import Trainer, TrainState


class DummyModel(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, batch):
        return torch.tensor([1, 2, 3])


class DummyTrainState(TrainState):

    def __init__(self, model):
        super().__init__(model)

    def train_step(self, batch: dict):
        return {'loss': self.model(torch.tensor([1, 2, 3]))}

    def resume(self, path: str):
        pass

    @classmethod
    def reduce_train_metrics(self, metrics: dict):
        return {"mean": torch.tensor([1.2])}


def main(_):
    config = FLAGS.config
    print(config)
    model = DummyModel()
    state = DummyTrainState(model)
    dataloader = iter(range(1, 1000))
    trainer = Trainer(config, state, dataloader)
    trainer.train(resume=config.checkpoint)


if __name__ == '__main__':
    app.run(main)
