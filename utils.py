import os
from copy import copy
from typing import List, Union

import torch
import torch.distributed as dist
import torch.optim as optim
from absl import logging
from torch.optim.lr_scheduler import _LRScheduler


def get_nested_attribute(obj, attr_path):
    if isinstance(obj, torch.nn.parallel.DistributedDataParallel):
        obj = obj.module
    attributes = attr_path.split('.')
    for attr in attributes:
        obj = getattr(obj, attr)
    return obj


class WarmupLR(_LRScheduler):
    """The WarmupLR scheduler

    This scheduler is almost same as NoamLR Scheduler except for following
    difference:

    NoamLR:
        lr = optimizer.lr * model_size ** -0.5
             * min(step ** -0.5, step * warmup_step ** -1.5)
    WarmupLR:
        lr = optimizer.lr * warmup_step ** 0.5
             * min(step ** -0.5, step * warmup_step ** -1.5)

    Note that the maximum lr equals to optimizer.lr in this scheduler.

    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: Union[int, float, List[Union[int, float]]] = 25000,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        # __init__() must be invoked before setting field
        # because step() is also invoked in __init__()
        super().__init__(optimizer, last_epoch)

    def __repr__(self):
        return f"{self.__class__.__name__}(warmup_steps={self.warmup_steps})"

    def get_lr(self):
        step_num = self.last_epoch + 1
        warmup_steps = self.warmup_steps
        if not isinstance(warmup_steps, List):
            warmup_steps = [self.warmup_steps] * len(self.base_lrs)

        def initlr_fn(lr):
            return lr * step_num**-0.5

        def warmuplr_fn(lr, warmup_step):
            return lr * warmup_step**0.5 * min(step_num**-0.5,
                                               step_num * warmup_step**-1.5)

        return [
            initlr_fn(lr) if warmup_steps[i] == 0 else warmuplr_fn(
                lr, warmup_steps[i]) for (i, lr) in enumerate(self.base_lrs)
        ]

    def set_step(self, step: int):
        self.last_epoch = step


def is_torch_npu_available() -> bool:
    '''
        check if torch_npu is available.
        torch_npu is a npu adapter of PyTorch
    '''
    try:
        import torch_npu  # noqa
        return True
    except ImportError:
        if not torch.cuda.is_available():
            print("Module \"torch_npu\" not found. \"pip install torch_npu\" \
                if you are using Ascend NPU, otherwise, ignore it")
    return False


TORCH_NPU_AVAILABLE = is_torch_npu_available()


def init_distributed(configs):

    local_rank = os.environ.get('LOCAL_RANK', 0)
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    rank = int(os.environ.get('RANK', 0))

    logging.info('training on multiple gpus, this gpu {}'.format(local_rank) +
                 ', rank {}, world_size {}'.format(rank, world_size))
    if configs.train_engine in ["torch_fsdp"]:
        if "cuda" in configs.device:
            torch.cuda.set_device(local_rank)
        elif "npu" in configs.device and TORCH_NPU_AVAILABLE:
            torch.npu.set_device(local_rank)
        else:
            logging.error(f"not supported device: {configs.device}")
        dist.init_process_group(configs.dist_backend)
    else:
        logging.error(f"not supported engine: {configs.train_engine}")
    return world_size, local_rank, rank


def if_print_model(configs, model):
    is_distributed = dist.is_initialized()
    if (is_distributed and dist.get_rank() == 0) or not is_distributed:
        if configs.print_model:
            print(model)
            num_params = sum(p.numel() for p in model.parameters())
            print('the number of model params: {:,d}'.format(num_params))


def init_optimizer_and_scheduler(configs, model, step):
    groups = []
    lr = configs['optim_conf'].get('lr')
    if isinstance(lr, List):
        assert configs['scheduler'] == 'warmuplr'
        modules_m = configs['optim_conf']['modules']
        assert isinstance(modules_m, List)
        assert len(modules_m) + 1 == len(lr)
        special_param_ids = set()
        rest_params = []
        for (i, m_str) in enumerate(modules_m):
            sub_module = get_nested_attribute(model, m_str)
            subs_params = []
            for _, sub_params in sub_module.named_parameters():
                subs_params.append(sub_params)
                special_param_ids.add(id(sub_params))
            groups.append({'params': subs_params, 'lr': lr[i]})
        # other model's parameters
        for _, param in model.named_parameters():
            if id(param) not in special_param_ids:
                rest_params.append(param)
        groups.append({'params': rest_params, 'lr': lr[-1]})

    params = groups if len(groups) > 0 else model.parameters()
    optim_conf = copy.deepcopy(configs['optim_conf'])
    if 'modules' in optim_conf:
        del optim_conf['modules']
    if isinstance(lr, List):
        optim_conf['lr'] = lr[-1]
    if configs['optim'] == 'adam':
        optimizer = optim.Adam(params, **optim_conf)
    elif configs['optim'] == 'adamw':
        optimizer = optim.AdamW(params, **optim_conf)
    else:
        raise ValueError("unknown optimizer: " + configs['optim'])

    if configs['scheduler'] == 'warmuplr':
        scheduler = WarmupLR(optimizer, **configs['scheduler_conf'])
    else:
        raise ValueError("unknown scheduler: " + configs['scheduler'])

    scheduler.set_step(step)
    return model, optimizer, scheduler


def wrap_cuda_model(model, configs):
    # TODO: support native 4d parallel
    from torch.distributed.device_mesh import init_device_mesh
    mesh_2d = init_device_mesh(
        "cuda", (configs.dcn_data_parallelism, configs.icn_fsdp_parallelism),
        mesh_dim_names=("replicate", "shard"))
    model = torch.distributed.fsdp.fully_shard(model, mesh=mesh_2d)
    device = torch.device(configs.device)

    if configs.fp16_grad_sync:
        from torch.distributed.algorithms.ddp_comm_hooks import \
            default as comm_hooks
        model.register_comm_hook(state=None,
                                 hook=comm_hooks.fp16_compress_hook)

    return model, device
