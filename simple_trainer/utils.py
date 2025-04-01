import os

import torch
import torch.distributed as dist
from absl import logging


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


def wrap_cuda_model(model, configs):
    local_world_size = int(os.environ.get('LOCAL_WORLD_SIZE', 1))
    world_size = int(os.environ.get('WORLD_SIZE', 1))

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
