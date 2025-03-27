import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    # parallel
    config.train_engine = 'torch_fsdp'  # [torch_ddp, torch_fsdp, deepspeed]
    config.dcn_data_parallelism = -1  # nnodes in torch
    config.icn_fsdp_parallelism = -1  # nproc in torch

    config.data_path = ''
    config.eval_data_path = ''
    config.batch_type = 'static'  # [static, bucket, dynamic]
    config.per_device_train_batch_size = 16
    config.per_device_eval_batch_size = 16
    config.bucket_boundaries = [500, 1000, 1500]
    config.bucket_batch_sizes = [82, 64, 32, 16]

    config.dataloader_num_workers = 20
    config.dataloader_prefetch_factor = 50
    config.max_steps = 1000000
    config.epochs = 1

    config.gradient_checkpointing = True
    config.save_interval = 100
    config.log_interval = 100

    config.optim = 'adamw'
    config.optim_conf = ml_collections.ConfigDict()
    config.optim_conf.lr = 0.0005

    config.scheduler = 'warmuplr'
    config.scheduler_conf = ml_collections.ConfigDict()
    config.scheduler_conf.warmup_steps = 25000

    config.init_infos = ml_collections.ConfigDict()

    config.accum_grad = 1
    config.grad_clip = 1

    config.checkpoint_dir = ''
    config.model_dir = ''
    config.async_checkpoint = False

    config.tensorboard_dir = ''
    config.dist_backend = 'nccl'

    # deepspeed
    config.deepspeed_config = ''
    config.deepspeed_save_states = ''

    config.precision = 'float32'  # [float32, bfloat16, float16 ...]

    config.device = 'cuda'
    config.fp16_grad_sync = False
    return config
