default_hooks = dict(logger=dict(type='LoggerHook', interval=4))

n_active_step = 1
n_skip_first = 2
n_warmup_step = 1
n_repeat = 50
hook_profiler = dict(
    type="ProfilerTestHook",
    by_epoch=False,
    schedule=dict(
        skip_first=n_skip_first,
        wait=n_warmup_step,
        warmup=n_warmup_step,
        active=n_active_step,
        repeat=n_repeat),
    activity_with_cpu=True,
    activity_with_cuda=True,
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
    on_trace_ready=dict(
        dir_name='/home/reusm/code/mmdet/work_dirs/pstr_r50/profile',
        exp_name='batch_1'))
custom_hooks = [hook_profiler]
test_dataloader = dict(
    # num_batch_per_epoch=n_skip_first + (n_repeat + 1) *
    # (n_active_step + n_warmup_step),
    num_workers=2,
    pin_memory=True)

# n_active = 1
# n_skip = 2
# n_cycle = 30
# cprofile_hook = dict(
#     type="cProfileHook",@
#     n_active=n_active,
#     n_skip=n_skip,
#     n_cycle=n_cycle,
#     base_filename="cprofile_cpu_b2",
#     profile_dir="/home/reusm/code/mmdet/work_dirs/pstr_r50/profile")

# test_dataloader = dict(
#     num_batch_per_epoch=(n_cycle) * (n_skip + n_active) + 1,
#     num_workers=0,
#     pin_memory=False,
# )
# custom_hooks = [cprofile_hook]

custom_hooks = []

