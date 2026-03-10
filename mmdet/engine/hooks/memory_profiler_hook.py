# Copyright (c) OpenMMLab. All rights reserved.
import cProfile
from pathlib import Path
from typing import Optional, Sequence

from mmengine.hooks import Hook, ProfilerHook
from mmengine.hooks.hook import DATA_BATCH
from mmengine.runner import Runner
import torch
from torch.profiler import profile

from mmdet.registry import HOOKS
from mmdet.structures import DetDataSample


@HOOKS.register_module()
class MemoryProfilerHook(Hook):
    """Memory profiler hook recording memory information including virtual
    memory, swap memory, and the memory of the current process.

    Args:
        interval (int): Checking interval (every k iterations).
            Default: 50.
    """

    def __init__(self, interval: int = 50) -> None:
        try:
            from psutil import swap_memory, virtual_memory
            self._swap_memory = swap_memory
            self._virtual_memory = virtual_memory
        except ImportError:
            raise ImportError('psutil is not installed, please install it by: '
                              'pip install psutil')

        try:
            from memory_profiler import memory_usage
            self._memory_usage = memory_usage
        except ImportError:
            raise ImportError(
                'memory_profiler is not installed, please install it by: '
                'pip install memory_profiler')

        self.interval = interval

    def _record_memory_information(self, runner: Runner) -> None:
        """Regularly record memory information.

        Args:
            runner (:obj:`Runner`): The runner of the training or evaluation
                process.
        """
        # in Byte
        virtual_memory = self._virtual_memory()
        swap_memory = self._swap_memory()
        # in MB
        process_memory = self._memory_usage()[0]
        factor = 1024 * 1024
        runner.logger.info(
            'Memory information '
            'available_memory: '
            f'{round(virtual_memory.available / factor)} MB, '
            'used_memory: '
            f'{round(virtual_memory.used / factor)} MB, '
            f'memory_utilization: {virtual_memory.percent} %, '
            'available_swap_memory: '
            f'{round((swap_memory.total - swap_memory.used) / factor)}'
            ' MB, '
            f'used_swap_memory: {round(swap_memory.used / factor)} MB, '
            f'swap_memory_utilization: {swap_memory.percent} %, '
            'current_process_memory: '
            f'{round(process_memory)} MB')

    def after_train_iter(self,
                         runner: Runner,
                         batch_idx: int,
                         data_batch: Optional[dict] = None,
                         outputs: Optional[dict] = None) -> None:
        """Regularly record memory information.

        Args:
            runner (:obj:`Runner`): The runner of the training process.
            batch_idx (int): The index of the current batch in the train loop.
            data_batch (dict, optional): Data from dataloader.
                Defaults to None.
            outputs (dict, optional): Outputs from model. Defaults to None.
        """
        if self.every_n_inner_iters(batch_idx, self.interval):
            self._record_memory_information(runner)

    def after_val_iter(
            self,
            runner: Runner,
            batch_idx: int,
            data_batch: Optional[dict] = None,
            outputs: Optional[Sequence[DetDataSample]] = None) -> None:
        """Regularly record memory information.

        Args:
            runner (:obj:`Runner`): The runner of the validation process.
            batch_idx (int): The index of the current batch in the val loop.
            data_batch (dict, optional): Data from dataloader.
                Defaults to None.
            outputs (Sequence[:obj:`DetDataSample`], optional):
                Outputs from model. Defaults to None.
        """
        if self.every_n_inner_iters(batch_idx, self.interval):
            self._record_memory_information(runner)

    def after_test_iter(
            self,
            runner: Runner,
            batch_idx: int,
            data_batch: Optional[dict] = None,
            outputs: Optional[Sequence[DetDataSample]] = None) -> None:
        """Regularly record memory information.

        Args:
            runner (:obj:`Runner`): The runner of the testing process.
            batch_idx (int): The index of the current batch in the test loop.
            data_batch (dict, optional): Data from dataloader.
                Defaults to None.
            outputs (Sequence[:obj:`DetDataSample`], optional):
                Outputs from model. Defaults to None.
        """
        if self.every_n_inner_iters(batch_idx, self.interval):
            self._record_memory_information(runner)


def trace_handler(dir_name: str, exp_name: str | None = None):
    """
    Outputs tracing files to directory of ``dir_name``, then that directory can
    be directly delivered to tensorboard as logdir.
    ``worker_name`` should be unique for each worker in distributed scenario,
    it will be set to '[hostname]_[pid]' by default.
    """
    from pathlib import Path
    import os
    import socket

    span_counter = 0

    def handler_fn(prof: profile) -> None:
        nonlocal span_counter
        nonlocal exp_name

        profile_dir = Path(dir_name)
        if not profile_dir.is_dir():
            try:
                Path.mkdir(dir_name, exist_ok=True)
            except Exception as e:
                raise RuntimeError("Can't create directory: " +
                                   dir_name) from e

        if not exp_name:
            exp_name = f"{socket.gethostname()}_{os.getpid()}"

        base_filename = f"{exp_name}.{span_counter}"

        trace_filename = f"{base_filename}.trace.json"
        trace_file = profile_dir / trace_filename
        prof.export_chrome_trace(str(trace_file))

        # option available in torch 2.X.X only.
        if torch.__version__[0] == '2' and prof.profile_memory:
            memory_filename = f"{base_filename}.mem.raw.json.gz"
            memory_file = profile_dir / memory_filename
            prof.export_memory_timeline(str(memory_file))

        span_counter += 1

    return handler_fn


@HOOKS.register_module()
class ProfilerTestHook(ProfilerHook):

    after_train_epoch = None
    after_train_iter = None

    def before_run(self, runner: Runner):
        """Initialize the profiler.

        Through the runner parameter, the validity of the parameter is further
        determined.
        """
        on_trace_ready = trace_handler(**self.on_trace_ready)
        self.profiler = profile(  # noqa
            activities=self.activities,
            schedule=self.schedule,
            on_trace_ready=on_trace_ready,
            record_shapes=self.record_shapes,
            profile_memory=self.profile_memory,
            with_stack=self.with_stack,
            experimental_config=torch._C._profiler._ExperimentalConfig(
                verbose=True),
            with_modules=True,
            with_flops=self.with_flops)

        self.profiler.start()
        runner.logger.info('profiler is profiling...')

    def before_test_iter(self,
                         runner: Runner,
                         batch_idx: int,
                         data_batch: DATA_BATCH = None) -> None:
        self.profiler.step()

    def after_run(self, runner: Runner) -> Runner:
        self.profiler.stop()


@HOOKS.register_module()
class cProfileHook(Hook):

    def __init__(self,
                 n_active: int,
                 base_filename: str,
                 profile_dir: str,
                 n_skip: int = 1,
                 n_cycle: int = 1):
        # assert (device := get_device()) == 'cpu', (  # breakline fmt
        #     f"run with `export CUDA_VISIBLE_DEVICES=-1` to profile with "
        #     f"cPython, current device is {device}")

        self.base_filename = base_filename
        self.profile_dir = Path(profile_dir)

        if self.profile_dir.exists():
            assert self.profile_dir.is_dir, f"{profile_dir} should be a folder"
        else:
            self.profile_dir.mkdir(parents=True)

        self.n_skip = n_skip
        self.n_active = n_active

        self.n_cycle = n_cycle
        #: Keep track of the cycle progess
        self.i_cycle = 1

        #: Profiling is ON
        self.enable = True

        # Init in _init_profiler at specific time.
        self.profiler: cProfile.Profile
        self.step: int

    def _init_profiler(self) -> None:
        self.profiler = cProfile.Profile()
        self.step = 0

    def before_run(self, runner: Runner) -> None:
        self._init_profiler()

    def _before_iter(self,
                     runner: Runner,
                     batch_idx: int,
                     data_batch: DATA_BATCH = None) -> None:
        # Enable profiler once n_skip steps has been done
        if self.enable and self.step == self.n_skip:
            self.profiler.enable()

    before_train_iter = _before_iter
    before_val_iter = _before_iter
    before_test_iter = _before_iter

    def _after_iter(self,
                    runner: Runner,
                    batch_idx: int,
                    data_batch: DATA_BATCH = None,
                    outputs: Optional[dict] = None) -> None:
        # Early stop if the cycles are exhausted is done
        if not self.enable:
            return

        # First n_skip steps are ignored
        if self.step < self.n_skip:
            self.step += 1
            return

        # positive: n_skip step are passed
        # more than n_active: cycle is done
        if (self.step - self.n_skip) < self.n_active:
            self.step += 1
            return

        self.profiler.disable()
        filename = f"{self.base_filename}_{self.i_cycle}.pstats"
        self.profiler.dump_stats(self.profile_dir / filename)

        # Manage cycle, if it is the last one then disable everything,
        # else reset it
        self.n_cycle -= 1
        if self.n_cycle == 0:
            self.enable = False
            self.profiler = None
        else:
            self._init_profiler()
            self.i_cycle += 1

    after_train_iter = _after_iter
    after_val_iter = _after_iter
    after_test_iter = _after_iter
