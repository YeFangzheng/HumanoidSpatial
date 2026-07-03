# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.hooks import Hook
from mmdet3d.registry import HOOKS
from . import is_parallel

__all__ = ['SequentialControlHook']


@HOOKS.register_module()
class SequentialControlHook(Hook):
    """ """

    def __init__(self, temporal_start_epoch=1, temporal_start_iter=-1):
        super().__init__()
        self.temporal_start_epoch = temporal_start_epoch
        self.temporal_start_iter = temporal_start_iter

    def set_temporal_flag(self, runner, flag):
        if is_parallel(runner.model):
            runner.model.module.with_prev = flag
        else:
            runner.model.with_prev = flag

    def before_train(self, runner):
        self.set_temporal_flag(runner, False)

    def before_train_epoch(self, runner):
        if runner.epoch > self.temporal_start_epoch and self.temporal_start_iter < 0:
            self.set_temporal_flag(runner, True)

    def before_test(self, runner):
        self.set_temporal_flag(runner, True)
    
