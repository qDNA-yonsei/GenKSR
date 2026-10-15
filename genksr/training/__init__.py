"""Training package with lazy exports for dependency-light schedule tests."""

__all__ = [
    'Trainer',
    'AverageMeter',
    'warmup_sqrt_decay_lr_scheduler',
    'warm_up_cosine_lr_scheduler',
    'dir_setup',
    'Logger',
]


def __getattr__(name):
    if name == 'Trainer':
        from .trainer import Trainer
        return Trainer
    if name in {
        'AverageMeter',
        'warmup_sqrt_decay_lr_scheduler',
        'warm_up_cosine_lr_scheduler',
        'dir_setup',
        'Logger',
    }:
        from . import lr_schedulers
        return getattr(lr_schedulers, name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
