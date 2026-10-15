import os
import shutil
import sys
import uuid

import torch

from genksr.training.schedules import warmup_cosine_factor


class AverageMeter:
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def average(self):
        return self.avg

    def value(self):
        return self.val


def warmup_sqrt_decay_lr_scheduler(optimizer, model_size, factor, warmup):
    def lr_decay_func(step):
        if step == 0:
            step = 1
        return factor * (model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda step: lr_decay_func(step))


def warm_up_cosine_lr_scheduler(optimizer, epochs, warm_up_epochs, eta_min):
    if epochs <= 0:
        raise ValueError(f'epochs must be positive, got {epochs}')
    if warm_up_epochs < 0 or warm_up_epochs >= epochs:
        raise ValueError(
            f'warm_up_epochs must satisfy 0 <= warm_up_epochs < epochs, got '
            f'{warm_up_epochs} for epochs={epochs}')
    if eta_min < 0:
        raise ValueError(f'eta_min must be non-negative, got {eta_min}')

    if warm_up_epochs <= 0:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=eta_min)
    else:
        lr_lambdas = []
        for param_group in optimizer.param_groups:
            base_lr = param_group['lr']
            if eta_min > base_lr:
                raise ValueError(
                    f'eta_min={eta_min} cannot exceed base learning rate {base_lr}')
            min_factor = eta_min / base_lr

            def lr_decay_func(epoch, min_factor=min_factor):
                return warmup_cosine_factor(
                    step=epoch,
                    total_steps=epochs,
                    warmup_steps=warm_up_epochs,
                    min_factor=min_factor,
                )

            lr_lambdas.append(lr_decay_func)

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer=optimizer, lr_lambda=lr_lambdas)

    return scheduler


def dir_setup(results_root, dataset_id, model_id, num_train, train_id, verbose=True):
    """Setup directory for training results and model checkpoints."""
    if train_id is None:
        train_id = 'id-' + str(uuid.uuid4())
        if verbose:
            print(f'==> generated random uuid {train_id}')

    model_dir = os.path.join(results_root, dataset_id, model_id)
    results_dir = os.path.join(model_dir, num_train, train_id)

    if train_id == 'debug' and os.path.exists(results_dir):
        shutil.rmtree(results_dir)
        if verbose:
            print(f'removed dir {results_dir}')

    os.makedirs(results_dir)
    if verbose:
        print(f'created dir {results_dir}')

    checkpoints_dir = os.path.join(results_dir, "checkpoints/")
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)

    return results_dir, model_dir, train_id


class Logger:
    def __init__(self, print_fp=None):
        self.terminal = sys.stdout
        self.log_file = "out.txt" if print_fp is None else print_fp
        self.encoding = sys.stdout.encoding

        if os.path.isfile(self.log_file):
            os.remove(self.log_file)
            print(f'removed {self.log_file}')

    def write(self, message):
        self.terminal.write(message)
        with open(self.log_file, "a") as log:
            log.write(message)

    def flush(self):
        pass
