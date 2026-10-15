import copy
import os
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from typing import Union, List

from genksr.constants import (
    COSINE_ANNEALING_WARM_RESTARTS_SCHEDULER,
    COSINE_ANNEALING_SCHEDULER,
    WARMUP_SQRT_SCHEDULER,
    WARMUP_COSINE_SCHEDULER,
)
from genksr.data.dataloading import (
    batch_generator_epoch_mixed,
    batch_generator_iterations_mixed,
    batch_generator_epoch,
    batch_generator_iterations,
)
from genksr.nn.common import KLDivLoss
from genksr.nn.gc_model import GCModel
from genksr.training.lr_schedulers import (
    AverageMeter,
    warmup_sqrt_decay_lr_scheduler,
    warm_up_cosine_lr_scheduler,
)

use_cuda = torch.cuda.is_available()


class Trainer:
    """Unified trainer for GCModel (Transformer or Mamba backend)."""

    def __init__(
            self,
            model: GCModel,
            train_dataset,
            val_dataset,
            test_dataset,
            save_dir: str,
            iterations: int,
            lr: float,
            final_lr: float,
            lr_scheduler: str,
            warmup_frac: float,
            weight_decay: float,
            batch_size: int,
            rng: np.random.Generator,
            device,
            tensorboard_dir: str = None,
            eval_every: int = 1,
            checkpoint_every: int = 1,
            epoch_mode=True,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.save_dir = save_dir
        self.iterations = iterations
        self.lr = lr
        self.final_lr = final_lr
        self.lr_scheduler_name = lr_scheduler
        self.warmup_frac = warmup_frac
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.rng = rng
        self.device = device
        self.eval_every = eval_every
        if checkpoint_every <= 0:
            raise ValueError("checkpoint_every must be positive")
        self.checkpoint_every = checkpoint_every
        self.epoch_mode = epoch_mode

        self.checkpoint_dir = os.path.join(save_dir, 'checkpoints/')

        # tensorboard
        tb_dir = tensorboard_dir if tensorboard_dir is not None else 'tensorboard'
        self.summary_writer = SummaryWriter(os.path.join(save_dir, tb_dir))

        # select batch generator based on dataset type
        if isinstance(self.train_dataset, List):
            self._batch_generator_epoch = batch_generator_epoch_mixed
            self._batch_generator_iters = batch_generator_iterations_mixed
            num_train_samples = sum([len(ds) for ds in train_dataset])
        else:
            self._batch_generator_epoch = batch_generator_epoch
            self._batch_generator_iters = batch_generator_iterations
            num_train_samples = len(train_dataset)

        self.total_steps = int(
            num_train_samples // batch_size + int(num_train_samples % batch_size > 0)
        ) * iterations

        # loss function
        self.criterion = KLDivLoss(padding_idx=model.pad_token)

        # init optimizers and meters
        self._initialize_optimizers()
        self._initialize_meters()

    def train(self):
        cudnn.benchmark = True if use_cuda else cudnn.benchmark

        if use_cuda:
            self.model = self.model.cuda()

        if self.epoch_mode:
            return self._train_loop_epochs(1, self.iterations + 1)

        raise NotImplementedError

    def save_model(self, epoch, is_best=False):
        ckpt_fn = 'model_best.pth.tar' if is_best else f'checkpoint_{epoch}.pth.tar'
        save_as = os.path.join(self.checkpoint_dir, ckpt_fn)
        torch.save({
            'epoch': epoch,
            'model_state_dict': copy.deepcopy(self.model.state_dict()),
            'optimizer': copy.deepcopy(self.optimizer.state_dict()),
        }, save_as)
        return save_as

    def _train_loop_epochs(self, start_epoch, end_epoch):
        best_val_loss = np.inf

        pbar = tqdm(range(start_epoch, end_epoch), desc='Epoch')

        for epoch in pbar:
            self.model.train()
            self._reset_meters()

            t0 = time.time()
            for tgt_batch, graphs_batch, _, noise_factors_batch in self._batch_generator_epoch(
                    self.train_dataset, batch_size=self.batch_size, pad_token=self.model.pad_token, rng=self.rng,
                    shuffle=True, device=self.device
            ):
                noise_factors_batch = noise_factors_batch.to(self.device)

                self._forward_step(
                    tgt=tgt_batch.tgt,
                    tgt_y=tgt_batch.tgt_y,
                    tgt_mask=tgt_batch.tgt_mask,
                    graphs_batch=graphs_batch,
                    noise_factor=noise_factors_batch,
                    norm=tgt_batch.ntokens.item(),
                    is_train=True,
                    update_meter=True
                )

                self.scheduler.step()

            train_total_loss = self.total_loss_meter.average()

            if epoch % self.eval_every == 0:
                val_total_loss = self._eval_epoch(self.val_dataset)
                test_total_loss = self._eval_epoch(self.test_dataset)

                if not np.isnan(val_total_loss) and val_total_loss < best_val_loss:
                    best_val_loss = val_total_loss
                    self.save_model(epoch, is_best=True)
            else:
                test_total_loss, val_total_loss = np.nan, np.nan

            epoch_time = time.time() - t0

            pstr = self._get_progress_str(train_total_loss, test_total_loss, val_total_loss, step=epoch)
            pbar.set_postfix_str(pstr)

            self._collect_summaries(train_total_loss, test_total_loss, val_total_loss, step=epoch, step_time=epoch_time)

            # Per-epoch wall-clock log (-> train_out.txt) and rolling checkpoint, so a run can be
            # stopped at any epoch and still leave a usable checkpoint_final.pth.tar.
            print(f'[epoch {epoch}/{end_epoch - 1}] {epoch_time:.1f}s | train_loss={train_total_loss:.4f}', flush=True)
            if epoch % self.checkpoint_every == 0 or epoch == end_epoch - 1:
                self.save_model('final', is_best=False)

        train_loss = self._eval_epoch(self.train_dataset)
        val_loss = self._eval_epoch(self.val_dataset)
        test_loss = self._eval_epoch(self.test_dataset)

        return train_loss, val_loss, test_loss

    def _eval_epoch(self, dataset):
        if dataset is None:
            return np.nan

        self.model.eval()
        self._reset_meters()

        for tgt_batch, graphs_batch, _, noise_factors_batch in self._batch_generator_epoch(
                dataset, batch_size=self.batch_size, pad_token=self.model.pad_token, shuffle=False, rng=self.rng,
                device=self.device
        ):
            noise_factors_batch = noise_factors_batch.to(self.device)

            self._forward_step(
                tgt=tgt_batch.tgt,
                tgt_y=tgt_batch.tgt_y,
                tgt_mask=tgt_batch.tgt_mask,
                graphs_batch=graphs_batch,
                noise_factor=noise_factors_batch,
                norm=tgt_batch.ntokens.item(),
                is_train=False,
                update_meter=True
            )

        return self.total_loss_meter.average()

    def _forward_step(self, tgt, tgt_y, tgt_mask, norm, graphs_batch, noise_factor, is_train, update_meter=True):
        graph_embed, out = self.model.forward(tgt, tgt_mask, coupling_graph=graphs_batch, noise_factor=noise_factor)

        log_probs = self.model.generator(out)

        if is_train:
            total_loss = self.criterion(
                log_probs.contiguous().view(-1, log_probs.size(-1)), tgt_y.contiguous().view(-1)
            ) / norm

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
        else:
            with torch.no_grad():
                total_loss = self.criterion(
                    log_probs.contiguous().view(-1, log_probs.size(-1)),
                    tgt_y.contiguous().view(-1)
                ) / norm

        total_loss = total_loss.item()

        if update_meter:
            self.total_loss_meter.update(val=total_loss, n=tgt.shape[0])

        return total_loss

    def _get_progress_str(self, train_total_loss, test_total_loss, val_total_loss, step):
        pstr = f'step {step} | lr: {self.scheduler.get_last_lr()[0]:.8f}, train total-loss: {train_total_loss:.4f}'

        if not np.isnan(test_total_loss):
            pstr = pstr + f', test total-loss: {test_total_loss:.4f}'
        if not np.isnan(val_total_loss):
            pstr = pstr + f', val total-loss: {val_total_loss:.4f}'

        return pstr

    def _collect_summaries(self, train_total_loss, test_total_loss, val_total_loss, step, step_time):
        self.summary_writer.add_scalars('total_loss/',
                                        {'train': train_total_loss, 'test': test_total_loss, 'val': val_total_loss},
                                        global_step=step)
        self.summary_writer.add_scalar('step_time/', step_time, global_step=step)
        self.summary_writer.add_scalar('learning_rate/', self.scheduler.get_last_lr()[0], global_step=step)
        self.summary_writer.flush()

    def _initialize_optimizers(self):
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=self.weight_decay)

        if self.lr_scheduler_name == COSINE_ANNEALING_WARM_RESTARTS_SCHEDULER:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, T_0=self.iterations // 5, T_mult=1
            )
        elif self.lr_scheduler_name == COSINE_ANNEALING_SCHEDULER:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.total_steps, eta_min=self.final_lr
            )
        elif self.lr_scheduler_name == WARMUP_SQRT_SCHEDULER:
            warmup_steps = int(self.warmup_frac * self.total_steps)
            self.scheduler = warmup_sqrt_decay_lr_scheduler(
                optimizer=self.optimizer, model_size=self.model.d_model, factor=1, warmup=warmup_steps
            )
        elif self.lr_scheduler_name == WARMUP_COSINE_SCHEDULER:
            warmup_steps = int(self.warmup_frac * self.total_steps)
            self.scheduler = warm_up_cosine_lr_scheduler(
                optimizer=self.optimizer, epochs=self.total_steps, warm_up_epochs=warmup_steps, eta_min=self.final_lr
            )
        else:
            raise ValueError(f'unknown lr_scheduler {self.lr_scheduler_name}')

    def _initialize_meters(self):
        self.total_loss_meter = AverageMeter()

    def _reset_meters(self):
        self.total_loss_meter.reset()
