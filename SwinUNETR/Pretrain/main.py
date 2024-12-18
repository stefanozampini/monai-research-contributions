# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
from utils.cpu_binding import num_threads, affinity
import os
from time import time
from functools import partial

import numpy as np
import torch
if affinity: # https://github.com/pytorch/pytorch/issues/99625
    os.sched_setaffinity(os.getpid(), affinity)
if num_threads > 0:
    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(num_threads)
import torch.distributed as dist
import torch.optim as optim
from losses.loss import Loss
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks.nets import SwinUNETR
from monai.transforms import Activations, AsDiscrete, Compose
from monai.inferers import sliding_window_inference
from models.ssl_head import SSLHead
from monai.utils.enums import MetricReduction
from seg_trainer import run_training, eval_segmentation
from optimizers.lr_scheduler import WarmupCosineSchedule, LinearWarmupCosineAnnealingLR
from torch import autocast
try:
    from torch import GradScaler
except ImportError:
    GradScaler = None
    pass
from torch.nn.parallel import DistributedDataParallel
from utils.data_utils import get_loader
from utils.ops import aug_rand, rot_rand
from utils.ckp import load_ckp, save_ckp

def model_view(model):
    cnt = 0
    out = []
    out.append(f'{model.__class__}')
    out.append('Parameters')
    for name, p in model.named_parameters():
        out.append(f'  {name}: {p.requires_grad} {p.shape} {p.dtype} {p.device}')
        cnt += np.prod(p.shape)
    out.append(f'  Total: {int(cnt)}')
    print('\n'.join(out), flush=True)

def main():

    def train(args, global_step, train_loader, val_best, scaler, epoch):
        model.train()
        loss_train = []
        loss_train_recon = []
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
            torch.distributed.barrier()

        for step, batch in enumerate(train_loader):
            t1 = time()
            x = batch["image"].to(args.device)
            x1, rot1 = rot_rand(args, x)
            x2, rot2 = rot_rand(args, x)
            x1_augment = aug_rand(args, x1)
            x2_augment = aug_rand(args, x2)
            t2 = time()
            optimizer.zero_grad()
            with autocast(args.autocast_device_type, enabled=args.amp):
                rot1_p, contrastive1_p, rec_x1 = model(x1_augment)
                rot2_p, contrastive2_p, rec_x2 = model(x2_augment)
                rot_p = torch.cat([rot1_p, rot2_p], dim=0)
                rots = torch.cat([rot1, rot2], dim=0)
                imgs_recon = torch.cat([rec_x1, rec_x2], dim=0)
                imgs = torch.cat([x1, x2], dim=0)
                loss, losses_tasks = loss_function(rot_p, rots, contrastive1_p, contrastive2_p, imgs_recon, imgs)
                del imgs, imgs_recon, rots, rot1_p, rot2_p, rot_p, contrastive1_p, contrastive2_p, rec_x1, rec_x2

            loss_train.append(loss.item())
            loss_train_recon.append(losses_tasks[2].item())
            if args.amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if args.grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()

            lrused = None
            if args.lrdecay:
                lrused = scheduler.get_lr()
                scheduler.step()
            t3 = time()

            global_step += 1
            val_cond = global_step % args.eval_num == 0

            if val_cond:
                if args.distributed:
                    torch.distributed.barrier()
                val_loss, val_loss_recon, img_list = validation(args, test_loader)
                if args.rank == 0:
                    print("Validation/loss_recon", val_loss_recon, global_step,flush=True)
                    print("train/loss_total", np.mean(loss_train), global_step,flush=True)
                    print("train/loss_recon", np.mean(loss_train_recon), global_step,flush=True)
                #writer.add_scalar("Validation/loss_recon", scalar_value=val_loss_recon, global_step=global_step)
                #writer.add_scalar("train/loss_total", scalar_value=np.mean(loss_train), global_step=global_step)
                #writer.add_scalar("train/loss_recon", scalar_value=np.mean(loss_train_recon), global_step=global_step)

                    if val_loss_recon < val_best:
                        model_pth = os.path.join(logdir,"model_bestValRMSE.pt")
                        save_ckp(args.task, model, optimizer, scheduler, global_step, model_pth)
                        print(
                            "Model was saved ! Best Recon. Val Loss: {:.4f}, Recon. Val Loss: {:.4f}".format(
                                val_best, val_loss_recon
                            ), flush=True
                        )
                    else:
                        print(
                            "Model was not saved ! Best Recon. Val Loss: {:.4f} Recon. Val Loss: {:.4f}".format(
                                val_best, val_loss_recon
                            ), flush=True
                        )

                if val_loss_recon < val_best:
                    val_best = val_loss_recon
                    dump_images(args, img_list, "test_best")
                del img_list
            del x, x1, rot1, x2, rot2, x1_augment, x2_augment
            if args.rank == 0:
                print("Step:{}/{}, Loss:{:.4f} ({:.4f}, {:.4f}, {:.4f}) LR {}, Time:{:.4f} ({:.4f}, {:.4f})".format(global_step - 1, args.num_steps, loss, losses_tasks[0].item(), losses_tasks[1].item(), losses_tasks[2].item(), lrused, time() - t1, t3 - t1, t2 - t1),flush=True)
            if global_step > args.num_steps:
                break
        if args.rank == 0:
            t1 = time()
            model_pth = os.path.join(logdir,"model_last_epoch.pt")
            save_ckp(args.task, model, optimizer, scheduler, global_step, model_pth)
        return global_step, loss, val_best

    def model_to_img(inputs, args, rec = None):
        if rec is None:
           rec = model(inputs)[2]
        return inputs.detach().cpu().numpy(), rec.detach().cpu().numpy()

    def validation(args, test_loader):
        model.eval()
        loss_val = []
        loss_val_recon = []
        img_list = []
        with torch.no_grad():
            for step, batch in enumerate(test_loader):
                val_inputs = batch["image"].to(args.device)
                x1, rot1 = rot_rand(args, val_inputs)
                x2, rot2 = rot_rand(args, val_inputs)
                x1_augment = aug_rand(args, x1)
                x2_augment = aug_rand(args, x2)
                with autocast(args.autocast_device_type, enabled=args.amp):
                    rot1_p, contrastive1_p, rec_x1 = model(x1_augment)
                    rot2_p, contrastive2_p, rec_x2 = model(x2_augment)
                    rot_p = torch.cat([rot1_p, rot2_p], dim=0)
                    rots = torch.cat([rot1, rot2], dim=0)
                    imgs_recon = torch.cat([rec_x1, rec_x2], dim=0)
                    imgs = torch.cat([x1, x2], dim=0)
                    loss, losses_tasks = loss_function(rot_p, rots, contrastive1_p, contrastive2_p, imgs_recon, imgs)

                    # images
                    or_in, rec_in = model_to_img(val_inputs, args)
                    or_x1, rec_x1 = model_to_img(x1_augment, args, rec=rec_x1)
                    or_x2, rec_x2 = model_to_img(x2_augment, args, rec=rec_x2)

                    del imgs, imgs_recon, rots, rot1_p, rot2_p, rot_p, contrastive1_p, contrastive2_p

                loss_recon = losses_tasks[2]
                loss_val.append(loss.item())
                loss_val_recon.append(loss_recon.item())
                img_list.append((or_in, rec_in, or_x1, rec_x1, or_x2, rec_x2))
                print("    [{}] Validation step:{}, Losses:{:.4f}, {:.4f}, {:.4f} bs: {}".format(args.rank, step, losses_tasks[0], losses_tasks[1], losses_tasks[2], val_inputs.shape[0]),flush=True)
                del val_inputs, x1, rot1, x2, rot2, x1_augment, x2_augment

        if args.cuda:
            torch.cuda.synchronize(args.device)
        dummy = torch.tensor([sum(loss_val), sum(loss_val_recon), len(loss_val)], device=args.device)
        dist.all_reduce(dummy, op=dist.ReduceOp.SUM)
        loss_val_mean = dummy[0]/dummy[2]
        loss_recon_mean = dummy[1]/dummy[2]
        return loss_val_mean.item(), loss_recon_mean.item(), img_list

    def dump_images(args, loader_or_img_list, basename, distributed=True):
        model.eval()
        islist = isinstance(loader_or_img_list, list)
        with torch.no_grad():
            for step, batch in enumerate(loader_or_img_list):
                if distributed and args.world_size > 1:
                   fname = f'{basename}_{step}_{args.rank}.npz'
                else:
                   fname = f'{basename}_{step}.npz'
                fname = os.path.join(logdir,fname)
                if islist:
                    or_in, rec_in, or_x1, rec_x1, or_x2, rec_x2 = batch
                else:
                    val_inputs = batch["image"].to(args.device)
                    x1, rot1 = rot_rand(args, val_inputs)
                    x2, rot2 = rot_rand(args, val_inputs)
                    x1_augment = aug_rand(args, x1)
                    x2_augment = aug_rand(args, x2)
                    with autocast(args.autocast_device_type, enabled=args.amp):
                        or_in, rec_in = model_to_img(val_inputs, args)
                        or_x1, rec_x1 = model_to_img(x1_augment, args)
                        or_x2, rec_x2 = model_to_img(x2_augment, args)
                np.savez(fname, or_in=or_in, or_x1=or_x1, or_x2=or_x2, rec_in=rec_in, rec_x1=rec_x1, rec_x2=rec_x2)


    parser = argparse.ArgumentParser(description="PyTorch Training")
    parser.add_argument("--logdir", default="test", type=str, help="directory to save")
    parser.add_argument("--datadir", default="dataset", type=str, help="directory where input data resides")
    parser.add_argument("--jsonlist", default=argparse.SUPPRESS, type=str, help="comma separated list of json files")
    parser.add_argument("--datasetlist",default=argparse.SUPPRESS,type=str, help="comma separated list of dataset names")
    parser.add_argument("--epochs", default=100, type=int, help="number of training epochs")
    parser.add_argument("--num_steps", default=100000, type=int, help="number of training iterations")
    parser.add_argument("--eval_num", default=500, type=int, help="evaluation frequency")
    parser.add_argument("--warmup_steps", default=500, type=int, help="warmup steps")
    parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
    parser.add_argument("--out_channels", default=1, type=int, help="number of output channels")
    parser.add_argument("--feature_size", default=48, type=int, help="embedding size")
    parser.add_argument("--dropout_path_rate", default=0.0, type=float, help="drop path rate")
    parser.add_argument("--use_checkpoint", action="store_true", help="use gradient checkpointing to save memory")
    parser.add_argument("--spatial_dims", default=3, type=int, help="spatial dimension of input data")
    parser.add_argument("--a_min", default=-1000, type=float, help="a_min in ScaleIntensityRanged")
    parser.add_argument("--a_max", default=1000, type=float, help="a_max in ScaleIntensityRanged")
    parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
    parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")
    parser.add_argument("--roi_x", default=96, type=int, help="roi size in x direction")
    parser.add_argument("--roi_y", default=96, type=int, help="roi size in y direction")
    parser.add_argument("--roi_z", default=96, type=int, help="roi size in z direction")
    parser.add_argument("--batch_size", default=2, type=int, help="train batch size")
    parser.add_argument("--test_batch_size", default=0, type=int, help="test batch size")
    parser.add_argument("--sw_batch_size", default=2, type=int, help="number of sliding window batch size")
    parser.add_argument("--infer_sw_batch_size", default=0, type=int, help="number of sliding window batch size for inference")
    parser.add_argument("--lr", default=4e-4, type=float, help="learning rate")
    parser.add_argument("--decay", default=0.1, type=float, help="decay rate")
    parser.add_argument("--momentum", default=0.9, type=float, help="momentum")
    parser.add_argument("--lrdecay", action="store_true", help="enable learning rate decay")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="maximum gradient norm")
    parser.add_argument("--loss_type", default="SSL", type=str)
    parser.add_argument("--opt", default="adamw", type=str, help="optimization algorithm")
    parser.add_argument("--lr_schedule", default="warmup_cosine", type=str)
    parser.add_argument("--resume", default=None, type=str, help="resume training")
    parser.add_argument("--resume_model_only", default=0, type=int, help="resume training only model")
    parser.add_argument("--grad_clip", action="store_true", help="gradient clip")
    parser.add_argument("--noamp", action="store_true", help="do NOT use amp for training")
    parser.add_argument("--smartcache_dataset", action="store_true", help="use monai smartcache Dataset")
    parser.add_argument("--cache_dataset", action="store_true", help="use monai cache Dataset")
    parser.add_argument("--check_images", action="store_true", help="dump images")
    parser.add_argument("--view_model", action="store_true", help="view model")
    parser.add_argument("--num_workers", default=4, type=int, help="number of workers")
    parser.add_argument("--task", default='pretrain', type=str, help="training task")
    parser.add_argument("--infer_overlap", default=0.5, type=float, help="sliding window inference overlap")
    #parser.add_argument('--no-zendnn', action='store_true', default=False, help='disables ZenDNN')
    args = parser.parse_args()

    if not 'jsonlist' in args:
      raise RuntimeError('Must supply --jsonlist argument')
    if not 'datasetlist' in args:
      raise RuntimeError('Must supply --datasetlist argument')
    args.jsonlist = args.jsonlist.split(',')
    args.datasetlist = args.datasetlist.split(',')

    if args.test_batch_size <= 0:
        args.test_batch_size = args.batch_size
    if args.infer_sw_batch_size <= 0:
        args.infer_sw_batch_size = args.sw_batch_size

    #use_zendnn = not args.no_zendnn
    #if use_zendnn:
    #    import zentorch
    args.cuda = torch.cuda.is_available()
    logdir = args.logdir
    args.amp = not args.noamp
    if args.cuda:
        torch.backends.cudnn.benchmark = True
    else:
        args.amp = False # XXX
    torch.autograd.set_detect_anomaly(True)
    args.distributed = False
    if "WORLD_SIZE" in os.environ:
        args.distributed = True #int(os.environ["WORLD_SIZE"]) > 1
    args.world_size = 1
    args.rank = 0
    device_type = 'GPU' if args.cuda else 'CPU'
    args.autocast_device_type = 'cuda' if args.cuda else 'cpu'
    args.local_rank = int(os.environ.get('LOCAL_RANK',0))
    if args.distributed:
        backend = "nccl" if args.cuda else "gloo"
        if args.cuda:
           os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
           args.device = torch.device('cuda', args.local_rank)
           torch.cuda.set_device(args.device)
        else:
           args.device = torch.device('cpu')
        torch.distributed.init_process_group(backend=backend)
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
        if args.rank == 0:
          print(
              "Training in distributed mode with multiple processes, 1 {0} per process. Total processes {1}."
              .format(device_type, args.world_size)
          )
    else:
        args.device = torch.device("cuda" if args.cuda else "cpu")
        if args.rank == 0:
          print(f"Training with a single process on 1 {device_type}.")
    if args.rank == 0:
        print(f"Number of threads used: {torch.get_num_threads()}.")

    if args.rank == 0:
        os.makedirs(logdir, exist_ok=True)

    if args.rank == 0:
       print("============ARGUMENTS==============")
       print('\n'.join(f'{k}={v}' for k, v in vars(args).items()))
       print("===================================")

    if args.task == 'pretrain':
        model = SSLHead(args)
    else:
        model = SwinUNETR(
            (args.roi_x, args.roi_y, args.roi_z),
            args.in_channels,
            args.out_channels,
            # img_size=(args.roi_x, args.roi_y, args.roi_z), # deprecated
            # in_channels=args.in_channels,
            # out_channels=args.out_channels,
            feature_size=args.feature_size,
            use_checkpoint=args.use_checkpoint,
        )

    if args.cuda:
        model.to(args.local_rank)
    #else:
    #    print(f'Compiling model ZEN ? {use_zendnn}',flush=True)
    #    # Too slow when tracing, segfaults!
    #    #if use_zendnn:
    #    #    model = torch.compile(model, backend='zentorch')
    #    #else:
    #    #    model = torch.compile(model, mode='reduce-overhead')
    # params: 19097191
    if args.view_model:
       model_view(model)

    optimizer = None
    scheduler = None
    if args.opt == "adam":
        optimizer = optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.decay)
    elif args.opt == "adamw":
        optimizer = optim.AdamW(params=model.parameters(), lr=args.lr, weight_decay=args.decay)
    else:
        optimizer = optim.SGD(params=model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.decay)

    if args.lrdecay:
        if args.task == "pretrain":
            if args.lr_schedule == "warmup_cosine":
                scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=args.num_steps)
            elif args.lr_schedule == "cosine_anneal":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_steps)
        else:
            if args.lr_schedule == "warmup_cosine":
                scheduler = LinearWarmupCosineAnnealingLR(
                    optimizer, warmup_epochs=args.warmup_steps, max_epochs=args.num_steps
                )
            elif args.lr_schedule == "cosine_anneal":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_steps)

    global_step = 0
    if args.resume:
        global_step = load_ckp(args.task, model, optimizer, scheduler, args.resume, args.resume_model_only)

    if args.task == 'pretrain':
        loss_function = Loss(args)
    else:
        loss_function = DiceLoss(to_onehot_y=True, softmax=True)
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if args.cuda:
           model = DistributedDataParallel(model, device_ids=[args.local_rank])
        else:
           model = DistributedDataParallel(model)
    train_loader, test_loader = get_loader(args)

    if args.check_images and args.task == "pretrain":
      dump_images(args, train_loader, "train")
      dump_images(args, test_loader, "test")
      if args.distributed:
          dist.destroy_process_group()
      return

    best_val = 1e8
    if args.amp and GradScaler is not None:
        scaler = GradScaler()
    else:
        scaler = None
    if args.task == 'pretrain':
        epoch = 0
        while global_step < args.num_steps:
            global_step, loss, best_val = train(args, global_step, train_loader, best_val, scaler, epoch)
            epoch += 1
    else:
        inf_size = [args.roi_x, args.roi_y, args.roi_z]
        post_label = AsDiscrete(to_onehot=args.out_channels)
        post_pred = AsDiscrete(argmax=True, to_onehot=args.out_channels) # dim=0 is ok since output is decollated
        dice_acc = DiceMetric(include_background=True, reduction=MetricReduction.MEAN_BATCH, get_not_nans=True)
        model_inferer = partial(
            sliding_window_inference,
            roi_size=inf_size,
            sw_batch_size=args.infer_sw_batch_size,
            predictor=model,
            overlap=args.infer_overlap,
            mode='gaussian',
        )
        if args.check_images:
            eval_segmentation(model, test_loader, args, model_inferer, post_pred, post_label)
            # eval_segmentation(model, train_loader, args, model_inferer, post_pred, post_label)
            if args.distributed:
                dist.destroy_process_group()
            return
        else:
            run_training(
                model=model,
                train_loader=train_loader,
                val_loader=test_loader,
                optimizer=optimizer,
                loss_func=loss_function,
                acc_func=dice_acc,
                args=args,
                model_inferer=model_inferer,
                scheduler=scheduler,
                scaler=scaler, # SZ
                start_epoch=global_step,
                end_epoch=args.num_steps,
                post_label=post_label,
                post_pred=post_pred,
                #semantic_classes=semantic_classes,
            )


    if args.rank == 0:
        torch.save(model.state_dict(), os.path.join(logdir,"final_model.pth"))
    if args.distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
