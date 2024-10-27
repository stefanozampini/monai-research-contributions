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
from utils.cpu_binding import cores_per_process
import os
from time import time

import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
from losses.loss import Loss
from models.ssl_head import SSLHead
from optimizers.lr_scheduler import WarmupCosineSchedule
from torch import autocast
try:
    from torch import GradScaler
except ImportError:
    GradScaler = None
    pass
from torch.nn.parallel import DistributedDataParallel
#from torch.utils.tensorboard import SummaryWriter
from utils.data_utils import get_loader
from utils.ops import aug_rand, rot_rand

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

def filter_load(model_pth):
        model_dict = torch.load(model_pth)
        state_dict = model_dict["state_dict"]
        # fix potential differences in state dict keys from pre-training to
        # fine-tuning
        if "module." in list(state_dict.keys())[0]:
            print("Tag 'module.' found in state dict - fixing!")
            for key in list(state_dict.keys()):
                state_dict[key.replace("module.", "")] = state_dict.pop(key)
        if "swin_vit" in list(state_dict.keys())[0]:
            print("Tag 'swin_vit' found in state dict - fixing!")
            for key in list(state_dict.keys()):
                state_dict[key.replace("swin_vit", "swinViT")] = state_dict.pop(key)
        return model_dict

def save_ckp(model, optimizer, scheduler, global_step, model_pth):
    checkpoint = {
        "global_step": global_step,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }
    torch.save(checkpoint, model_pth)

def load_ckp(model, optimizer, scheduler, model_pth, model_only=False):
    ckp_dict = filter_load(model_pth)
    global_step = ckp_dict["global_step"]
    model_dict = ckp_dict["state_dict"]
    model.load_state_dict(model_dict)
    if model_only:
      return global_step
    if "optimizer" in ckp_dict:
      optimizer_dict = ckp_dict["optimizer"]
      optimizer.load_state_dict(optimizer_dict)
    if "scheduler" in ckp_dict:
      scheduler_dict = ckp_dict["scheduler"]
      scheduler.load_state_dict(scheduler_dict)
    return global_step

def main():

    def train(args, global_step, train_loader, val_best, scaler):
        model.train()
        loss_train = []
        loss_train_recon = []

        for step, batch in enumerate(train_loader):
            t1 = time()
            x = batch["image"].to(args.device)
            x1, rot1 = rot_rand(args, x)
            x2, rot2 = rot_rand(args, x)
            x1_augment = aug_rand(args, x1)
            x2_augment = aug_rand(args, x2)
            x1_augment = x1_augment
            x2_augment = x2_augment
            t2 = time()
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

            if args.lrdecay:
                scheduler.step()
            optimizer.zero_grad()
            if args.rank == 0:
                print("Step:{}/{}, Loss:{:.4f} ({:.4f}, {:.4f}, {:.4f}), Time:{:.4f} ({:.4f})".format(global_step, args.num_steps, loss, losses_tasks[0].item(), losses_tasks[1].item(), losses_tasks[2].item(), time() - t1, t2 - t1),flush=True)

            global_step += 1
            if args.distributed:
                val_cond = (dist.get_rank() == 0) and (global_step % args.eval_num == 0)
            else:
                val_cond = global_step % args.eval_num == 0

            if val_cond:
                val_loss, val_loss_recon, img_list = validation(args, test_loader)
                print("Validation/loss_recon", val_loss_recon, global_step,flush=True)
                print("train/loss_total", np.mean(loss_train), global_step,flush=True)
                print("train/loss_recon", np.mean(loss_train_recon), global_step,flush=True)
                #writer.add_scalar("Validation/loss_recon", scalar_value=val_loss_recon, global_step=global_step)
                #writer.add_scalar("train/loss_total", scalar_value=np.mean(loss_train), global_step=global_step)
                #writer.add_scalar("train/loss_recon", scalar_value=np.mean(loss_train_recon), global_step=global_step)

                if val_loss_recon < val_best:
                    model_pth = os.path.join(logdir,"model_bestValRMSE.pt")
                    save_ckp(model, optimizer, scheduler, global_step, model_pth)
                    dump_images(args, img_list, "test_best", distributed=False)
                    print(
                        "Model was saved ! Best Recon. Val Loss: {:.4f}, Recon. Val Loss: {:.4f}".format(
                            val_best, val_loss_recon
                        ), flush=True
                    )
                    val_best = val_loss_recon
                else:
                    print(
                        "Model was not saved ! Best Recon. Val Loss: {:.4f} Recon. Val Loss: {:.4f}".format(
                            val_best, val_loss_recon
                        ), flush=True
                    )
                del img_list
            del x, x1, rot1, x2, rot2, x1_augment, x2_augment
            if global_step > args.num_steps:
               break
        model_pth = os.path.join(logdir,"model_last_epoch.pt")
        save_ckp(model, optimizer, scheduler, global_step, model_pth)
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
                print("    Validation step:{}, Losses:{:.4f}, {:.4f}, {:.4f}".format(step, losses_tasks[0], losses_tasks[1], losses_tasks[2]),flush=True)
                del val_inputs, x1, rot1, x2, rot2, x1_augment, x2_augment

        return np.mean(loss_val), np.mean(loss_val_recon), img_list

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
    parser.add_argument("--batch_size", default=2, type=int, help="number of batch size")
    parser.add_argument("--sw_batch_size", default=2, type=int, help="number of sliding window batch size")
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
    #parser.add_argument('--no-zendnn', action='store_true', default=False, help='disables ZenDNN')
    args = parser.parse_args()

    if not 'jsonlist' in args:
      raise RuntimeError('Must supply --jsonlist argument')
    if not 'datasetlist' in args:
      raise RuntimeError('Must supply --datasetlist argument')
    args.jsonlist = args.jsonlist.split(',')
    args.datasetlist = args.datasetlist.split(',')

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
    if cores_per_process > 0:
       torch.set_num_threads(cores_per_process)
       torch.set_num_interop_threads(cores_per_process)
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
        print(f"Training with a single process on 1 {device_type}.")
    print(f"Number of threads used: {torch.get_num_threads()}.")

    if args.rank == 0:
        os.makedirs(logdir, exist_ok=True)

    if args.rank == 0:
       print("============ARGUMENTS==============")
       print('\n'.join(f'{k}={v}' for k, v in vars(args).items()))
       print("===================================")

    model = SSLHead(args)
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

    if args.opt == "adam":
        optimizer = optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.decay)

    elif args.opt == "adamw":
        optimizer = optim.AdamW(params=model.parameters(), lr=args.lr, weight_decay=args.decay)

    elif args.opt == "sgd":
        optimizer = optim.SGD(params=model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.decay)

    if args.lrdecay:
        if args.lr_schedule == "warmup_cosine":
            scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=args.num_steps)

        elif args.lr_schedule == "poly":

            def lambdas(epoch):
                return (1 - float(epoch) / float(args.epochs)) ** 0.9

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambdas)

    global_step = 0
    if args.resume:
        global_step = load_ckp(model, optimizer, scheduler, args.resume, args.resume_model_only)

    loss_function = Loss(args)
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if args.cuda:
           model = DistributedDataParallel(model, device_ids=[args.local_rank])
        else:
           model = DistributedDataParallel(model)
    train_loader, test_loader = get_loader(args)

    if args.check_images:
      dump_images(args, train_loader, "train")
      dump_images(args, test_loader, "test")
      return

    best_val = 1e8
    if args.amp and GradScaler is not None:
        scaler = GradScaler()
    else:
        scaler = None
    while global_step < args.num_steps:
        global_step, loss, best_val = train(args, global_step, train_loader, best_val, scaler)

    if args.distributed:
        if dist.get_rank() == 0:
            torch.save(model.state_dict(), os.path.join(logdir,"final_model.pth"))
        dist.destroy_process_group()
    else:
        torch.save(model.state_dict(), os.path.join(logdir,"final_model.pth"))


if __name__ == "__main__":
    main()
