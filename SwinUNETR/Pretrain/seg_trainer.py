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

import os
import time

import numpy as np
import torch
import torch.nn.parallel
import torch.distributed as dist
from torch import autocast
from utils.ckp import save_ckp
from utils.utils import AverageMeter, distributed_all_gather

from monai.data import decollate_batch

def dump_seg_list(args, img_list, basename, distributed=True):
    for step, batch in enumerate(img_list):
        if distributed and args.world_size > 1:
           fname = f'{basename}_{step}_{args.rank}.npz'
        else:
           fname = f'{basename}_{step}.npz'
        fname = os.path.join(args.logdir,fname)
        data, target, segmented = batch
        np.savez(fname, data=data, target=target, segmented=segmented)


def train_epoch(model, loader, optimizer, scaler, epoch, loss_func, args):
    model.train()
    # print(f"[{args.rank}] NUM TRAIN {len(loader)}",flush=True)
    for idx, batch_data in enumerate(loader):
        optimizer.zero_grad()
        start_time = time.time()
        data, target = batch_data["image"], batch_data["label"]
        data, target = data.to(args.device), target.to(args.device)
        # print(f"[{args.rank}] TRAIN shape {idx} {data.shape}",flush=True)
        with autocast(args.autocast_device_type, enabled=args.amp):
            loss = loss_func(model(data), target)
        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        if args.distributed:
            dummy = torch.tensor([loss.item()], device=args.device)
            dist.all_reduce(dummy, op=dist.ReduceOp.SUM)
            full_loss = dummy.item() / args.world_size
        else:
            full_loss = loss.item()
        if args.rank == 0:
            print(
                "  step {}/{}".format(idx, len(loader)),
                "  loss: {:.4f}".format(full_loss),
                "  time {:.2f}s".format(time.time() - start_time),
                flush=True,
            )
        del data, target

def val_epoch(model, loader, epoch, args, acc_func=None, model_inferer=None, post_label=None, post_pred=None):
    model.eval()

    # print(f"[{args.rank}] NUM VALID {len(loader)}",flush=True)
    seg_list = []
    run_acc = AverageMeter()
    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            start_time = time.time()
            data, target = batch_data["image"], batch_data["label"]
            data, target = data.to(args.device), target.to(args.device)
            with autocast(args.autocast_device_type, enabled=args.amp):
                logits = model_inferer(data)
            val_labels_list = decollate_batch(target)
            val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
            val_outputs_list = decollate_batch(logits)
            val_outputs_convert = [post_pred(val_pred_tensor).bool() for val_pred_tensor in val_outputs_list]

            acc_func.reset()
            acc_func(y_pred=val_outputs_convert, y=val_labels_convert)
            acc, not_nans = acc_func.aggregate() # multi-process reduction inside
            acc.to(args.device)
            run_acc.update(acc.cpu().numpy(), n=not_nans.cpu().numpy())

            in_s = np.stack([v.detach().cpu().numpy().astype(int).squeeze() for v in val_labels_list])
            out_s = np.stack([np.argmax(v.detach().cpu().numpy(), axis=0).astype(int) for v in val_outputs_convert])
            seg_list.append((data.detach().cpu().numpy(), in_s, out_s))

            if args.rank == 0:
                avg_acc = np.mean(run_acc.avg)
                print("  Val {}/{}".format(idx, len(loader)),flush=True)
                print("    mean acc {}".format(avg_acc),flush=True)
                print("    acc {}".format(run_acc.avg),flush=True)
                print("    time {:.2f}s".format(time.time() - start_time),flush=True)

            del data, target, val_labels_list, val_labels_convert, val_outputs_list, val_outputs_convert, in_s, out_s

    return np.mean(run_acc.avg), seg_list


def run_training(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_func,
    acc_func,
    args,
    model_inferer=None,
    scheduler=None,
    scaler=None,
    start_epoch=0,
    end_epoch=0,
    post_label=None,
    post_pred=None,
    semantic_classes=None,
):
    val_acc_max = 0.0
    for epoch in range(start_epoch, end_epoch):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
            torch.distributed.barrier()

        lr = None
        if scheduler is not None:
            lr = scheduler.get_lr()
        if args.rank == 0:
            print(f"Epoch: {epoch} LR {lr}", flush=True)
        train_epoch(
            model, train_loader, optimizer, scaler=scaler, epoch=epoch, loss_func=loss_func, args=args
        )
        if args.rank == 0:
            model_pth = os.path.join(args.logdir,"model_last_epoch.pt")
            save_ckp(args.task, model, optimizer, scheduler, epoch, model_pth)
        if (epoch + 1) % args.eval_num == 0:
            if args.distributed:
                torch.distributed.barrier()
            epoch_time = time.time()
            save = False
            val_avg_acc, img_list = val_epoch(
                model,
                val_loader,
                epoch,
                args,
                acc_func=acc_func,
                model_inferer=model_inferer,
                post_label=post_label,
                post_pred=post_pred,
            )
            if val_avg_acc > val_acc_max:
                save = True
            if args.rank == 0:
                print(f"Mean_Val_Dice {val_avg_acc}, best {val_acc_max}", flush=True)
                if save:
                    print("  saving new best", flush=True)
                    model_pth = os.path.join(args.logdir,"model_best_acc.pt")
                    save_ckp(args.task, model, optimizer, scheduler, epoch + 1, model_pth)
            if save:
                val_acc_max = val_avg_acc
                dump_seg_list(args, img_list, "test_best_seg", distributed=True)
            del img_list

        if scheduler is not None:
            scheduler.step()

def eval_segmentation(model, loader, args, model_inferer, post_pred, post_label):
    model.eval()

    seg_list = []
    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            print(f"[{args.rank}] eval {idx}/{len(loader)}", flush=True)
            data, target = batch_data["image"], batch_data["label"]
            data = data.to(args.device)
            # logits = model_inferer(data)
            logits = model(data)

            val_labels_list = decollate_batch(target)
            val_outputs_list = decollate_batch(logits)
            # val_outputs_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]

            in_s = np.stack([v.detach().cpu().numpy().astype(int).squeeze() for v in val_labels_list])
            out_s = np.stack([np.argmax(v.detach().cpu().numpy(), axis=0).astype(int) for v in val_outputs_list])
            seg_list.append((data.detach().cpu().numpy(), in_s, out_s))

            del data, target, val_labels_list, val_outputs_list, in_s, out_s

    dump_seg_list(args, seg_list, "eval_segmentation", distributed=True)
    del seg_list
