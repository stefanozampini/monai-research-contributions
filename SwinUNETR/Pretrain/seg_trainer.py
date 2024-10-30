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
from utils.utils import AverageMeter, distributed_all_gather
from utils.ckp import save_ckp

from monai.data import decollate_batch

def dump_seg(args, img_list, basename, distributed=True):
    for step, batch in enumerate(img_list):
        if distributed and args.world_size > 1:
           fname = f'{basename}_{step}_{args.rank}.npz'
        else:
           fname = f'{basename}_{step}.npz'
        fname = os.path.join(args.logdir,fname)
        target, segmented = batch
        np.savez(fname, target=target, segmented=segmented)


def train_epoch(model, loader, optimizer, scaler, epoch, loss_func, args):
    model.train()
    #run_loss = AverageMeter()
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
# XXX???
#        if args.distributed:
#            loss_list = distributed_all_gather([loss], out_numpy=True, is_valid=idx < loader.sampler.valid_length)
#            run_loss.update(
#                np.mean(np.mean(np.stack(loss_list, axis=0), axis=0), axis=0), n=args.batch_size * args.world_size
#            )
#        else:
#            run_loss.update(loss.item(), n=args.batch_size)
#        if args.rank == 0:
#            print(
#                "  step {}/{}".format(idx, len(loader)),
#                "  loss: {:.4f}".format(run_loss.avg),
#                "  time {:.2f}s".format(time.time() - start_time),
#                flush=True,
#            )
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
    #return run_loss.avg

def val_epoch(model, loader, epoch, acc_func, args, model_inferer=None, post_sigmoid=None, post_pred=None):
    model.eval()
    run_acc = AverageMeter()

    # print(f"[{args.rank}] NUM VALID {len(loader)}",flush=True)
    seg_list = []
    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            start_time = time.time()
            data, target = batch_data["image"], batch_data["label"]
            data, target = data.to(args.device), target.to(args.device)
            # print(f"[{args.rank}] VALID shape {idx} {data.shape}",flush=True)
            with autocast(args.autocast_device_type, enabled=args.amp):
                logits = model_inferer(data)
            val_labels_list = decollate_batch(target)
            val_outputs_list = decollate_batch(logits)
            val_output_convert = [post_pred(post_sigmoid(val_pred_tensor)) for val_pred_tensor in val_outputs_list]

            in_s = np.stack([v.detach().cpu().numpy().astype(np.uint8).squeeze() for v in val_labels_list])
            out_s = np.stack([np.argmax(v.detach().cpu().numpy(),axis=0).astype(np.uint8) for v in val_output_convert])
            seg_list.append((in_s, out_s))

            acc_func.reset()
            acc_func(y_pred=val_output_convert, y=val_labels_list)
            acc, not_nans = acc_func.aggregate()
            acc = acc.to(args.device)
            if args.distributed:
                acc_list, not_nans_list = distributed_all_gather(
                    [acc, not_nans], out_numpy=True, # is_valid=idx < loader.sampler.valid_length
                )
                for al, nl in zip(acc_list, not_nans_list):
                    run_acc.update(al, n=nl)
            else:
                run_acc.update(acc.cpu().numpy(), n=not_nans.cpu().numpy())

            if args.rank == 0:
                print("  Val {}/{}".format(idx, len(loader)),flush=True)
                print("    run_avg {}".format(run_acc.avg),flush=True)
                print("    time {:.2f}s".format(time.time() - start_time),flush=True)
            del data, target, val_labels_list, val_outputs_list, val_output_convert, in_s, out_s

    return run_acc.avg, seg_list


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
    post_sigmoid=None,
    post_pred=None,
    semantic_classes=None,
):
    val_acc_max = 0.0
    for epoch in range(start_epoch, end_epoch):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
            torch.distributed.barrier()
        if args.rank == 0:
            print("Epoch:", epoch, flush=True)
        train_epoch(
            model, train_loader, optimizer, scaler=scaler, epoch=epoch, loss_func=loss_func, args=args
        )
        if args.rank == 0:
            model_pth = os.path.join(args.logdir,"model_last_epoch.pt")
            save_ckp(args.task, model, optimizer, scheduler, epoch, model_pth)
        b_new_best = False
        if (epoch + 1) % args.eval_num == 0:
            if args.distributed:
                torch.distributed.barrier()
            epoch_time = time.time()
            val_acc, img_list = val_epoch(
                model,
                val_loader,
                epoch=epoch,
                acc_func=acc_func,
                model_inferer=model_inferer,
                args=args,
                post_sigmoid=post_sigmoid,
                post_pred=post_pred,
            )

            val_avg_acc = np.mean(val_acc)
            save = False
            if val_avg_acc > val_acc_max:
                save = True
                val_acc_max = val_avg_acc
            if args.rank == 0:
                print(f"Mean_Val_Dice {val_avg_acc}, best {val_acc_max}", flush=True)

                if save:
                    print("  saving new best", flush=True)
                    model_pth = os.path.join(args.logdir,"model_best_acc.pt")
                    save_ckp(args.task, model, optimizer, scheduler, epoch, model_pth)
            if save:
               dump_seg(args, img_list, "test_best_seg", distributed=True)
            del img_list

        if scheduler is not None:
            scheduler.step()
