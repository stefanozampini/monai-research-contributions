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
from monai.data import CacheDataset, DataLoader, Dataset, DistributedSampler, SmartCacheDataset, load_decathlon_datalist
from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandCropByPosNegLabeld,
    RandSpatialCropSamplesd,
    RandFlipd,
    ScaleIntensityRanged,
    Spacingd,
    SpatialPadd,
    ToTensord,
)

def get_loader(args):
    data_dir = args.datadir
    jsonfiles = args.jsonlist
    datasets_dir = args.datasetlist
    list_dir = os.path.join(data_dir,"jsons")
    jsonlist = [os.path.join(list_dir,json_f) for json_f in jsonfiles]
    datadirlist = [os.path.join(data_dir,d) for d in datasets_dir]

    datalist = [load_decathlon_datalist(j, False, "training", base_dir=d) for j,d in zip(jsonlist, datadirlist)]
    vallist = [load_decathlon_datalist(j, False, "validation", base_dir=d) for j,d in zip(jsonlist, datadirlist)]
    if args.rank == 0:
      for i,(d,v,j) in enumerate(zip(datalist, vallist, jsonlist)):
         _, n = os.path.split(j)
         print(f"Dataset {i} {n}: number of training data: {len(d)}")
         print(f"Dataset {i} {n}: number of validation data: {len(v)}")
    datalist = sum(datalist, [])
    vallist = sum(vallist, [])
    if args.rank == 0:
       print("Dataset all training: number of data: {}".format(len(datalist)))
       print("Dataset all validation: number of data: {}".format(len(vallist)))
    if args.task == 'pretrain':
        trans = []
        keys = ["image"]
        trans.append(LoadImaged(keys=keys))
        if args.in_channels == 1:
           trans.append(EnsureChannelFirstd(keys=keys, channel_dim="no_channel"))
        # NO NEED TO CROP FOREGROUND
        #trans.append(CropForegroundd(select_fn=exclude_air, keys=["image"], source_key="image", k_divisible=[args.roi_x, args.roi_y, args.roi_z], allow_smaller=allow_smaller))
        # data already prepared
        # trans.append(ScaleIntensityRanged(keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True))
        # trans.append(SpatialPadd(keys="image", spatial_size=[args.roi_x, args.roi_y, args.roi_z]))
        trans.append(RandSpatialCropSamplesd(keys=keys,
                                             roi_size=[args.roi_x, args.roi_y, args.roi_z],
                                             num_samples=args.sw_batch_size,
                                             random_center=True,
                                             random_size=False))
        trans.append(ToTensord(keys=keys))


        train_transforms = Compose(trans)
        val_transforms = Compose(trans)
    else:
        train_trans = []
        val_trans = []
        keys = ["image", "label"]
        train_trans.append(LoadImaged(keys=keys))
        train_trans.append(EnsureChannelFirstd(keys="label", channel_dim="no_channel"))
        train_trans.append(RandSpatialCropSamplesd(keys=keys,
                                             roi_size=[args.roi_x, args.roi_y, args.roi_z],
                                             num_samples=args.sw_batch_size, # XXX BRATS has num_samples=1 XXX
                                             random_center=True,
                                             random_size=False))
        train_trans.append(RandFlipd(keys=keys, prob=0.5, spatial_axis=0))
        train_trans.append(RandFlipd(keys=keys, prob=0.5, spatial_axis=1))
        train_trans.append(RandFlipd(keys=keys, prob=0.5, spatial_axis=2))
        # img <- img - mean(img) / std(img)
        #transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        # img <- img * (1 + factor)
        #transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
        # img <- img + offset
        #transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
        train_trans.append(ToTensord(keys=keys))
        train_transforms = Compose(train_trans)

        val_trans.append(LoadImaged(keys=keys))
        val_trans.append(EnsureChannelFirstd(keys="label", channel_dim="no_channel"))
        #transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        val_trans.append(ToTensord(keys=keys))
        val_transforms = Compose(val_trans)
        # train_transforms = Compose(val_trans)

    kwargs = {}
    if args.num_workers > 0:
       kwargs['num_workers'] = args.num_workers
    if args.cache_dataset:
        cache_rate = 1.0
        if args.rank == 0: print("Using MONAI Cache Dataset")
        train_ds = CacheDataset(data=datalist, transform=train_transforms, cache_rate=cache_rate, **kwargs)
    elif args.smartcache_dataset:
        if args.rank == 0: print("Using MONAI SmartCache Dataset")
        train_ds = SmartCacheDataset(
            data=datalist,
            transform=train_transforms,
            replace_rate=1.0,
            cache_num=2 * args.batch_size * args.sw_batch_size,
        )
    else:
        if args.rank == 0: print("Using generic dataset")
        train_ds = Dataset(data=datalist, transform=train_transforms)

    if args.distributed:
        train_sampler = DistributedSampler(dataset=train_ds, even_divisible=True, shuffle=True)
    else:
        train_sampler = None
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=train_sampler, drop_last=True, **kwargs,
    )

    val_ds = Dataset(data=vallist, transform=val_transforms)

    #val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=True, **kwargs)
    if args.distributed:
        #val_sampler = DistributedSampler(dataset=val_ds, even_divisible=False, shuffle=False, drop_last=False) # drop_last goes to torch distributed sampler
        val_sampler = DistributedSampler(dataset=val_ds, even_divisible=True, shuffle=False, drop_last=False) # drop_last goes to torch distributed sampler
    else:
        val_sampler = None
    val_loader = DataLoader(
        val_ds, batch_size=args.test_batch_size, sampler=val_sampler, drop_last=False, **kwargs,
    )
    return train_loader, val_loader
