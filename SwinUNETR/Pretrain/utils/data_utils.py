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

    num_workers = 4
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
    original_transforms = False
    if original_transforms:
        #monai.transforms.croppad.dictionary CropForegroundd.__init__:allow_smaller: Current default value of argument `allow_smaller=True` has been deprecated since version 1.2. It will be changed to `allow_smaller=False` in version 1.5.
        allow_smaller=True
        train_transforms = Compose(
            [
                LoadImaged(keys=["image"]),
                # AddChanneld(keys=["image"]),
                EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
                Orientationd(keys=["image"], axcodes="RAS"),
                ScaleIntensityRanged(
                    keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
                ),
                SpatialPadd(keys="image", spatial_size=[args.roi_x, args.roi_y, args.roi_z]),
                CropForegroundd(keys=["image"], source_key="image", k_divisible=[args.roi_x, args.roi_y, args.roi_z], allow_smaller=allow_smaller),
                RandSpatialCropSamplesd(
                    keys=["image"],
                    roi_size=[args.roi_x, args.roi_y, args.roi_z],
                    num_samples=args.sw_batch_size,
                    random_center=True,
                    random_size=False,
                ),
                ToTensord(keys=["image"]),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image"]),
                # AddChanneld(keys=["image"]),
                EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
                Orientationd(keys=["image"], axcodes="RAS"),
                ScaleIntensityRanged(
                    keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
                ),
                SpatialPadd(keys="image", spatial_size=[args.roi_x, args.roi_y, args.roi_z]),
                CropForegroundd(keys=["image"], source_key="image", k_divisible=[args.roi_x, args.roi_y, args.roi_z], allow_smaller=allow_smaller),
                RandSpatialCropSamplesd(
                    keys=["image"],
                    roi_size=[args.roi_x, args.roi_y, args.roi_z],
                    num_samples=args.sw_batch_size,
                    random_center=True,
                    random_size=False,
                ),
                ToTensord(keys=["image"]),
            ]
        )
    else:
        # See /project/k10123/datasets/generate_geo_1.py
        def exclude_air(img):
            return img > 0

        #allow_smaller=True
        train_transforms = Compose(
            [
                LoadImaged(keys=["image"]),
                EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
                # NO NEED TO CROP FOREGROUND
                # CropForegroundd(select_fn=exclude_air, keys=["image"], source_key="image", k_divisible=[args.roi_x, args.roi_y, args.roi_z], allow_smaller=allow_smaller),
                # Normalize to 0,1
                ScaleIntensityRanged(
                    keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
                ),
                SpatialPadd(keys="image", spatial_size=[args.roi_x, args.roi_y, args.roi_z]),
                RandSpatialCropSamplesd(
                    keys=["image"],
                    roi_size=[args.roi_x, args.roi_y, args.roi_z],
                    num_samples=args.sw_batch_size,
                    random_center=True,
                    random_size=False,
                ),
                ToTensord(keys=["image"]),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image"]),
                EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
                # NO NEED TO CROP FOREGROUND
                # CropForegroundd(select_fn=exclude_air, keys=["image"], source_key="image", k_divisible=[args.roi_x, args.roi_y, args.roi_z], allow_smaller=allow_smaller),
                ScaleIntensityRanged(
                    keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
                ),
                SpatialPadd(keys="image", spatial_size=[args.roi_x, args.roi_y, args.roi_z]),
                RandSpatialCropSamplesd(
                    keys=["image"],
                    roi_size=[args.roi_x, args.roi_y, args.roi_z],
                    num_samples=args.sw_batch_size,
                    random_center=True,
                    random_size=False,
                ),
                ToTensord(keys=["image"]),
            ]
        )

    if args.cache_dataset:
        cache_rate = 1.0
        if args.rank == 0: print("Using MONAI Cache Dataset")
        train_ds = CacheDataset(data=datalist, transform=train_transforms, cache_rate=cache_rate, num_workers=num_workers)
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
        train_ds, batch_size=args.batch_size, num_workers=num_workers, sampler=train_sampler, drop_last=True
    )

    val_ds = Dataset(data=vallist, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=num_workers, shuffle=False, drop_last=True)
    return train_loader, val_loader
