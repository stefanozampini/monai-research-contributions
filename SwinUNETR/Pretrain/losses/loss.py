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

import torch
from torch.nn import functional as F
from monai.losses import ContrastiveLoss, DiceCELoss, FocalLoss

class Loss(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.rot_loss = torch.nn.CrossEntropyLoss()
        if args.out_channels > 1:
            # self.recon_loss = DiceCELoss(to_onehot_y=True, softmax=True)
            self.recon_loss = FocalLoss(to_onehot_y=True, use_softmax=True)
        else:
            self.recon_loss = torch.nn.L1Loss()
        self.contrast_loss = ContrastiveLoss()
        if args.cuda:
          self.rot_loss.cuda()
          self.recon_loss.cuda()
          self.contrast_loss.cuda()
        self.alpha1 = 1.0
        self.alpha2 = 1.0
        self.alpha3 = 1.0

    def __call__(self, output_rot, target_rot, output_contrastive, target_contrastive, output_recons, target_recons):
        rot_loss = self.alpha1 * self.rot_loss(output_rot, target_rot)
        contrast_loss = self.alpha2 * self.contrast_loss(output_contrastive, target_contrastive)
        recon_loss = self.alpha3 * self.recon_loss(output_recons, target_recons)
        total_loss = rot_loss + contrast_loss + recon_loss

        return total_loss, (rot_loss, contrast_loss, recon_loss)
