'''
 * Copyright (c) 2023, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Le Xue
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from deepspeed.runtime.activation_checkpointing.checkpointing import cuda_device

from utils import utils

class ULIPWithImageLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.labels = None
        self.last_local_batch_size = None

    def forward(self, outputs):
        pc_embed = outputs['pc_embed']
        text_embed = outputs['text_embed']
        image_embed = outputs['image_embed']
        logit_scale = outputs['logit_scale']
        local_batch_size = pc_embed.size(0)

        if local_batch_size != self.last_local_batch_size:
            self.labels = local_batch_size * utils.get_rank() + torch.arange(
                local_batch_size, device=pc_embed.device
            )
            self.last_local_batch_size = local_batch_size

        # normalized features
        pc_embed = F.normalize(pc_embed, dim=-1, p=2)
        text_embed = F.normalize(text_embed, dim=-1, p=2)
        image_embed = F.normalize(image_embed, dim=-1, p=2)

        # gather features from all GPUs
        pc_embed_all, text_embed_all, image_embed_all = \
            utils.all_gather_batch([pc_embed, text_embed, image_embed])

        # cosine similarity as logits
        logits_per_pc_text = logit_scale * pc_embed @ text_embed_all.t()
        logits_per_text_pc = logit_scale * text_embed @ pc_embed_all.t()
        logits_per_pc_image = logit_scale * pc_embed @ image_embed_all.t()
        logits_per_image_pc = logit_scale * image_embed @ pc_embed_all.t()

        loss = (F.cross_entropy(logits_per_pc_text, self.labels) + \
                F.cross_entropy(logits_per_text_pc, self.labels)) / 2 + \
                (F.cross_entropy(logits_per_pc_image, self.labels) + F.cross_entropy(logits_per_image_pc, self.labels)) / 2

        # compute accuracy
        with torch.no_grad():
            pred = torch.argmax(logits_per_pc_text, dim=-1)
            correct = pred.eq(self.labels).sum()
            pc_text_acc = 100 * correct / local_batch_size

            pred = torch.argmax(logits_per_pc_image, dim=-1)
            correct = pred.eq(self.labels).sum()
            pc_image_acc = 100 * correct / local_batch_size

        return {'loss': loss, 'ulip_loss': loss, 'ulip_pc_image_acc': pc_image_acc, 'ulip_pc_text_acc': pc_text_acc}

class ULIPWithoutImageLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.labels = None
        self.last_local_batch_size = None

    def forward(self, outputs, labels=None):
        pc_embed = outputs['pc_embed']
        text_embed = outputs['text_embed']
        logit_scale = outputs['logit_scale']
        local_batch_size = pc_embed.size(0)

        if local_batch_size != self.last_local_batch_size:
            self.labels = local_batch_size * utils.get_rank() + torch.arange(
                local_batch_size, device=pc_embed.device
            )
            self.last_local_batch_size = local_batch_size

        print(f"self.labels batch size: {self.labels.size(0)}")

        # normalized features
        pc_embed = F.normalize(pc_embed, dim=-1, p=2)
        text_embed = F.normalize(text_embed, dim=-1, p=2)

        # gather features from all GPUs
        pc_embed_all, text_embed_all = \
            utils.all_gather_batch([pc_embed, text_embed])

        print(pc_embed.shape)
        print(text_embed.shape)
        print(pc_embed_all.shape)
        print(text_embed_all.shape)

        # cosine similarity as logits
        logits_per_pc_text = logit_scale * pc_embed @ text_embed_all.t()
        logits_per_text_pc = logit_scale * text_embed @ pc_embed_all.t()

        print(f"logits_per_pc_text: {logits_per_pc_text.size()}")
        print(f"logits_per_text_pc: {logits_per_text_pc.size()}")

        # 确保logits和labels的batch size一致
        assert logits_per_pc_text.size(0) == self.labels.size(0), \
            f"Batch size mismatch: logits_per_pc_text {logits_per_pc_text.size(0)}, labels {self.labels.size(0)}"
        assert logits_per_text_pc.size(0) == self.labels.size(0), \
            f"Batch size mismatch: logits_per_text_pc {logits_per_text_pc.size(0)}, labels {self.labels.size(0)}"

        loss = (F.cross_entropy(logits_per_pc_text, self.labels) + F.cross_entropy(logits_per_text_pc, self.labels)) / 2

        # compute accuracy
        with torch.no_grad():
            pred = torch.argmax(logits_per_pc_text, dim=-1)
            correct = pred.eq(self.labels).sum()
            pc_text_acc = 100 * correct / local_batch_size


        return {'loss': loss, 'ulip_loss': loss, 'ulip_pc_text_acc': pc_text_acc}


class ULIPClsLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, label):
        pc_embed = outputs['pc_embed']
        text_embed = outputs['text_embed']

        logit_scale = outputs['logit_scale']
        local_batch_size = pc_embed.size(0)

        pc_embed = pc_embed / pc_embed.norm(dim=-1, keepdim=True)
        text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)

        logits = logit_scale * pc_embed @ text_embed.t()


        label = torch.tensor(label).to(logits.device)

        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, label)

        # compute accuracy
        with torch.no_grad():
            pred = torch.argmax(logits, dim=-1)
            correct = pred.eq(label).sum()
            acc = 100 * correct / local_batch_size


        return {'loss': loss, 'ulip_loss': loss, 'ulip_pc_text_acc': acc}
