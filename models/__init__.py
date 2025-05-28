#  @title # models/\_\_init\_\_
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
from tqdm import tqdm

from models.model import PromptIR


class PromptIRModel:
    def __init__(self, device, loss_fn='l1', lr=0.1,
                 scheduler_step=5, scheduler_rate=0.1,
                 depthwise_separable_enable=False,
                 instance_normalize_enable=False):
        print('__init__')

        self.device = device
        self.model = PromptIR(
            decoder=True,
            depthwise_separable_enable=depthwise_separable_enable,
            instance_normalize_enable=instance_normalize_enable)
        # self.model = self.model.to(self.device, dtype=torch.float16)
        self.model = self.model.to(self.device)

        loss_fn_dict = {
            'l1': nn.L1Loss(),
            'l2': nn.MSELoss(),
        }
        # loss = l1_weight * L1Loss(predicted, target) + l2_weight * L2Loss(predicted, target)
        self.criterion = loss_fn_dict[loss_fn]

    def parameters(self):
        return self.model.parameters()

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))

    def train(self, dataloader, optimizer, debug=False):
        print('train')

        self.model.train()

        total_loss = 0
        total = 0
        original_images = []
        degraded_images = []
        restored_images = []

        batch_num = len(dataloader)
        for images, targets in tqdm(dataloader, total=batch_num,
                                    desc='Training', unit='batch',
                                    position=0, leave=True):
            images = images.to(self.device)
            targets = targets.to(self.device)

            # Forward pass
            # torch.cuda.synchronize(self.device) # 等待 GPU 結束
            outputs = self.model(images)

            # Compute the loss
            loss = self.criterion.forward(outputs, targets)
            total_loss += loss.item() * len(images)
            total += len(images)

            # Zero the gradients before the backward pass
            optimizer.zero_grad()

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # 移到 CPU 處理
            images = images.detach().cpu().numpy()
            targets = targets.detach().cpu().numpy()
            outputs = outputs.detach().cpu().numpy()

            original_images.append(targets)
            degraded_images.append(images)
            restored_images.append(outputs)

            # free GPU memory
            del images, targets, outputs, loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if debug and total > 100:
                break
            # break

        original_images = np.concatenate(original_images, axis=0)
        degraded_images = np.concatenate(degraded_images, axis=0)
        restored_images = np.concatenate(restored_images, axis=0)

        result = {
            'avg_loss': total_loss / total,
            'original_images': original_images,
            'degraded_images': degraded_images,
            'restored_images': restored_images
        }

        return result

    @torch.no_grad()
    def validate(self, dataloader):
        print('validate')

        self.model.eval()

        total_loss = 0
        total = 0
        original_images = []
        degraded_images = []
        restored_images = []

        batch_num = len(dataloader)
        for images, targets in tqdm(dataloader, total=batch_num,
                                    desc='Validating', unit='batch',
                                    position=0, leave=True):
            images = images.to(self.device, dtype=torch.float32)
            targets = targets.to(self.device, dtype=torch.float32)

            # torch.cuda.synchronize(self.device) # 等待 GPU 結束
            outputs = self.model(images)

            loss = self.criterion.forward(outputs, targets)
            total_loss += loss.item() * len(images)
            total += len(images)

            # 移到 CPU 處理
            images = images.detach().cpu().numpy()
            targets = targets.detach().cpu().numpy()
            outputs = outputs.detach().cpu().numpy()
            outputs = np.clip(outputs, 0, 1)

            original_images.append(targets)
            degraded_images.append(images)
            restored_images.append(outputs)

            # free GPU memory
            del images, targets, outputs, loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        original_images = np.concatenate(original_images, axis=0)
        degraded_images = np.concatenate(degraded_images, axis=0)
        restored_images = np.concatenate(restored_images, axis=0)

        result = {
            'avg_loss': total_loss / total,
            'original_images': original_images,
            'degraded_images': degraded_images,
            'restored_images': restored_images
        }

        return result

    @torch.no_grad()
    def test(self, dataloader):
        print('test')

        self.model.eval()

        names = []
        degraded_images = []
        restored_images = []

        batch_num = len(dataloader)
        for images, targets in tqdm(dataloader, total=batch_num,
                                    desc='Testing', unit='batch',
                                    position=0, leave=True):
            images = images.to(self.device, dtype=torch.float32)
            outputs = self.model(images)

            images = images.detach().cpu().numpy()
            outputs = outputs.detach().cpu().numpy()
            outputs = np.clip(outputs, 0, 1)

            names.extend(targets)
            degraded_images.append(images)
            restored_images.append(outputs)

            # free GPU memory
            del images, outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        degraded_images = np.concatenate(degraded_images, axis=0)
        restored_images = np.concatenate(restored_images, axis=0)

        result = {
            'names': names,
            'degraded_images': degraded_images,
            'restored_images': restored_images
        }

        return result
