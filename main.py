import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms.v2 as transformsT
import torchvision.transforms.functional as transformsF
# from torchvision.transforms import ToPILImage, Compose, RandomCrop
from tqdm import tqdm
from time import time
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import matplotlib.pyplot as plt

from models import PromptIRModel
from datasets import TrainDataset, TestDataset
from utils.data_io import load_json, save_json, save_images2npz
from utils.image_utils import show_diff


class ToTensor:
    def __call__(self, image, target=None):
        # # detact the type
        # dtype = image.dtype
        # # convert to tensor
        # type_mapping = {
        #     'uint8': torch.uint8,
        #     'int32': torch.int32,
        #     'float16': torch.float16,
        #     'float32': torch.float32,
        #     'float64': torch.float64,
        # }
        if isinstance(image, Image.Image):
            image = np.array(image)

        image = transformsF.to_tensor(image)
        # image as_array
        # image = np.moveaxis(image, -1, 0)
        # image = torch.asarray(image, dtype=torch.float16) / 255.0
        # print(image)

        if target is None:
            return image


        if isinstance(image, Image.Image):
            target = np.array(target)

        target = transformsF.to_tensor(target)

        return image, target

# 定義影像轉換 (資料增強等)
def get_transform(train, crop_size=128):
    transforms_list = []
    # 將 PIL Image 或 Tensor 轉換為 Tensor
    if train:
        transforms_list.append(transformsT.ToPILImage())
        transforms_list.append(transformsT.RandomCrop(size=crop_size))
        # transforms_list.append(transformsT.RandomHorizontalFlip())

    transforms_list.append(ToTensor())

    return transformsT.Compose(transforms_list)


if __name__ == '__main__':
    # 避免 GPU 記憶體碎片化，導致記憶體空間不足
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    # parse arguments
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'directory',
        help='The directory where you save all the data. '
        + 'Should include train, val, test.'
    )
    parser.add_argument('--depthwise-separable', action='store_true')
    parser.add_argument('--instance-normalize', action='store_true')
    parser.add_argument(
        '--loss-fn', type=str, default='l1',
        choices=['l1', 'l2'],
        help='Choose loss function for training.'
    )
    parser.add_argument(
        '--optimizer', type=str, default='SGD',
        choices=['SGD', 'AdamW'],
        help='Choose optimizer for training.'
    )
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--weight-decay', type=float, default=0.0005)
    parser.add_argument('--scheduler-step', type=int, default=5)
    parser.add_argument('--scheduler-rate', type=float, default=0.31622776601)
    parser.add_argument('--epoch', type=int, default=16)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    # data
    data_root = args.directory
    train_data_dir = os.path.join(data_root, 'train')
    test_data_dir = os.path.join(data_root, 'test')

    # model
    loss_fn = args.loss_fn
    depthwise_separable = args.depthwise_separable
    instance_normalize = args.instance_normalize

    # hyperparameter
    optimizer_name = args.optimizer
    lr = args.lr
    weight_decay = args.weight_decay
    scheduler_step = args.scheduler_step
    scheduler_rate = args.scheduler_rate

    # others
    epoch_num = args.epoch
    batch_size = args.batch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device: ', device)

    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)             # cpu
    torch.cuda.manual_seed(seed)        # current gpu
    torch.cuda.manual_seed_all(seed)    # all gpu

    debug = args.debug


    # 訓練和測試資料集
    train_dataset = TrainDataset(train_data_dir, get_transform(train=True))
    test_dataset = TestDataset(test_data_dir, get_transform(train=False))

    # 隨機分割出訓練和驗證資料集
    data_size = len(train_dataset)
    train_size = int(data_size * 0.95)
    valid_size = data_size - train_size
    train_dataset, valid_dataset = random_split(
        train_dataset,
        (train_size, valid_size)
    )

    train_data_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=2
    )
    valid_data_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False,
        num_workers=2
    )

    test_data_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        num_workers=2
    )


    model = PromptIRModel(device, loss_fn=loss_fn,
                          depthwise_separable_enable=depthwise_separable,
                          instance_normalize_enable=instance_normalize)
    # Your model size (trainable parameters) should less than 200M.
    print('parameters: ', sum(p.numel() for p in model.parameters()
                                            if p.requires_grad))

    params = [p for p in model.parameters() if p.requires_grad]
    if optimizer_name == 'SGD':
        optimizer = torch.optim.SGD(params,
                                    lr=lr, momentum=0.9,
                                    weight_decay=weight_decay)
    else:
        optimizer = torch.optim.AdamW(params,
                                      lr=lr,
                                      weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=scheduler_step,
                                                gamma=scheduler_rate)

    num_epochs = epoch_num
    training_losses = []
    validating_losses = []
    train_start_time = time()
    for epoch in range(num_epochs):
        print(f'Epoch #{epoch+1}')
        print(f'current lr: {scheduler.get_last_lr()}')

        result = model.train(train_data_loader, optimizer, debug=debug)
        loss = result['avg_loss']
        training_losses.append(loss)
        print(f'training loss: {loss}')

        scheduler.step()

        result = model.validate(valid_data_loader)
        loss = result['avg_loss']
        validating_losses.append(loss)
        print(f'validating losses: {loss}')

        originals = result['original_images']
        degradeds = result['degraded_images']
        restoreds = result['restored_images']

        mask = np.random.randint(0, len(originals), size=5)
        masked_originals = originals[mask]
        masked_degradeds = degradeds[mask]
        masked_restoreds = restoreds[mask]

        for i in range(len(masked_originals)):
            original = masked_originals[i].transpose(1, 2, 0)
            degraded = masked_degradeds[i].transpose(1, 2, 0)
            restored = masked_restoreds[i].transpose(1, 2, 0)
            show_diff(degraded, restored, f'epoch_{epoch+1}_{i}_restored')
            show_diff(original, restored, f'epoch_{epoch+1}_{i}_rest')

        # break

    train_end_time = time()
    model.save_model('model.pth')
    print(f'{num_epochs} 個 epoch，共計 {train_end_time - train_start_time} 秒')

    plt.plot(training_losses, 'r-', label='training loss')
    plt.plot(validating_losses, 'g-', label='validating loss')
    plt.legend()
    plt.savefig('runtime_stat.png')
    plt.show()
    plt.close()

    # test_dataset
    model.load_model('model.pth')
    result = model.test(test_data_loader)
    data = []
    for datum in zip(result['names'], result['restored_images']):
        data.append(datum)
    save_images2npz('./pred.npz', data)
