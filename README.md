# ImageRestorationPromptIR

## Introduction

A "All-in One" model by PromptIR for rain or snow degraded image restoration.

## Environment

On colab: Should be able to train with batch size 4.

And there are the used lib:
```
einops==0.8.1
fonttools==4.58.0
imageio==2.37.0
kaggle==1.7.4.5
matplotlib==3.10.3
numpy==2.2.6
pillow==11.2.1
scikit-image==0.25.2
torch==2.7.0
torchvision==0.22.0
tqdm==4.67.1
```

## Install

1. download
2. upload to colab
3. download the dataset
4. upload and name it as ```dataset```
5. run the following command:
```bash
!python3 main.py ../dataset
```

## Usage

```bash
python main.py <data-directory> [options]
```

**Needed:**

`<data-directory>`: The path to where you store data, the directory should be like the following:

```
--root
  |--train
  |  |--clean
  |  |  |--rain_clean-1.png
  |  |  |  ...
  |  |  |--snow_clean-1.png
  |  |  |  ...
  |  |--degraded
  |  |  |--rain-1.png
  |  |  |  ...
  |--test
  |  |--degraded
  |  |  |--1.png
  |  |  ...
```

**Options:**

* `--depthwise-separable`: Enable depthwise separable conv for a little bit faster training.

* `--loss-fn`: Choose l1 (nn.L1Loss) and l2 (nn.MSELoss). Default='l1'.

* `--optimizer`: Choose ```SGD``` and ```AdamW```. Default='SGD'.

* `--lr`: Learning rate. Default=0.0002

* `--weight-decay`: Weight decay. Default=0.0005

* `--scheduler-step`: Step size for the learning rate scheduler. Default=5

* `--scheduler-rate`: Rate to multiply the learning rate by at each scheduler step. Default=0.31622776601

* `--epoch`: Number of training epochs. Default=16

* `--batch`: Batch size for training. Default=4

* `--seed`: Random seed for reproducibility. Default=42

**Example Command (Configuration yielding best result):**

```bash
python main.py /path/to/your/data --epoch 12 --batch 4 --seed 42 --optimizer 'AdamW' --lr 0.0001 --weight-decay 0.01
```
The program will run training, validation and testing individually. The submission file for the test dataset will be saved as `test-results.json`.

## Results

Test PSNR: 0.28
