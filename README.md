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

* `--model-version`: Choose v1 (maskrcnn_resnet50_fpn) and v2 (maskrcnn_resnet50_fpn_v2). Default='v1'.

* `--freeze`: The number of layers you DON'T WANT to freeze. Default=3, integer between 1 to 6.

* `--instance-num`: Maximum number of instances to predict per image during inference. Default=100.

* `--lr`: Learning rate. Default=0.0002

* `--scheduler-step`: Step size for the learning rate scheduler. Default=5

* `--scheduler-rate`: Rate to multiply the learning rate by at each scheduler step. Default=0.31622776601

* `--epoch`: Number of training epochs. Default=16

* `--batch`: Batch size for training. Default=4

* `--seed`: Random seed for reproducibility. Default=42

**Example Command (Configuration yielding best result):**

```bash
python main.py /path/to/your/data --epoch 12 --scheduler-step 3 --batch 1 --seed 774 --predict-instance-num 256 --model-version 'v1'
```
The program will run training, validation and testing individually. The submission file for the test dataset will be saved as `test-results.json`.

## Results

Val AP@0.50-0.95: 0.35

Val AP@0.50: 0.50

Test AP@0.50-0.95: 0.4054
