# Brenden's README

Step by step instructions for any idiot.

## Getting Started

StyleALAE was made in 2020, so you'll need to create a virtual environment with Python 3.8 before installing dependencies.
I have significantly altered requirements.txt (called "requirements_BE.txt") to make sure dependency versions are compatible with eachother.

```bash
cd C:\Users\Brend\OneDrive\Desktop\ALAE
conda create -n alae python=3.8
conda activate alae
pip install -r requirements_BE.txt
```

I had to take out torch and torchvision from requirements. This is so we can install them with CUDA 11.1 support.

```bash
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
```

To download the pre-trained models and try out the demo:

```bash
python training_artifacts/download_all.py
python interactive_demo.py -c celeba-hq256
```


## Prepare dataset

dataset_preparation/prepare_evox.py was written using ChatGPT. Run it from the root directory. It takes the raw .png images from data/datasets/evox/cars, resizes them, converts them to 128x128, then saves them in a tfrecords file.

Note: You can add labels later in dataset_preparation/prepare_evox.py when you write images to TFrecord.

```bash
python -m dataset_preparation.prepare_evox
```


## Train network

```bash
python train_alae.py -c evox
```


```bash
cd C:\Users\Brend\OneDrive\Desktop\ALAE
conda activate alae
python train_alae.py -c evox
```