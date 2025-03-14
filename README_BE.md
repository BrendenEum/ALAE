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



