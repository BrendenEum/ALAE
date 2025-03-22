#!/bin/bash
#SBATCH --account=def-webbr
#SBATCH --time=0-12:00:00 # DD-HH:MM:SS
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=32000M
#SBATCH --output=job-logs/train_alae-%j.out

# Load older modules
module load StdEnv/2020
module load python/3.8.10
module load cuda/11.0
export LD_LIBRARY_PATH=/cvmfs/soft.computecanada.ca/easybuild/software/2020/Core/cudacore/11.0.2/lib/ 
export CXXFLAGS="-D_GLIBCXX_USE_RANDOM_TR1"

# Go to project folder
cd projects/def-webbr/beum/ALAE/

# Activate virtual environment for StyleALAE
source env/bin/activate

# Dataset prep
# python -m dataset_preparation.prepare_evox

# Train
python train_alae.py -c evox
