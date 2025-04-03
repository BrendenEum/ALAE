#######################
# Interactive Debugging
#######################

# Start an interactive session for debugging
salloc --account=def-webbr  --time=00:59:00 --gres=gpu:1 --mem=16G --ntasks=1 --cpus-per-task=2

#######################
# Setting up the compute node
#######################

# Go to project folder
# git clone https://github.com/BrendenEum/ALAE
#cd projects/def-webbr/beum/ALAE 
cd scratch/ALAE 

# Load older modules 
module load StdEnv/2020
module load python/3.8.10
module load cuda/11.0
# path to this cuda library
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"/cvmfs/soft.computecanada.ca/easybuild/software/2020/Core/cudacore/11.0.2/lib/"
# random_device throws an error on cluster, tell it fall back to this pseudo-RNG
# export CXXFLAGS="-D_GLIBCXX_USE_RANDOM_TR1" 

#######################
# Setting up the virtual environment
#######################

# Make and activate venv
# virtualenv --no-download env
source env/bin/activate

# Relax the restrinctions on where to download from.
unset PIP_CONFIG_FILE
unset PYTHONPATH

# Manually load all the packages. 
pip install -r requirements.txt

#######################
# Jobs
#######################

# Dataset prep
python -m dataset_preparation.prepare_evox_256x256_1-3

# Train
python train_alae.py -c evox_256x256_1-3

#######################
# Monitor progress and stress
#######################

# Check output if you sbatch
tail -f job-logs/train_alae-#.out

# Watch GPU usage (in a separate terminal, ssh into the compute node)
ssh beum@cedar.alliancecan.ca
ssh beum@{MACHINE NAME}
watch -n 1 nvidia-smi