# Start an interactive session for debugging
salloc --account=def-webbr  --time=04:00:00 --gres=gpu:1 --mem=16G --ntasks=1 --cpus-per-task=2
salloc --account=def-webbr  --time=01:00:00 --gres=gpu:1 --mem=16G --ntasks=1 --cpus-per-task=1
# Go to project folder
# git clone https://github.com/BrendenEum/ALAE
cd projects/def-webbr/beum/ALAE 

# Load older modules 
module load StdEnv/2020
module load python/3.8.10
module load cuda/11.0
export LD_LIBRARY_PATH="/cvmfs/soft.computecanada.ca/easybuild/software/2020/Core/cudacore/11.1.1/lib/"
export CXXFLAGS="-D_GLIBCXX_USE_RANDOM_TR1" 

#export LD_LIBRARY_PATH="/cvmfs/soft.computecanada.ca/easybuild/software/2020/Core/cudacore/11.0.2/lib/"
#libcusolver.so.11 is in LD_LIBRARY_PATH="/cvmfs/soft.computecanada.ca/easybuild/software/2020/Core/cudacore/11.1.1/lib/"
#libcudnn.so.8 is in LD_LIBRARY_PATH="/cvmfs/soft.computecanada.ca/easybuild/software/2020/CUDA/cuda11.1/cudnn/8.2.0/lib/"

# path to this cuda library
# random_device throws an error on cluster, tell it fall back to this pseudo-RNG

# Make and activate venv
# virtualenv --no-download env
source env/bin/activate

# Relax the restrinctions on where to download from.
unset PIP_CONFIG_FILE
unset PYTHONPATH

# Manually load all the packages. 
pip install -r requirements.txt

# Dataset prep
python -m dataset_preparation.prepare_evox

# Train
python train_alae.py -c evox

# Check output if you sbatch
tail -f job-logs train_alae-#.out

# Separate terminal. Check how much my job is stressing the GPU and CPU.
ssh beum@cedar.alliancecan.ca
ssh beum@{MACHINE NAME}
nvidia-smi