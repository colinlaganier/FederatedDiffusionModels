
#!/bin/bash -l

# Batch script to run a GPU job under SGE.

# Request a number of GPU cards, in this case 2 (the maximum)
#$ -l gpu=1

# Request ten minutes of wallclock time (format hours:minutes:seconds).
#$ -l h_rt=24:00:0

# Request 1 gigabyte of RAM (must be an integer followed by M, G, or T)
#$ -l mem=32G

# Request 15 gigabyte of TMPDIR space (default is 10 GB)
#$ -l tmpfs=32G

# Set the name of the job.
#$ -N DDPM_Fed_10_100

# Set the working directory to somewhere in your scratch space
#$ -wd /home/ucabcuf/Scratch/FederatedDiffusionModels

# Change into temporary directory to run work
# cd $TMPDIR

# load the cuda module (in case you are running a CUDA program)
module purge
module load default-modules
module unload compilers mpi
module load gcc-libs/4.9.2
module load python/miniconda3/4.10.3

# Activate conda environment
source $UCL_CONDA_PATH/etc/profile.d/conda.sh
conda activate FedKDD

# Run the application
nvidia-smi
cd DDPM
sh run.sh -c 5 -r 100 -e 1