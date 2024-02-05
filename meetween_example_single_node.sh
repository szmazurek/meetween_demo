#!/bin/bash -l
## Job name
#SBATCH -J multi_gpu_test  
## Number of allocated nodes
#SBATCH -N 1
## Tasks per node, (one task per one GPU)
#SBATCH --ntasks-per-node=2
## Number of cores per task
#SBATCH --cpus-per-task=6
## Amount of RAM per CPU
#SBATCH --mem-per-cpu=4GB
## Max time of the job hours:minutes:seconds
#SBATCH --time=1:00:00
## Name of the grant for resource usage (FOR MEETWEEN IT WILL BE DIFFERENT)
#SBATCH -A plgdyplomanci5-gpu-a100
## Choosing partition
#SBATCH --partition plgrid-gpu-a100
## Number of GPUs per node
#SBATCH --gpus-per-node=2
## STDOUT file
#SBATCH --output="/net/tscratch/people/plgmazurekagh/meetween_test/output_files/out/out_test.out"
## STDERR file
#SBATCH --error="/net/tscratch/people/plgmazurekagh/meetween_test/output_files/err/eeg_test.err"

## Load necessary modules
ml CUDA/11.8

# move to the directory with the code
cd $SCRATCH/meetween_test
# Copy the dataset to the directory from group storage
# WARNING! For the Meetween project, the folder in group storage will be different!
# cp -r $PLG_GROUPS_STORAGE/plggtraining/data ./dataset
echo "Dataset copied to the local storage"

export LOGLEVEL=INFO # set the log level to INFO
# activate the conda environment if needed
conda activate metween_env/
# Launch the training script
# note that for our main process ip we use the head_node_ip and default port
# sometimes this port can be unavailable, so it will need modification
srun torchrun \
    --standalone \
    --nproc_per_node $SLURM_NTASKS_PER_NODE \
    torch_distributed_example_single_node.py  50 10 --dataset_path ./dataset

# Copy the results to the group storage - the script saved the example snapshots in the current directory
# We will copy them to the group storage as an example
mkdir -p $PLG_GROUPS_STORAGE/plggtraining/snapshots # create the directory if it does not exist
cp snapshot.pt $PLG_GROUPS_STORAGE/plggtraining/snapshots/snapshot.pt

# If no more experiments are planned in the upcoming time, remove the dataset
# from $SCRATCH filesystem
rm -r ./dataset