#!/bin/bash -l
## Job name
#SBATCH -J multi_gpu_test  
## Number of allocated nodes
#SBATCH -N 2
## Tasks per node, (one task per one GPU)
#SBATCH --ntasks-per-node=2
## Number of cores per task
#SBATCH --cpus-per-task=6
## Amount of RAM per CPU
#SBATCH --mem-per-cpu=4GB
## Max time of the job hours:minutes:seconds
#SBATCH --time=1:00:00
## Name of the grant for resource usage (FOR METWEEN IT WILL BE DIFFERENT)
#SBATCH -A plgdyplomanci5-gpu-a100
## Choosing partition
#SBATCH --partition plgrid-gpu-a100
## Number of GPUs per node
#SBATCH --gpus-per-node=2
## STDOUT file
#SBATCH --output="/net/tscratch/people/plgmazurekagh/meetween_test/output_files/out/out_test.out"
## Plik ze standardowym wyjściem błędó0w
#SBATCH --error="/net/tscratch/people/plgmazurekagh/meetween_test/output_files/err/eeg_test.err"

## Load necessary modules
ml CUDA/11.8

# move to the directory with the code
cd $SCRATCH/meetween_test
# Copy the dataset to the directory from group storage
# WARNING! For the Meetween project, the folder in group storage will be different!
cp -r $PLG_GROUPS_STORAGE/plggtraining/data ./dataset
echo "Dataset copied to the local storage"
# some configuration for the torchrun
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) ) # this command gives us all nodes in the job
nodes_array=($nodes) # convert to array
head_node=${nodes_array[0]} # get the first node - this will be our main process
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address) # get the IP address of the first node
echo Node IP: $head_node_ip # print the IP address of the main node
export LOGLEVEL=INFO # set the log level to INFO
# activate the conda environment if needed
conda activate meetween_env/
# Launch the training script
# note that for our main process ip we use the head_node_ip and default port
# sometimes this port can be unavailable, so it will need modification
srun torchrun \
    --nnodes $SLURM_NNODES \
    --nproc_per_node $SLURM_NTASKS_PER_NODE \
    --rdzv_id $RANDOM \
    --rdzv_backend c10d \
    --rdzv_endpoint $head_node_ip:29500 \
    torch_distributed_example_multinode.py  50 10 --dataset_path ./dataset

# Copy the results to the group storage - the script saved the example snapshots in the current directory
# We will copy them to the group storage as an example
mkdir -p $PLG_GROUPS_STORAGE/plggtraining/snapshots # create the directory if it does not exist
cp snapshot.pt $PLG_GROUPS_STORAGE/plggtraining/snapshots/snapshot.pt

# If no more experiments are planned in the upcoming time, remove the dataset
# from $SCRATCH filesystem
rm -r ./dataset