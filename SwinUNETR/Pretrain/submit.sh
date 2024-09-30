#!/bin/bash
#SBATCH --partition=72hours
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=192
#SBATCH --hint=nomultithread
#SBATCH --account=k10123
#SBATCH --time=72:00:00
#SBATCH --output=slurm-%A.out
#SBATCH --error=slurm-%A.out

# load conda environment
. /scratch/zampins/iops/sw/miniforge3/bin/activate fm4g

MAIN=/project/k10123/monai-research-contributions/SwinUNETR/Pretrain/main.py

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
HOST_NODE_ADDR=$head_node_ip:29500

NUM_THREADS=96 # number of CPU threads to use in pytorch

BS=4 # mini-batch size per node
SWBS=2 # sliding-window mini-batch size per node (not sure what does it mean)

# -> (total mini-batch size is nodes * bs *sw_bs)
STEPS=100000 # how many steps to take

EVALEVERY=50 # test validation every given number of steps

OPTOPTS="--lrdecay --lr=4e-4 --decay=1.e-5 --momentum=0.9 --opt=adamw --lr_schedule=warmup_cosine" # optimizer options (https://arxiv.org/pdf/2111.14791)

# See /project/k10123/datasets/generate_geo_1.py
# scale caterogical to [0,1]
AMIN=0
AMAX=14
IMAGE_TRANSFORM_OPTIONS="--a_min $AMIN --a_max $AMAX"

# input/output classes
IN_CHANNELS=1
OUT_CHANNELS=15
HEAD_OPTIONS="--in_channels=$IN_CHANNELS --out_channels=$OUT_CHANNELS"

DATADIR="/project/k10123/datasets/test-geo-100" # where to find datasets and jsons
JSONLIST="dataset_0.json" # comma separated list of json filenames
DATASETLIST="dataset" # comma separated list of folder name where datasets are stored

#DATADIR="/project/k10123/datasets/SwinUNETR-Pretrain"
#JSONLIST="dataset_LUNA16_0.json"
#DATASETLIST="dataset1"

# if you want to resume from previously trained weights
# RESUME="--resume=/scratch/zampins/fm4g/first-tests-runs/1407742/model_bestValRMSE.pt"

# if you just want to dump images
#DUMPIMG="--check_images"

# Some more informative name for the output dir
JOB_SUFFIX=_focal

LOGDIR=/scratch/zampins/fm4g/first-tests-runs/${SLURM_JOB_ID}${JOB_SUFFIX} # where to dump the intermediate checkpoints

# Python script with arguments
SCRIPT="$MAIN --num_steps=$STEPS $OPTOPTS --eval_num=$EVALEVERY --batch_size=$BS --sw_batch_size=$SWBS --datadir=$DATADIR --logdir=$LOGDIR --jsonlist=$JSONLIST --datasetlist=$DATASETLIST $IMAGE_TRANSFORM_OPTIONS $HEAD_OPTIONS $RESUME $DUMPIMG"

mkdir -p $LOGDIR
echo "Running LD_PRELOAD=/project/k10123/local/lib/libfakeintel.so MKL_NUM_THREADS=$NUM_THREADS srun -n $SLURM_NNODES -c 192 --ntasks-per-node 1 --hint=nomultithread --cpu-bind=cores torchrun --max_restarts=0 --nnodes=$SLURM_NNODES --nproc-per-node=1 --rdzv-id=$SLURM_JOB_ID --rdzv-backend=c10d --rdzv-endpoint=$HOST_NODE_ADDR $SCRIPT" > $LOGDIR/joboutput.log
LD_PRELOAD=/project/k10123/local/lib/libfakeintel.so MKL_NUM_THREADS=$NUM_THREADS srun -n $SLURM_NNODES -c 192 --ntasks-per-node 1 --hint=nomultithread --cpu-bind=cores torchrun --max_restarts=0 --nnodes=$SLURM_NNODES --nproc-per-node=1 --rdzv-id=$SLURM_JOB_ID --rdzv-backend=c10d --rdzv-endpoint=$HOST_NODE_ADDR $SCRIPT 2>&1 >> $LOGDIR/joboutput.log
