#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
##SBATCH --gpus-per-node=8
##SBATCH --gpus-per-node=4
#SBATCH --gpus-per-node=2
#SBATCH --constraint=v100
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%A.out
#SBATCH --error=slurm-%A.out

# load conda environment
. /ibex/ai/home/zampins/miniforge/bin/activate fm4g

MAIN=/ibex/ai/home/zampins/fm4g/monai-research-contributions/SwinUNETR/Pretrain/main.py

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
HOST_NODE_ADDR=$head_node_ip:29500

BS=2 # mini-batch size per device (BS=4 OOMs on V100)
SWBS=2 # sliding-window mini-batch size per device (not sure what does it mean)

# -> (total mini-batch size is nodes * bs *sw_bs)
STEPS=100000 # how many steps to take

EVALEVERY=100 # test validation every given number of steps

OPTOPTS="--lrdecay --lr=1.e-4 --decay=1.e-1 --momentum=0.9 --opt=adamw --lr_schedule=warmup_cosine" # new optimizer options

# See /project/k10123/datasets/generate_geo_1.py
# scale caterogical to [0,1]
AMIN=0
AMAX=14
IMAGE_TRANSFORM_OPTIONS="--a_min $AMIN --a_max $AMAX"

# input/output channels
IN_CHANNELS=1
#OUT_CHANNELS=15
OUT_CHANNELS=1
HEAD_OPTIONS="--in_channels=$IN_CHANNELS --out_channels=$OUT_CHANNELS"

DATADIR="/ibex/ai/home/zampins/fm4g/datasets/test-geo-1000" # where to find datasets and jsons
JSONLIST="dataset_0.json" # comma separated list of json filenames
DATASETLIST="dataset" # comma separated list of folder name where datasets are stored

# if you want to resume from previously trained weights
# RESUME="--resume=/scratch/zampins/fm4g/first-tests-runs/1407742/model_bestValRMSE.pt"

# if you just want to dump images
#DUMPIMG="--check_images"

# Some more informative name for the output dir
JOB_SUFFIX=_1000_newparams

LOGDIR=/ibex/ai/home/zampins/fm4g/logs/${SLURM_JOB_ID}${JOB_SUFFIX} # where to dump the intermediate checkpoints

# Python script with arguments
SCRIPT="$MAIN --num_workers 4 --num_steps=$STEPS $OPTOPTS --eval_num=$EVALEVERY --batch_size=$BS --sw_batch_size=$SWBS --datadir=$DATADIR --logdir=$LOGDIR --jsonlist=$JSONLIST --datasetlist=$DATASETLIST $IMAGE_TRANSFORM_OPTIONS $HEAD_OPTIONS $RESUME $DUMPIMG"

# SRUN command (1 torchrun instance per node, torchrun does the multi-processing on node)
SRUN_CMD="srun -n $SLURM_NNODES -c $SLURM_CPUS_ON_NODE --ntasks-per-node 1 torchrun --max_restarts=0 --nnodes=$SLURM_NNODES --nproc-per-node=$SLURM_GPUS_PER_NODE --rdzv-id=$SLURM_JOB_ID --rdzv-backend=c10d --rdzv-endpoint=$HOST_NODE_ADDR $SCRIPT"

mkdir -p $LOGDIR
echo $SRUN_CMD > $LOGDIR/joboutput.log
$SRUN_CMD 2>&1 >> $LOGDIR/joboutput.log
