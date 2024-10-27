#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
##SBATCH --gpus-per-node=8
##SBATCH --gpus-per-node=4
##SBATCH --gpus-per-node=2
#SBATCH --constraint=a100
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%A.out
#SBATCH --error=slurm-%A.out
##SBATCH --time=2:00:00
##SBATCH --partition=debug
#SBATCH --gpus-per-node=1

# load conda environment
. /ibex/ai/home/zampins/miniforge/bin/activate fm4g

MAIN=/ibex/ai/home/zampins/fm4g/monai-research-contributions/SwinUNETR/Pretrain/main.py

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
HOST_NODE_ADDR=$head_node_ip:29500

BS=4 # mini-batch size per device (BS=2 OOMs on V100)
SWBS=2 # sliding-window mini-batch size per device (not sure what does it mean)

# -> (total mini-batch size is nodes * bs *sw_bs)
STEPS=100000 # how many steps to take

EVALEVERY=100 # test validation every given number of steps

OPTOPTS="--lrdecay --lr=1.e-5 --decay=1.e-3 --momentum=0.9 --opt=adamw --lr_schedule=warmup_cosine" # new optimizer options
# OPTOPTS="--lrdecay --lr=1.e-4 --decay=1.e-1 --momentum=0.9 --opt=adamw --lr_schedule=warmup_cosine" # new optimizer options

AMIN=0
AMAX=1
IMAGE_TRANSFORM_OPTIONS="--a_min $AMIN --a_max $AMAX"

# input/output channels
IN_CHANNELS=2
OUT_CHANNELS=2
HEAD_OPTIONS="--in_channels=$IN_CHANNELS --out_channels=$OUT_CHANNELS"

DATADIR="/ibex/ai/home/zampins/fm4g/datasets/multichannel_output_100" # where to find datasets and jsons
JSONLIST="dataset_0.json" # comma separated list of json filenames
DATASETLIST="dataset" # comma separated list of folder name where datasets are stored

# if you want to resume from previously trained weights
RESUME="--resume=/ibex/ai/home/zampins/fm4g/logs/35770044_100_multichannel_step4_d1me3/model_bestValRMSE.pt"

# if you just want to dump images
#DUMPIMG="--check_images"

# Some more informative name for the output dir
JOB_SUFFIX=_100_multichannel_step5_d1me3_lr1em5

LOGDIR=/ibex/ai/home/zampins/fm4g/logs/${SLURM_JOB_ID}${JOB_SUFFIX} # where to dump the intermediate checkpoints

# NOAMP=--noamp

# Python script with arguments
SCRIPT="$MAIN --num_workers 2 --num_steps=$STEPS $OPTOPTS --eval_num=$EVALEVERY --batch_size=$BS --sw_batch_size=$SWBS --datadir=$DATADIR --logdir=$LOGDIR --jsonlist=$JSONLIST --datasetlist=$DATASETLIST $IMAGE_TRANSFORM_OPTIONS $HEAD_OPTIONS $RESUME $DUMPIMG $NOAMP"

# SRUN command (1 torchrun instance per node, torchrun does the multi-processing on node)
#SRUN_CMD="srun -n $SLURM_NNODES -c $SLURM_CPUS_ON_NODE --ntasks-per-node 1 torchrun --max_restarts=0 --nnodes=$SLURM_NNODES --nproc-per-node=$SLURM_GPUS_PER_NODE --rdzv-id=$SLURM_JOB_ID --rdzv-backend=c10d --rdzv-endpoint=$HOST_NODE_ADDR $SCRIPT"
SRUN_CMD="srun -n $SLURM_NNODES -c $SLURM_CPUS_ON_NODE --ntasks-per-node 1 python $SCRIPT"

mkdir -p $LOGDIR
#$SRUN_CMD 2>&1  #>> $LOGDIR/joboutput.log
echo $SRUN_CMD > $LOGDIR/joboutput.log
$SRUN_CMD 2>&1 >> $LOGDIR/joboutput.log
