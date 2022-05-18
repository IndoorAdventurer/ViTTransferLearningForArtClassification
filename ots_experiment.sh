#!/bin/bash

#SBATCH --job-name=ots_tl_with_vits
#SBATCH --mail-type=ALL
#SBATCH --mail-user=v.tonkes@student.rug.nl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --array=1-8

module purge
module load Python/3.9.6-GCCcore-11.2.0

mkdir -p /local/tmp/dataset
tar xzf /data/$USER/jpg_fullsize.tar.gz -C /local/tmp/dataset

infile=ots_models.txt
model=$(sed "${SLURM_ARRAY_TASK_ID}q;d" $infile)

source /data/$USER/bachproj/bin/activate
python ots_experiment.py ots_mat1_$model $model /local/tmp/dataset/fullsize /local/tmp/dataset