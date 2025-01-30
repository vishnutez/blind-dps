#!/bin/bash
##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=GenAI     #Set the job name to "JobExample1"
#SBATCH --time=01:30:00            #Set the wall clock limit to 1hr and 30min
#SBATCH --ntasks=1                 #Request 1 task
#SBATCH --ntasks-per-node=1        #Request 1 task/core per node
#SBATCH --mem=32G               #Request 64GB per node
## SBATCH --gres=gpu:a100:1     #Request 1 GPU
#SBATCH --output=Semiblind.%j  #Output file name stdout to [JobID]


cd $SCRATCH/semiblind-dps/blind-dps
ml Miniconda3
source activate semiblind-dps


python3 blind_deblur_demo.py \
    --img_model_config=configs/model_config.yaml \
    --kernel_model_config=configs/kernel_model_config.yaml \
    --diffusion_config=configs/diffusion_config.yaml \
    --task_config=configs/motion_deblur_config.yaml \
    --n_gen_samples=20 \
    --reg_ord=1 \
    --reg_scale=1.0;