#!/usr/bin/env bash
#OAR -p gpu='YES'
#OAR -l /gpunum=1,walltime=48

module load conda/2020.11-python3.8
source activate audiostylenet

module load cuda/11.0
module load cudnn/8.0-cuda-11.0
module load gcc/7.3.0

wandb agent rntc/audiodriven/1snl615d
