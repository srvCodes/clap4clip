#!/bin/bash
#PBS -q gpuvolta
#PBS -l storage=gdata/kf26
#PBS -l walltime=30:00:00
#PBS -l ngpus=2
#PBS -l ncpus=24
#PBS -P kf26
#PBS -l mem=40GB
#PBS -l jobfs=40GB
#PBS -l wd
#module load intel-mkl/2020.3.304
# module load python3/3.9.2
# module load cudnn/8.2.2-cuda11.4
# module load cuda/11.4.1
for ARCH in ViT-B-16 #ViT-L-14
do
for RUN in 0 #1 2 3 4 5 6 7 8 9
do
for EPOCH in 5
do
for DATASET in cifar100 imagenet-r
do
python3 main_incremental_submit.py --db_name $DATASET --finetuning --finetune-epochs 2 --num-run $RUN --compute-ece --compute-bwt --train_batch 32 --exemplar-selector random --root ../mammoth_datasets/ --multi-gpu --gpus 0,1 --default-gpu 0 --model clip_adapter --epochs $EPOCH --arch $ARCH  --method er
done
done
done 
done
