#!/bin/bash
cd CSF
CUDA_VISIBLE_DEVICES=0                     python CSF.py                     --Method CSF                     --model_path_1 /mnt/h/backup/Thermalfusion/TRICK/Benchmark/Checkpoint/CSF/EC.ckpt                     --model_path_2 /mnt/h/backup/Thermalfusion/TRICK/Benchmark/Checkpoint/CSF/ED.ckpt                     --ir_dir /mnt/h/backup/Thermalfusion/TRICK/Benchmark/Dataset/ir                    --vi_dir /mnt/h/backup/Thermalfusion/TRICK/Benchmark/Dataset/vi                     --save_dir /mnt/h/backup/Thermalfusion/TRICK/Benchmark/Results/CSF                     --is_RGB True
cd ..
