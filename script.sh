#!/bin/bash
cd SuperFusion
CUDA_VISIBLE_DEVICES=0                     python SuperFusion.py                     --Method SuperFusion                     --model_path /mnt/h/backup/Thermalfusion/TRICK/Benchmark/Checkpoint/SuperFusion/MSRS.pth                     --ir_dir /mnt/h/backup/Thermalfusion/TRICK/Benchmark/Dataset/ir                    --vi_dir /mnt/h/backup/Thermalfusion/TRICK/Benchmark/Dataset/vi                     --save_dir /mnt/h/backup/Thermalfusion/TRICK/Benchmark/Results/SuperFusion                     --is_RGB True
cd ..
