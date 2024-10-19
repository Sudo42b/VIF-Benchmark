#!/bin/bash
cd CUFD
CUDA_VISIBLE_DEVICES=0                     python CUFD.py                     --Method CUFD                     --model_path_1 /mnt/h/backup/Thermalfusion/TRICK/Benchmark/Checkpoint/CUFD/1part1_model.ckpt                     --model_path_2 /mnt/h/backup/Thermalfusion/TRICK/Benchmark/Checkpoint/CUFD/part2_model.ckpt                     --ir_dir /mnt/h/backup/Thermalfusion/TRICK/Benchmark/Dataset/ir                    --vi_dir /mnt/h/backup/Thermalfusion/TRICK/Benchmark/Dataset/vi                     --save_dir /mnt/h/backup/Thermalfusion/TRICK/Benchmark/Results/CUFD                     --is_RGB True
cd ..
