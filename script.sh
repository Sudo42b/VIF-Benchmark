#!/bin/bash
cd GANMcC
CUDA_VISIBLE_DEVICES=0                     python GANMcC.py                     --Method GANMcC                     --model_path /workspace/VIF/Checkpoint/GANMcC/GANMcC                     --ir_dir /workspace/VIF/datasets/test_imgs/ir                    --vi_dir /workspace/VIF/datasets/test_imgs/vi                     --save_dir /workspace/VIF/Results/GANMcC                     --is_RGB True
cd ..
