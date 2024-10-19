import os
from tqdm import tqdm 
import sys
import argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

argparser = argparse.ArgumentParser()
argparser.add_argument('--path', type=str, default=os.path.join(os.getcwd(), 'Dataset'))
argparser.add_argument('--save', type=str, default=os.path.join(os.getcwd(), 'Results'))

args = argparser.parse_args()
save_dir = os.path.join(args.save)
os.makedirs(save_dir, exist_ok=True)
ir_dir = os.path.join(args.path, 'ir')
vi_dir = os.path.join(args.path, 'vi')
if not os.path.exists(ir_dir):
    print('Please download the dataset first!')
    print(ir_dir + ' does not exist!')
    sys.exit(0)
if not os.path.exists(vi_dir):
    print('Please download the dataset first!')
    print(vi_dir + ' does not exist!')
    sys.exit(0)

model_path_dict = dict()
model_path_dict_1 = dict()
model_path_dict_2 = dict()

model_path_dict_1['CSF'] = os.path.join(os.getcwd(), 'Checkpoint/CSF/EC.ckpt')
model_path_dict_2['CSF'] = os.path.join(os.getcwd(), 'Checkpoint/CSF/ED.ckpt')

model_path_dict_1['CUFD'] = os.path.join(os.getcwd(), 'Checkpoint/CUFD/1part1_model.ckpt')
model_path_dict_2['CUFD'] = os.path.join(os.getcwd(), 'Checkpoint/CUFD/part2_model.ckpt')

model_path_dict_1['DIDFuse'] = os.path.join(os.getcwd(), 'Checkpoint/DIDFuse/Encoder.pkl')
model_path_dict_2['DIDFuse'] = os.path.join(os.getcwd(), 'Checkpoint/DIDFuse/Decoder.pkl')

model_path_dict_1['DIVFusion'] = os.path.join(os.getcwd(), 'Checkpoint/DIVFusion/decom.ckpt')
model_path_dict_2['DIVFusion'] = os.path.join(os.getcwd(), 'Checkpoint/DIVFusion/enhance.ckpt')

model_path_dict_1['RFN-Nest'] = os.path.join(os.getcwd(), 'Checkpoint/RFN-Nest/RFN_Nest.model')
model_path_dict_2['RFN-Nest'] = os.path.join(os.getcwd(), 'Checkpoint/RFN-Nest/NestFuse.model')

model_path_dict['DenseFuse'] = os.path.join(os.getcwd(), 'Checkpoint/DenseFuse/DeseFuse.ckpt')

model_path_dict['FusionGAN'] = os.path.join(os.getcwd(), 'Checkpoint/FusionGAN/FusionGAN')

model_path_dict['GAN-FM'] = os.path.join(os.getcwd(), 'Checkpoint/GAN-FM/model.ckpt')

model_path_dict['GANMcC'] = os.path.join(os.getcwd(), 'Checkpoint/GANMcC/GANMcC')

model_path_dict['NestFuse'] = os.path.join(os.getcwd(), 'Checkpoint/NestFuse/nestfuse.model')

model_path_dict['PIAFusion'] = os.path.join(os.getcwd(), 'Checkpoint/PIAFusion')

model_path_dict['PMGI'] = os.path.join(os.getcwd(), 'Checkpoint/PMGI/PMGI')

model_path_dict['SDNet'] = os.path.join(os.getcwd(), 'Checkpoint/SDNet/SDNet.model')

model_path_dict['STDFusionNet'] = os.path.join(os.getcwd(), 'Checkpoint/STDFusionNet/Fusion.model-29')

model_path_dict['SeAFusion'] = os.path.join(os.getcwd(), 'Checkpoint/SeAFusion/SeAFusion.pth')

model_path_dict['SuperFusion'] = os.path.join(os.getcwd(), 'Checkpoint/SuperFusion/MSRS.pth')

model_path_dict['SwinFusion'] = os.path.join(os.getcwd(), 'Checkpoint/SwinFusion/SwinFusion.pth')

model_path_dict['TarDAL'] = os.path.join(os.getcwd(), 'Checkpoint/TarDAL/tardal++.pt')

model_path_dict['U2Fusion'] = os.path.join(os.getcwd(), 'Checkpoint/U2Fusion/model.ckpt')

model_path_dict['IFCNN'] = os.path.join(os.getcwd(), 'Checkpoint/IFCNN/IFCNN-MAX.pth')

model_path_dict['UMF-CMGR'] = os.path.join(os.getcwd(), 'Checkpoint/UMF-CMGR/UMF_CMGR.pth')

Method_list = ['CSF', 'CUFD', 'DIDFuse', 'DIVFusion', 'DenseFuse', 
               'FusionGAN', 'GAN-FM', 'GANMcC', 'IFCNN', 'NestFuse', 
               'PIAFusion', 'PMGI', 'RFN-Nest', 'SDNet', 'STDFusionNet', 
               'SeAFusion', 'SuperFusion', 'SwinFusion', 'TarDAL', 'U2Fusion', 
               'UMF-CMGR']


print(len(Method_list))
two_model_list =['CSF', 'CUFD', 'DIDFuse', 'DIVFusion', 'RFN-Nest'] 


for Method in tqdm(Method_list):
    save_dir_method = os.path.join(save_dir, Method)
    if Method not in two_model_list:
        with open('script.sh', 'w') as f:
            f.write('#!/bin/bash\n')
            f.write("cd {}\n".format(Method))
            print(Method.replace('-', ''))
            f.write(f"CUDA_VISIBLE_DEVICES=0 \
                    python {Method.replace('-', '')}.py \
                    --Method {Method} \
                    --model_path {model_path_dict[Method]} \
                    --ir_dir {ir_dir}\
                    --vi_dir {vi_dir} \
                    --save_dir {save_dir_method} \
                    --is_RGB {True}\n")
            f.write("cd ..\n".format(Method))
        os.system('bash script.sh')
        # os.system('bash script.sh')
    else:
        with open('script.sh', 'w') as f:
            f.write('#!/bin/bash\n')
            f.write("cd {}\n".format(Method))
            print(Method.replace('-', ''))
            f.write(f"CUDA_VISIBLE_DEVICES=0 \
                    python {Method.replace('-', '')}.py \
                    --Method {Method} \
                    --model_path_1 {model_path_dict_1[Method]} \
                    --model_path_2 {model_path_dict_2[Method]} \
                    --ir_dir {ir_dir}\
                    --vi_dir {vi_dir} \
                    --save_dir {save_dir_method} \
                    --is_RGB {True}\n")
            f.write("cd ..\n".format(Method))
        os.system('bash script.sh')