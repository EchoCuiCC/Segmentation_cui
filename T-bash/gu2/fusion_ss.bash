export CUDA_VISIBLE_DEVICES=0
nnUNet_train 3d_fullres FusionTrainer_ss Task050_pelvis 1 -c
nnUNet_predict -i  /root/workspace/work/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task050_pelvis/imagesTs/fold1 -o /root/workspace/data/INFER_DATA/Task050_pelvis/fusion_ss/fold1 -t 050 -m 3d_fullres -f 1 -tr FusionTrainer_ss

nnUNet_train 3d_fullres FusionTrainer_ss Task050_pelvis 3
nnUNet_predict -i  /root/workspace/work/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task050_pelvis/imagesTs/fold3 -o /root/workspace/data/INFER_DATA/Task050_pelvis/fusion_ss/fold3 -t 050 -m 3d_fullres -f 3 -tr FusionTrainer_ss