export CUDA_VISIBLE_DEVICES=3
nnUNet_predict -i  /root/workspace/work/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task050_pelvis/imagesTs/fold1 -o /root/workspace/data/INFER_DATA/Task050_pelvis/FuV3_ss_l3_1_05/fold1 -t 050 -m 3d_fullres -f 1 -tr FuV3_ss_l3_1_05

nnUNet_train 3d_fullres FuV3_ss_l3_1_05 Task050_pelvis 3
nnUNet_predict -i  /root/workspace/work/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task050_pelvis/imagesTs/fold3 -o /root/workspace/data/INFER_DATA/Task050_pelvis/FuV3_ss_l3_1_05/fold3 -t 050 -m 3d_fullres -f 3 -tr FuV3_ss_l3_1_05

nnUNet_train 3d_fullres FuV3_ss_l3_1_07 Task050_pelvis 0
nnUNet_predict -i  /root/workspace/work/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task050_pelvis/imagesTs/fold0 -o /root/workspace/data/INFER_DATA/Task050_pelvis/FuV3_ss_l3_1_07/fold0 -t 050 -m 3d_fullres -f 0 -tr FuV3_ss_l3_1_07