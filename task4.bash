export CUDA_VISIBLE_DEVICES=0
nnUNet_predict -i  /root/workspace/work/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task050_pelvis/imagesTs/fold1 -o /root/workspace/data/INFER_DATA/Task050_pelvis/FuV3_ss_l3_1_02/fold1 -t 050 -m 3d_fullres -f 1 -tr FuV3_ss_l3_1_02
