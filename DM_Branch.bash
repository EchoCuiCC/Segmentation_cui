export CUDA_VISIBLE_DEVICES=1
nnUNet_train 3d_fullres Branch_ss_l3_DM Task050_pelvis 0
nnUNet_predict -i  /root/workspace/work/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task050_pelvis/imagesTs/fold0 -o /root/workspace/data/INFER_DATA/Task050_pelvis/Branch_ss_l3_DM/fold0 -t 050 -m 3d_fullres -f 0 -tr Branch_ss_l3_DM
