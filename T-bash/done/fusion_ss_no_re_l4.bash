export CUDA_VISIBLE_DEVICES=2
nnUNet_train 3d_fullres Fusion_ss_no_re_l4 Task050_pelvis 2
nnUNet_predict -i  /root/workspace/work/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task050_pelvis/imagesTs/fold2 -o /root/workspace/data/INFER_DATA/Task050_pelvis/Fusion_ss_no_re_l4/fold2 -t 050 -m 3d_fullres -f 2 -tr Fusion_ss_no_re_l4
