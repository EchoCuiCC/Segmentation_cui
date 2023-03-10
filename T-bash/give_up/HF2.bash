export CUDA_VISIBLE_DEVICES=1
nnUNet_train 3d_fullres HFTrainerV2 Task050_pelvis 2
nnUNet_predict -i  /root/workspace/work/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task050_pelvis/imagesTs/fold2 -o /root/workspace/data/INFER_DATA/Task050_pelvis/hf_v2/fold2 -t 050 -m 3d_fullres -f 2 -tr HFTrainerV2