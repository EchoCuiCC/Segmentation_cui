export CUDA_VISIBLE_DEVICES=2
nnUNet_train 3d_fullres sk_trainer Task050_pelvis 1 -c
nnUNet_predict -i  /root/workspace/work/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task050_pelvis/imagesTs/fold1 -o /root/workspace/data/INFER_DATA/Task050_pelvis/sk_trainer/fold1 -t 050 -m 3d_fullres -f 1 -tr sk_trainer

