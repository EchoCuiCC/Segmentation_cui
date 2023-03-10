export CUDA_VISIBLE_DEVICES=3
nnUNet_train 3d_fullres sk_trainer Task050_pelvis 3 -c
nnUNet_predict -i  /root/workspace/work/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task050_pelvis/iskUNetmagesTs/fold3 -o /root/workspace/data/INFER_DATA/Task050_pelvis/sk_trainer/fold3 -t 050 -m 3d_fullres -f 3 -tr sk_trainer
