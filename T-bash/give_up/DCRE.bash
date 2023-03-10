export CUDA_VISIBLE_DEVICES=2
nnUNet_train 3d_fullres DCRETrainer Task050_pelvis 2
nnUNet_predict -i  /root/workspace/work/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task050_pelvis/imagesTs/fold2 -o /root/workspace/data/INFER_DATA/Task050_pelvis/DCRE_v1/fold2 -t 050 -m 3d_fullres -f 2 -tr DCRETrainer