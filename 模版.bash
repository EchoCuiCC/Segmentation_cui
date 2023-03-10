export CUDA_VISIBLE_DEVICES=X
nnUNet_train 3d_fullres trainer Task050_pelvis 0
nnUNet_predict -i  /root/workspace/work/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task050_pelvis/imagesTs/fold0 -o /root/workspace/data/INFER_DATA/Task050_pelvis/trainer/fold0 -t 050 -m 3d_fullres -f 0 -tr trainer

nnUNet_train 3d_fullres trainer Task050_pelvis 1
nnUNet_predict -i  /root/workspace/work/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task050_pelvis/imagesTs/fold1 -o /root/workspace/data/INFER_DATA/Task050_pelvis/trainer/fold1 -t 050 -m 3d_fullres -f 1 -tr trainer

nnUNet_train 3d_fullres trainer Task050_pelvis 2
nnUNet_predict -i  /root/workspace/work/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task050_pelvis/imagesTs/fold2 -o /root/workspace/data/INFER_DATA/Task050_pelvis/trainer/fold2 -t 050 -m 3d_fullres -f 2 -tr trainer

nnUNet_train 3d_fullres trainer Task050_pelvis 3
nnUNet_predict -i  /root/workspace/work/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task050_pelvis/imagesTs/fold3 -o /root/workspace/data/INFER_DATA/Task050_pelvis/trainer/fold3 -t 050 -m 3d_fullres -f 3 -tr trainer
                                                                                                                                                                                                                           