export CUDA_VISIBLE_DEVICES=3
# nnUNet_train 3d_fullres nnUNetTrainerV2 Task050_pelvis 1  --line
# nnUNet_predict -i  /root/workspace/work/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task050_pelvis/imagesTs/fold1 -o /root/workspace/work/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task050_pelvis/infer_decoder76/fold1 -t 050 -m 3d_fullres -f 1

nnUNet_train 3d_fullres nnUNetTrainerV2 Task050_pelvis 2
nnUNet_predict -i  /root/workspace/work/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task050_pelvis/imagesTs/fold2 -o /root/workspace/work/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task050_pelvis/infer_decoder78/fold2 -t 050 -m 3d_fullres -f 2



# nnUNet_train 2d nnUNetTrainerV2 Task120_DB 1 --line
# nnUNet_predict -i  /root/workspace/work/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task120_DB/imagesTs/fold1 -o /root/workspace/work/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task120_DB/infer_ce/fold1 -t 120 -m 2d -f 1

# nnUNet_train 2d nnUNetTrainerV2 Task120_DB 2 --line
# nnUNet_predict -i  /root/workspace/work/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task120_DB/imagesTs/fold2 -o /root/workspace/work/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task120_DB/infer_ce/fold2 -t 120 -m 2d -f 2

# nnUNet_train 2d nnUNetTrainerV2 Task120_DB 3 --line
# nnUNet_predict -i  /root/workspace/work/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task120_DB/imagesTs/fold3 -o /root/workspace/work/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task120_DB/infer_ce/fold3 -t 120 -m 2d -f 3

# nnUNet_predict -i  /root/workspace/work/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task050_pelvis/imagesTs/fold3 -o /root/workspace/work/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task050_pelvis/infer_MSE -t 050 -m 3d_fullres -f 3




