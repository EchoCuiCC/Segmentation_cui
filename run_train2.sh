export CUDA_VISIBLE_DEVICES=2
# 三通道 CFP图像

nnUNet_predict -i  /root/workspace/work/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/CRVO/CFP/image -o /root/workspace/work/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/CRVO/CFP/infer -t 6 -m 2d -f 0

nnUNet_predict -i  /root/workspace/work/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/CRVO/CFP_O/image -o /root/workspace/work/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/CRVO/CFP_O/infer -t 6 -m 2d -f 0

nnUNet_predict -i  /root/workspace/work/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/CRVO/FFA/image -o /root/workspace/work/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/CRVO/FFA/infer -t 6 -m 2d -f 0

nnUNet_predict -i  /root/workspace/work/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/CRVO/FFA_O/image -o /root/workspace/work/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/CRVO/FFA_O/infer -t 6 -m 2d -f 0

nnUNet_predict -i  /root/workspace/work/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/CRVO/SLO/image -o /root/workspace/work/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/CRVO/SLO/infer -t 6 -m 2d -f 0

nnUNet_predict -i  /root/workspace/work/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/CRVO/SLO_O/image -o /root/workspace/work/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/CRVO/SLO_O/infer -t 6 -m 2d -f 0
# 单通道灰度图 使用X广冠脉分割数据集
# nnUNet_predict -i  /root/workspace/work/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/BRVO/FFA/image -o /root/workspace/work/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/BRVO/FFA/infer -t 121 -m 2d -f 1

# # nnUNet_predict -i  /root/workspace/work/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task002_PortalVein/imagesTs/fold2 -o /root/workspace/work/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task002_PortalVein/infer/fold2 -t 2 -m 3d_fullres -f 2
# nnUNet_train 3d_fullres nnUNetTrainerV2 Task002_PortalVein --line 2

# nnUNet_train 3d_fullres nnUNetTrainerV2 Task002_PortalVein --line 3 
# nnUNet_predict -i  /root/workspace/work/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task002_PortalVein/imagesTs/fold3 -o /root/workspace/work/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task002_PortalVein/infer2/fold3 -t 2 -m 3d_fullres -f 3

# nnUNet_predict -i  /root/workspace/work/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task002_PortalVein/imagesTs/fold2 -o /root/workspace/work/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task002_PortalVein/infer2/fold2 -t 2 -m 3d_fullres -f 2