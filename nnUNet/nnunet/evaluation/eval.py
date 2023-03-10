from nnunet.evaluation.metrics import ConfusionMatrix, ALL_METRICS
import nibabel
import numpy as np
import os
from batchgenerators.utilities.file_and_folder_operations import save_json, subfiles, join
from tqdm import tqdm
import surface_distance as surfdist
import prettytable as pt
from skimage.morphology import skeletonize, skeletonize_3d

METRICS =[
    ['Accuracy','Precision','Recall'], # 体素精度，准确度(TP + TN) / (TP + FP + FN + TN)，精确度 TP / (TP + FP)，召回率(TP / (TP + FN))
    ['Dice'], # 相似度，骰子系数和中心线系数
    # ['']
] 

def cl_score(v, s):
    """[this function computes the skeleton volume overlap]
    Args:
        v ([bool]): [image]
        s ([bool]): [skeleton]
    Returns:
        [float]: [computed skeleton volume intersection]
    """
    return np.sum(v*s)/np.sum(s)


def clDice(v_p, v_l):
    """[this function computes the cldice metric]
    Args:
        v_p ([bool]): [predicted image]
        v_l ([bool]): [ground truth image]
    Returns:
        [float]: [cldice metric]
    """
    if len(v_p.shape)==2:
        tprec = cl_score(v_p,skeletonize(v_l))
        tsens = cl_score(v_l,skeletonize(v_p))
    elif len(v_p.shape)==3:
        tprec = cl_score(v_p,skeletonize_3d(v_l))
        tsens = cl_score(v_l,skeletonize_3d(v_p))
    return 2*tprec*tsens/(tprec+tsens)

def get_files_metrics(matrix,result=None):
    surface_distances = 0
    for item in result:
        if result[item]==None:
            result[item]=[]
                
        if item =='clDice':
            result[item].append(clDice(matrix['data'].test,matrix['data'].reference)*100)
            continue

        if item =='Hausdorff Distance 95':
            if surface_distances==0:
                surface_distances = surfdist.compute_surface_distances(matrix['data'].test==1, matrix['data'].reference==1, spacing_mm=matrix['voxel_spacing'])
            hd95 = surfdist.compute_robust_hausdorff(surface_distances, 95)
            result[item].append(hd95)
            continue

        if item =='Avg. Symmetric Surface Distance':
            if surface_distances==0:
                surface_distances = surfdist.compute_surface_distances(matrix['data'].test==1, matrix['data'].reference==1, spacing_mm=matrix['voxel_spacing'])                    
            asd = surfdist.compute_average_surface_distance(surface_distances)
            assd = (asd[0]+asd[1])/2
            result[item].append(assd)
            continue
        
        result[item].append(ALL_METRICS[item](confusion_matrix=matrix['data'])*100)
    return result

def get_fold_result(seg_fold,label_fold,cut):
    patients = os.listdir(seg_fold)
    result = {}.fromkeys([item for sublist in METRICS for item in sublist])
    matrix = []
    for i in tqdm(patients):
        try:
            seg = nibabel.load(join(seg_fold,i)).get_fdata().astype(np.uint8)
            label = nibabel.load(join(label_fold,i))
            voxel_spacing = label.header['pixdim'][1:4]
            label = label.get_fdata().astype(np.uint8)
            if cut:
                zz,_,_ = np.where(label)
                seg = seg[np.min(zz): np.max(zz)+1]
                label = label[np.min(zz): np.max(zz)+1]
            matrix={'data':ConfusionMatrix(test=seg==1,reference=label==1),'voxel_spacing':voxel_spacing}
            result = get_files_metrics(matrix,result)
        except:
            # print('missing '+i)
            continue
    return result

def evaluate(seg_dir,label_dir,fold_list=[0],cut=True):
    tb = pt.PrettyTable()
    tb.field_names = ['Fold Name']+[item for sublist in METRICS for item in sublist]
    results = {i:{} for i in fold_list}
    average_result_fold = []
    for fold in fold_list:
        result = get_fold_result(join(seg_dir,'fold_'+str(fold),'validation_raw'),label_dir,cut)
        result_list = []
        item = []
        for m in result:
            nums = result[m]
            result[m] = {'data':nums}
            nums = np.array(nums)
            result[m]['mean']=nums.mean()
            result[m]['std']=nums.std()
            result_list.append('{:.5f}±{:.5f}'.format(nums.mean(),nums.std()))
            item.append(nums.mean())
        tb.add_row(['fold'+str(fold)]+result_list)
        average_result_fold.append(item)
        results[fold]=result
        del result

    average_result_fold = np.stack(average_result_fold)
    avg_mean = average_result_fold.mean(axis=0)
    avg_std = average_result_fold.std(axis=0)
    average_result_fold = [ '{:.5f}±{:.5f}'.format(avg_mean[i],avg_std[i]) for i in range(len(tb.field_names)-1)]
    tb.add_row(['fold avg']+average_result_fold) 

    return results,tb

import pandas as pd

def tb_write_csv(file_name,prettyTable):
    data = prettyTable.get_csv_string().replace("'",'').split('\r\n')
    data = [ i.split(',') for i in data]
    df = pd.DataFrame(data = data[1:-1],columns=data[0])
    df.to_csv(file_name+'.csv')
    print('save successfully!')

def main():
    # 分割模型的地址
    seg_dir = '/root/workspace/work/nnUNetFrame/DATASET/nnUNet_trained_models/nnUNet/3d_fullres/Task050_pelvis/Branch_ss_l1__nnUNetPlansv2.1'
    # 标签的文件夹地址
    label_dir = '/root/workspace/work/nnUNetFrame/DATASET/nnUNet_raw/nnUNet_raw_data/Task050_pelvis/labelsTr/'
    # 结果存储的位置，加号后面是文件的名字
    csv_save_path = '/root/workspace/work/nnUNetFrame/'+'test'
    # 要算哪个fold的结果,如果要放五折，就写[0,1,2,3,4],[0,1,2],[2,3]
    fold_list = [0]
    result,tb = evaluate(seg_dir,label_dir,fold_list,False)
    # print(tb)
    tb_write_csv(csv_save_path,tb)

if __name__ == "__main__":
    main()
