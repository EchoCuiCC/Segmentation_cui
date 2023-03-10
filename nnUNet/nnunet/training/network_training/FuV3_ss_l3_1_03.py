from collections import OrderedDict
from typing import Tuple

import numpy as np
import torch
from nnunet.training.data_augmentation.data_augmentation_moreDA import get_moreDA_augmentation
from nnunet.training.loss_functions.deep_supervision import MultipleOutputLoss2
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda
from nnunet.network_architecture.Fusion_UNetV3 import Fusion_UNetV3
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.network_architecture.neural_network import SegmentationNetwork
from nnunet.training.data_augmentation.default_data_augmentation import default_2D_augmentation_params, \
    get_patch_size, default_3D_augmentation_params
from nnunet.training.dataloading.dataset_loading import unpack_dataset
from nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer
from nnunet.utilities.nd_softmax import softmax_helper
from sklearn.model_selection import KFold
from torch import nn
from torch.cuda.amp import autocast
from nnunet.training.learning_rate.poly_lr import poly_lr
from batchgenerators.utilities.file_and_folder_operations import *


# editing
import skimage

class FuV3_ss_l3_1_03(nnUNetTrainer):
    """
    Info for Fabian: same as internal nnUNetTrainerV2_2
    """
# editing 增加了centerline
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False,centerline = False,daw=False,ccp=False,tcl=False):
        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)
        #额外添加的
        self.centerline= True
        self.daw = False
        self.change_patch_size = {
            'change_or_not':True,
            'size':[64,128,128],
            'batch_size':2,
        }
        self.ccp=False
        self.tcl=True
        self.branch_block_num = 3 if self.tcl else 0
        self.fusion_num = 1
        self.splitUp = True
        self.alpha = 0.3
        # epcho数
        self.max_num_epochs = 100
        self.initial_lr = 1e-2
        self.deep_supervision_scales = None
        self.ds_loss_weights = None

        self.pin_memory = True

        if daw:
            self.weight = []
            # 所有任务一个epoch内的所有损失记录
            self.train_losses_epoch_tasks = []
            # 所有任务每个epoch的平均损失
            self.train_losses_tasks = []
        if centerline:
            self.seg_dice = []
            self.cen_dice = []

    def initialize(self, training=True, force_load_plans=False):
        """
        - replaced get_default_augmentation with get_moreDA_augmentation
        - enforce to only run this code once
        - loss function wrapper for deep supervision

        :param training:
        :param force_load_plans:
        :return:
        """
        if not self.was_initialized:
            maybe_mkdir_p(self.output_folder)

            # 如果没有计划文件或者这里要求使用强制plan计划 
            if force_load_plans or (self.plans is None):
                self.load_plans_file()

            self.process_plans(self.plans)

            # AUG:Trainer配置增强参数载入
            self.setup_DA_params()

            ################# Here we wrap the loss for deep supervision ############
            # we need to know the number of outputs of the network
            # 这个net_num_pool_op_kernel_sizes参数是下采样的倍数展示，
            # 'pool_op_kernel_sizes': [[1, 2, 2],
            # [2, 2, 2],
            # [2, 2, 2],
            # [2, 2, 2],
            # [2, 2, 2]],
            # net_numpool则表示下采样层（list类型的参数）
            net_numpool = len(self.net_num_pool_op_kernel_sizes)

            # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
            # this gives higher resolution outputs more weight in the loss
            # 给每个输出一个权重,这些权重随着分辨率的降低呈指数下降（除以 2）
            # 这使更高分辨率的输出在损失中具有更大的权重
            weights = np.array([1 / (2 ** i) for i in range(net_numpool)])

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            # 它这里其实只有最后一层的不算
            mask = np.array([True] + [True if i < net_numpool - 1 else False for i in range(1, net_numpool)])
            # ndrry的写法里可以这么选择部分数据
            # a = np.stack([1,2,3,4,5])
            # a[[True,True,False,True,True]] = 1,2,4,5
            weights[~mask] = 0
            weights = weights / weights.sum()
            self.ds_loss_weights = weights


            # now wrap the loss
            # LOSS:深监督嵌入到loss
            # 这里没有搞懂是怎么搞的，它里面有一个对loss的操作
            # 这里会对原本的celoss+softdiceloss的值来进行一个权重计算，还没搞懂怎么回事儿 
            self.loss = MultipleOutputLoss2(self.loss, self.ds_loss_weights) 
            ################# END ###################

            self.folder_with_preprocessed_data = join(self.dataset_directory, self.plans['data_identifier'] +
                                                      "_stage%d" % self.stage)
            if training:
                # aug:trainnerV2的训练验证集的dataloader
                self.dl_tr, self.dl_val = self.get_basic_generators()
                if self.unpack_data:

                    # TAG:unpack_data参数指令将npz文件打开成npy文件存放
                    # 如果数据量过大，这里可以选择不unpack
                    # print("unpacking dataset")
                    unpack_dataset(self.folder_with_preprocessed_data)
                    # print("done")
                else:
                    print(
                        "INFO: Not unpacking data! Training may be slow due to that. Pray you are not using 2d or you "
                        "will wait all winter for your model to finish!")

                # aug:Generator增强器
                # 这里的patch_size是网络输入的patch_size，generator_batch_size则是在dataloader里基本块
                self.tr_gen, self.val_gen = get_moreDA_augmentation(
                    self.dl_tr, self.dl_val,
                    self.data_aug_params[
                        'patch_size_for_spatialtransform'],
                    self.data_aug_params,
                    deep_supervision_scales=self.deep_supervision_scales,
                    pin_memory=self.pin_memory,
                    use_nondetMultiThreadedAugmenter=False
                )
                self.print_to_log_file("TRAINING KEYS:\n %s" % (str(self.dataset_tr.keys())),
                                       also_print_to_console=False)
                self.print_to_log_file("VALIDATION KEYS:\n %s" % (str(self.dataset_val.keys())),
                                       also_print_to_console=False)
                self.print_to_log_file('Turning on: Centerline [%s] CCP [%s] TCL [%s] SplitUp[%s]' % (str(self.centerline),str(self.ccp),str(self.tcl),str(self.splitUp)))    
                self.print_to_log_file('The num pooling:{}, tcl begin level:{}, fusion_num:{}'.format(len(self.net_num_pool_op_kernel_sizes),self.branch_block_num,self.fusion_num))            
            else:
                pass

            #net:trainerV2,网络初始化
            self.initialize_network()
            #net:trainerV2,优化器学习率设置
            self.initialize_optimizer_and_scheduler()

            assert isinstance(self.network, (SegmentationNetwork, nn.DataParallel))
        else:
            self.print_to_log_file('self.was_initialized is True, not running self.initialize again')
        self.was_initialized = True

    def initialize_network(self):
        """
        - momentum 0.99
        - SGD instead of Adam
        - self.lr_scheduler = None because we do poly_lr
        - deep supervision = True
        - i am sure I forgot something here

        Known issue: forgot to set neg_slope=0 in InitWeights_He; should not make a difference though
        :return:
        """
        if self.threeD:
            conv_op = nn.Conv3d
            dropout_op = nn.Dropout3d
            # todo:InstanceNorm 为啥采用像素归一化?
            # 
            norm_op = nn.InstanceNorm3d

        else:
            conv_op = nn.Conv2d
            dropout_op = nn.Dropout2d
            norm_op = nn.InstanceNorm2d

        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op_kwargs = {'p': 0, 'inplace': True}
        net_nonlin = nn.LeakyReLU
        net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        # net:网络初始化
        # 这里采用卷积上采样，卷积池化
        # editing 增加了centerline部分
        #deep_supervision开关的地方
        self.network = Fusion_UNetV3(self.num_input_channels, self.base_num_features, self.num_classes,
                                    len(self.net_num_pool_op_kernel_sizes),
                                    self.conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                                    dropout_op_kwargs,
                                    net_nonlin, net_nonlin_kwargs, True, False, lambda x: x, InitWeights_He(1e-2),
                                    self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True,     
                                    True,centerline=self.centerline,branch_block_num=self.branch_block_num,
                                    fusion_num=self.fusion_num,splitUp=self.splitUp,alpha=self.alpha)
        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper

    def initialize_optimizer_and_scheduler(self):
        assert self.network is not None, "self.initialize_network must be called first"
        self.optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                                         momentum=0.99, nesterov=True)
        self.lr_scheduler = None

    def run_online_evaluation(self, output, target):
        """
        due to deep supervision the return value and the reference are now lists of tensors. We only need the full
        resolution output because this is what we are interested in in the end. The others are ignored
        :param output:
        :param target:
        :return:
        """
        target = target[0]
        output = output[0]
        return super().run_online_evaluation(output, target)

    def validate(self, do_mirroring: bool = True, use_sliding_window: bool = True,
                 step_size: float = 0.5, save_softmax: bool = True, use_gaussian: bool = True, overwrite: bool = True,
                 validation_folder_name: str = 'validation_raw', debug: bool = False, all_in_gpu: bool = False,
                 segmentation_export_kwargs: dict = None, run_postprocessing_on_folds: bool = True):
        """
        We need to wrap this because we need to enforce self.network.do_ds = False for prediction
        """
        ds = self.network.do_ds
        self.network.do_ds = False
        ret = super().validate(do_mirroring=do_mirroring, use_sliding_window=use_sliding_window, step_size=step_size,
                               save_softmax=save_softmax, use_gaussian=use_gaussian,
                               overwrite=overwrite, validation_folder_name=validation_folder_name, debug=debug,
                               all_in_gpu=all_in_gpu, segmentation_export_kwargs=segmentation_export_kwargs,
                               run_postprocessing_on_folds=run_postprocessing_on_folds)

        self.network.do_ds = ds
        return ret

    def predict_preprocessed_data_return_seg_and_softmax(self, data: np.ndarray, do_mirroring: bool = True,
                                                         mirror_axes: Tuple[int] = None,
                                                         use_sliding_window: bool = True, step_size: float = 0.5,
                                                         use_gaussian: bool = True, pad_border_mode: str = 'constant',
                                                         pad_kwargs: dict = None, all_in_gpu: bool = False,
                                                         verbose: bool = True, mixed_precision=True) :
                                                        #  Tuple[np.ndarray, np.ndarray]
        """
        We need to wrap this because we need to enforce self.network.do_ds = False for prediction
        """
        ds = self.network.do_ds
        self.network.do_ds = False
        ret = super().predict_preprocessed_data_return_seg_and_softmax(data,
                                                                       do_mirroring=do_mirroring,
                                                                       mirror_axes=mirror_axes,
                                                                       use_sliding_window=use_sliding_window,
                                                                       step_size=step_size, use_gaussian=use_gaussian,
                                                                       pad_border_mode=pad_border_mode,
                                                                       pad_kwargs=pad_kwargs, all_in_gpu=all_in_gpu,
                                                                       verbose=verbose,
                                                                       mixed_precision=mixed_precision)
        self.network.do_ds = ds
        return ret

    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False):
        """
        gradient clipping improves training stability

        :param data_generator:
        :param do_backprop:
        :param run_online_evaluation:
        :return:
        """
        data_dict = next(data_generator)
        data = data_dict['data']
        target = data_dict['target']

        # editing 增加了centerline
        if self.centerline:
            skeletons=[]
            # 因为可能有深监督，所以这里target的结构是list([layer1],[layer2],[layer3],[layer4],[layer5])
            # 这里layer的层次结构式 [b,c,x,y,z]
            for layer in target:
                # 这里好像在3D里面，会需要[ skimage.morphology.dilation(skimage.morphology.skeletonize(img[0].numpy()))[:][None]/255 for img in layer]
                # 但是原因我不懂。。。。
                # centerline
                deep_superversion = np.stack([ skimage.morphology.dilation(skimage.morphology.skeletonize(img[0].numpy()))[:][None]/255 for img in layer])
                deep_superversion = deep_superversion*layer.numpy()

                # #distance map
                # from scipy.ndimage import distance_transform_edt as distance
                # deep_superversion = np.stack([ np.around(distance(img[0].numpy()))[:][None] for img in layer])
                # deep_superversion[deep_superversion>8]=8

                skeletons.append(deep_superversion)
                skeletons = maybe_to_torch(skeletons)


        data = maybe_to_torch(data)
        target = maybe_to_torch(target)

        if torch.cuda.is_available():
            data = to_cuda(data)
            target = to_cuda(target)
            if self.centerline:
                skeletons = to_cuda(skeletons)

        self.optimizer.zero_grad()

        if self.fp16:
            with autocast():
                if self.centerline:
                    seg_output,centerline_output = self.network(data)
                    del data

                    seg_l,seg_ce,seg_dice = self.loss(seg_output,target)
                    centerline_l,centerline_ce,centerline_dice = self.loss(centerline_output,skeletons)
                    # 正常结果，不加惩罚项
                    l = 0.5*seg_l+0.5*centerline_l
                    # print('[seg,centerline,tcl]  -1 loss:{:.3f}, -2 loss:{:.3f}'.format(seg_dice.item(),centerline_dice.item()))

                        # # --MSE的方式 
                        # re = (seg_dice-centerline_dice)**2
                        # l = 0.5*seg_l+0.5*centerline_l+re
                        # # --MSE的方式
                                        
                else:    
                    output = self.network(data)
                    del data
                    l,ce,dice = self.loss(output, target)

            if do_backprop:
                # tag:混合精度运算
                # 因为开启了混合精度运算,故这里需要将scaler进行一些操作,下面三步时必须进行的
                # scaler.scale(loss).backward()
                # scaler.step(optimizer)
                # scaler.update()
                # 这个unscale操作时可选的,如果要选unscale就要做clip_grad_norm
                self.amp_grad_scaler.scale(l).backward()

                self.amp_grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)

                self.amp_grad_scaler.step(self.optimizer)
                self.amp_grad_scaler.update()
        else:
            output = self.network(data)
            del data
            l,_,_ = self.loss(output, target)

            if do_backprop:
                l.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.optimizer.step()

        if run_online_evaluation:
            if self.centerline:
                self.run_online_evaluation(seg_output,target)
            else:
                self.run_online_evaluation(output, target)

        del target

        if self.centerline:
            # --【DC】这里记录了中间的dice，不需要记录的话，这里只需要返回l
            return l.detach().cpu().numpy(),seg_dice.detach().cpu().numpy(),centerline_dice.detach().cpu().numpy()
            # --【DC】这里记录了中间的dice，不需要记录的话，这里只需要返回l
        
        return l.detach().cpu().numpy()

    def do_split(self):
        """
        The default split is a 5 fold CV on all available training cases. nnU-Net will create a split (it is seeded,
        so always the same) and save it as splits_final.pkl file in the preprocessed data directory.
        Sometimes you may want to create your own split for various reasons. For this you will need to create your own
        splits_final.pkl file. If this file is present, nnU-Net is going to use it and whatever splits are defined in
        it. You can create as many splits in this file as you want. Note that if you define only 4 splits (fold 0-3)
        and then set fold=4 when training (that would be the fifth split), nnU-Net will print a warning and proceed to
        use a random 80:20 data split.
        :return:
        """
        if self.fold == "all":
            # if fold==all then we use all images for training and validation
            tr_keys = val_keys = list(self.dataset.keys())
        else:
            splits_file = join(self.dataset_directory, "splits_final.pkl")

            # if the split file does not exist we need to create it
            if not isfile(splits_file):
                self.print_to_log_file("Creating new 4-fold cross-validation split...")
                splits = []
                all_keys_sorted = np.sort(list(self.dataset.keys()))
                kfold = KFold(n_splits=4, shuffle=True, random_state=12345)
                for i, (train_idx, test_idx) in enumerate(kfold.split(all_keys_sorted)):
                    train_keys = np.array(all_keys_sorted)[train_idx]
                    test_keys = np.array(all_keys_sorted)[test_idx]
                    splits.append(OrderedDict())
                    splits[-1]['train'] = train_keys
                    splits[-1]['val'] = test_keys
                save_pickle(splits, splits_file)

            else:
                self.print_to_log_file("Using splits from existing split file:", splits_file)
                splits = load_pickle(splits_file)
                self.print_to_log_file("The split file contains %d splits." % len(splits))

            self.print_to_log_file("Desired fold for training: %d" % self.fold)
            if self.fold < len(splits):
                tr_keys = splits[self.fold]['train']
                val_keys = splits[self.fold]['val']
                self.print_to_log_file("This split has %d training and %d validation cases."
                                       % (len(tr_keys), len(val_keys)))
            else:
                self.print_to_log_file("INFO: You requested fold %d for training but splits "
                                       "contain only %d folds. I am now creating a "
                                       "random (but seeded) 80:20 split!" % (self.fold, len(splits)))
                # if we request a fold that is not in the split file, create a random 80:20 split
                rnd = np.random.RandomState(seed=12345 + self.fold)
                keys = np.sort(list(self.dataset.keys()))
                idx_tr = rnd.choice(len(keys), int(len(keys) * 0.8), replace=False)
                idx_val = [i for i in range(len(keys)) if i not in idx_tr]
                tr_keys = [keys[i] for i in idx_tr]
                val_keys = [keys[i] for i in idx_val]
                self.print_to_log_file("This random 80:20 split has %d training and %d validation cases."
                                       % (len(tr_keys), len(val_keys)))

        tr_keys.sort()
        val_keys.sort()
        self.dataset_tr = OrderedDict()
        for i in tr_keys:
            self.dataset_tr[i] = self.dataset[i]

        self.dataset_val = OrderedDict()
        for i in val_keys:
            self.dataset_val[i] = self.dataset[i]

    def setup_DA_params(self):
        """
        - we increase roation angle from [-15, 15] to [-30, 30]
        - scale range is now (0.7, 1.4), was (0.85, 1.25)
        - we don't do elastic deformation anymore

        :return:
        """
        # list1+list2 =list1.append(list2)
        # [[1,2,3]]+[[4,5,6],[7,8,9]]=[[1,2,3],[4,5,6],[7,8,9]]
        # deep_supervision_scales就是每层网络的下采样倍数[[1,1,1],[0.5,0.5,0.5],[0.25,0.25,0.25]]
        self.deep_supervision_scales = [[1, 1, 1]] + list(list(i) for i in 1 / np.cumprod(
            np.vstack(self.net_num_pool_op_kernel_sizes), axis=0))[:-1]

        if self.threeD:
            # 如果是3D的数据，v2就在默认3d augmentation的基础上修改部分数据
            self.data_aug_params = default_3D_augmentation_params
            self.data_aug_params['rotation_x'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            self.data_aug_params['rotation_y'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            self.data_aug_params['rotation_z'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
            if self.do_dummy_2D_aug:
                self.data_aug_params["dummy_2D"] = True
                self.print_to_log_file("Using dummy2d data augmentation")
                self.data_aug_params["elastic_deform_alpha"] = \
                    default_2D_augmentation_params["elastic_deform_alpha"]
                self.data_aug_params["elastic_deform_sigma"] = \
                    default_2D_augmentation_params["elastic_deform_sigma"]
                self.data_aug_params["rotation_x"] = default_2D_augmentation_params["rotation_x"]
        else:
            # 如果不是3D的话，这里将会在默认2D增强的基础上修改部分数据
            self.do_dummy_2D_aug = False
            if max(self.patch_size) / min(self.patch_size) > 1.5:
                default_2D_augmentation_params['rotation_x'] = (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi)
            self.data_aug_params = default_2D_augmentation_params
        self.data_aug_params["mask_was_used_for_normalization"] = self.use_mask_for_norm

        if self.do_dummy_2D_aug:
            # 如果做‘呆板’2D增强的话，这里因为旋转会要算出更大的一个框
            self.basic_generator_patch_size = get_patch_size(self.patch_size[1:],
                                                             self.data_aug_params['rotation_x'],
                                                             self.data_aug_params['rotation_y'],
                                                             self.data_aug_params['rotation_z'],
                                                             self.data_aug_params['scale_range'])
            self.basic_generator_patch_size = np.array([self.patch_size[0]] + list(self.basic_generator_patch_size))
            patch_size_for_spatialtransform = self.patch_size[1:]
        else:
            # 
            # 修改patch_size
            if self.change_patch_size['change_or_not']:
                self.print_to_log_file('the planning self.patch_size is',self.patch_size,' change into ',self.change_patch_size['size'])
                self.patch_size = self.change_patch_size['size']

            self.basic_generator_patch_size = get_patch_size(self.patch_size, self.data_aug_params['rotation_x'],
                                                             self.data_aug_params['rotation_y'],
                                                             self.data_aug_params['rotation_z'],
                                                             self.data_aug_params['scale_range'])
            patch_size_for_spatialtransform = self.patch_size

        self.data_aug_params["scale_range"] = (0.7, 1.4)
        self.data_aug_params["do_elastic"] = False
        self.data_aug_params['selected_seg_channels'] = [0]
        self.data_aug_params['patch_size_for_spatialtransform'] = patch_size_for_spatialtransform

        self.data_aug_params["num_cached_per_thread"] = 2

    def maybe_update_lr(self, epoch=None):
        """
        if epoch is not None we overwrite epoch. Else we use epoch = self.epoch + 1

        (maybe_update_lr is called in on_epoch_end which is called before epoch is incremented.
        herefore we need to do +1 here)

        :param epoch:
        :return:
        """
        if epoch is None:
            ep = self.epoch + 1
        else:
            ep = epoch
        self.optimizer.param_groups[0]['lr'] = poly_lr(ep, self.max_num_epochs, self.initial_lr, 0.9)
        self.print_to_log_file("lr:", np.round(self.optimizer.param_groups[0]['lr'], decimals=6))

    def on_epoch_end(self):
        """
        overwrite patient-based early stopping. Always run to 1000 epochs
        :return:
        """
        super().on_epoch_end()
        continue_training = self.epoch < self.max_num_epochs

        # it can rarely happen that the momentum of nnUNetTrainerV2 is too high for some dataset. If at epoch 100 the
        # estimated validation Dice is still 0 then we reduce the momentum from 0.99 to 0.95
        if self.epoch == 100:
            if self.all_val_eval_metrics[-1] == 0:
                self.optimizer.param_groups[0]["momentum"] = 0.95
                self.network.apply(InitWeights_He(1e-2))
                self.print_to_log_file("At epoch 100, the mean foreground Dice was 0. This can be caused by a too "
                                       "high momentum. High momentum (0.99) is good for datasets where it works, but "
                                       "sometimes causes issues such as this one. Momentum has now been reduced to "
                                       "0.95 and network weights have been reinitialized")
        return continue_training

    def run_training(self):
        """
        if we run with -c then we need to set the correct lr for the first epoch, otherwise it will run the first
        continued epoch with self.initial_lr

        we also need to make sure deep supervision in the network is enabled for training, thus the wrapper
        :return:
        """
        self.maybe_update_lr(self.epoch)  # if we dont overwrite epoch then self.epoch+1 is used which is not what we
        # want at the start of the training
        ds = self.network.do_ds
        self.network.do_ds = True
        ret = super().run_training()
        self.network.do_ds = ds
        return ret
 