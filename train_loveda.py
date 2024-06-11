import argparse
import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
import pickle
from torch.autograd import Variable
import torch.optim as optim
import scipy.misc
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import sys
import os
import os.path as osp
import random
import time
import yaml
from tensorboardX import SummaryWriter
from utils.config_loveda import cfg
from LOVEDA.trainer_loveda import AD_Trainer
from utils.loss import CrossEntropy2d
from utils.tool import adjust_learning_rate
from albumentations import Compose, Normalize

from dataset.loveda_dataset import lovedaDataSet


MODEL = 'DeepLab'
TRANSFORMS=dict(transforms= Compose([
        Normalize(mean=(123.675, 116.28, 103.53),
                  std=(58.395, 57.12, 57.375),
                  max_pixel_value=1, always_apply=True),
        # er.preprocess.albu.ToTensor()
    ]))
BATCH_SIZE = 2 #CHANGED from 6
ITER_SIZE = 1156 // BATCH_SIZE
NUM_WORKERS = 1 #Changed from 2
DATA_DIRECTORY = '/home/sanskar/Unsupervised_Domian_Adaptation/LoveDA/Train/Urban/images_png/'
DATA_DIRECTORY_TEST= '/home/sanskar/Unsupervised_Domian_Adaptation/LoveDA/Val/Urban/images_png/'
DATA_LABEL_DIRECTORY = "/home/sanskar/GTA5/data/Urban/Pseudo_labels/train"
DROPRATE = 0.2
IGNORE_LABEL = 255
INPUT_SIZE = '1024,1024'
DATA_DIRECTORY_TARGET = '/home/sanskar/Unsupervised_Domian_Adaptation/LoveDA/Train/Urban/images_png/'
DATA_LABEL_DIRECTORY_TARGET= '/home/sanskar/Unsupervised_Domian_Adaptation/LoveDA/Train/Urban/masks_png/'
DATA_LABEL_DIRECTORY_TEST= '/home/sanskar/Unsupervised_Domian_Adaptation/LoveDA/Val/Urban/masks_png/'
INPUT_SIZE_TARGET = '1024,1024'
CROP_SIZE = '1024,1024'
LEARNING_RATE = 1e-4
MOMENTUM = 0.9
MAX_VALUE = 7
NUM_CLASSES = 7
NUM_STEPS = 100
NUM_STEPS_STOP = 40  # early stopping
POWER = 0.9
RANDOM_SEED = 1234
RESTORE_FROM = '/home/sanskar/GTA5/pretrained/DeepLab_resnet_pretrained_imagenet.pth'
SAVE_NUM_IMAGES = 2
EVALUATE_EVERY = 2
SAVE_EVALUATE = 0
SAVE_TB_EVERY = 2 #100
THRESHOLD = 1.0
WEIGHT_DECAY = 0.0005
WARM_UP =30
LAMBDA_LOSS_IA = 0.2
LAMBDA_LOSS_PE = 0.5
LAMBDA_LOSS_PS = 0.04
LAMBDA_LOSS_IM = 2.

TARGET = 'urban'
SET_TARGET = 'train'
TRAIN_METHOD = 'test'



def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="available options : DeepLab")
    parser.add_argument("--target", type=str, default=TARGET,
                        help="available options : cityscapes")
    parser.add_argument("--transform", type=dict, default=TRANSFORMS,
                        help="dictionary containing data transformations.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--iter-size", type=int, default=ITER_SIZE,
                        help="Accumulate gradients for ITER_SIZE iterations.")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--data-dir-test", type=str, default=DATA_DIRECTORY_TEST,
                        help="Path to the directory containing the test target dataset.")
    parser.add_argument("--data-label-dir", type=str, default=DATA_LABEL_DIRECTORY,
                        help="Path to the directory containing the source dataset label.")
    parser.add_argument("--data-label-dir-target", type=str, default=DATA_LABEL_DIRECTORY_TARGET,
                        help="Path to the directory containing the target dataset label.")
    parser.add_argument("--data-label-dir-test", type=str, default=DATA_LABEL_DIRECTORY_TEST,
                        help="Path to the directory containing the test label data.")
    parser.add_argument("--droprate", type=float, default=DROPRATE,
                        help="DropRate.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--crop-size", type=str, default=CROP_SIZE,
                        help="Comma-separated string with height and width of source images.")
    parser.add_argument("--data-dir-target", type=str, default=DATA_DIRECTORY_TARGET,
                        help="Path to the directory containing the target dataset.")
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")  # 没用
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--max-value", type=float, default=MAX_VALUE,
                        help="Max Value of Class Weight.")
    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--num-steps-stop", type=int, default=NUM_STEPS_STOP,
                        help="Number of training steps for early stopping.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-mirror_target", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--fp16", action="store_true",
                        help="Use FP16.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--evaluate_every", type=int, default=EVALUATE_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--save_evaluate", type=int, default=SAVE_EVALUATE,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--save-tb-every", type=int, default=SAVE_TB_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=None,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--train-method", type=str, default=TRAIN_METHOD, help = 'warm up iteration')
    parser.add_argument("--warm-up", type=float, default=WARM_UP, help = 'warm up iteration')
    parser.add_argument("--threshold", type=float, default=THRESHOLD, help = 'warm up iteration')
    parser.add_argument("--cpu", action='store_true', help="choose to use cpu device.")
    parser.add_argument("--class-balance", action='store_true', default=True, help="class balance.")
    parser.add_argument("--use-se", action='store_true', default=True, help="use se block.")
    parser.add_argument("--train_bn", action='store_true', default=True, help="train batch normalization.")
    parser.add_argument("--sync_bn", action='store_true', help="sync batch normalization.")
    parser.add_argument("--often-balance", action='store_true', default=True, help="balance the apperance times.")
    parser.add_argument("--gpu-ids", type=str, default='0', help = 'choose gpus')
    parser.add_argument("--tensorboard", action='store_false', help="choose whether to use tensorboard.")
    parser.add_argument("--log-dir", type=str, default=None,
                        help="Path to the directory of log.")
    parser.add_argument("--set_target", type=str, default=SET_TARGET,
                        help="choose adaptation set.")
    parser.add_argument('-lpl',"--lambda_loss_pseudo_label", type=float, default=LAMBDA_LOSS_IA)
    parser.add_argument('-lcl',"--lambda_loss_clustering_label", type=float, default=LAMBDA_LOSS_PE)
    parser.add_argument('-lcsl',"--lambda_loss_clustering_sym_label", type=float, default=LAMBDA_LOSS_PS)
    parser.add_argument('-lps',"--lambda_loss_prediction_self", type=float, default=LAMBDA_LOSS_IM)
    return parser.parse_args()


args = get_arguments()
# for arg in vars(args):
#     print(format(arg, '<20'), format(str(getattr(args, arg)), '<'))


def main():
    # INIT
    _init_fn = None
    transform= args.transform["transforms"]
    if args.random_seed:
        torch.manual_seed(args.random_seed)
        torch.cuda.manual_seed(args.random_seed)
        np.random.seed(args.random_seed)
        random.seed(args.random_seed)

        def _init_fn(worker_id):
            np.random.seed(args.random_seed + worker_id)

    w, h = map(int, args.crop_size.split(','))
    args.crop_size = (h, w)

    cudnn.enabled = True
    cudnn.benchmark = True

    if args.snapshot_dir:
        if not os.path.exists(args.snapshot_dir):
            os.makedirs(args.snapshot_dir)
    else:
        EXP_NAME = time.strftime('%Y%m%d%H%M')+ f'_{args.train_method}'
        args.snapshot_dir = os.path.join('./log', EXP_NAME)
        if not os.path.exists(args.snapshot_dir):
            os.makedirs(args.snapshot_dir)

    with open('%s/opts.yaml' % args.snapshot_dir, 'w') as fp:
        yaml.dump(vars(args), fp, default_flow_style=False)

    if args.tensorboard:
        args.log_dir = os.path.join(args.snapshot_dir, 'tensorboard')
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)
        writer = SummaryWriter(args.log_dir)

    str_ids = args.gpu_ids.split(',')
    gpu_ids = []
    for str_id in str_ids:
        gid = int(str_id)
        if gid >= 0:
            gpu_ids.append(gid)

    num_gpu = len(gpu_ids)
    args.multi_gpu = False
    if num_gpu > 1:
        args.multi_gpu = True
        Trainer = AD_Trainer(args)
        Trainer.G = torch.nn.DataParallel(Trainer.G, gpu_ids)
    else:
        Trainer = AD_Trainer(args)

    trainloader = data.DataLoader(lovedaDataSet(args.data_dir, args.data_label_dir,mirror=True,set='train',transforms=transform),
                                  batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                  pin_memory=True, drop_last=True, worker_init_fn=_init_fn)
    

    targetloader = data.DataLoader(lovedaDataSet(args.data_dir_target,args.data_label_dir_target, mirror=args.random_mirror_target,set=args.set_target, transforms=transform),
                                   batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                   pin_memory=True, drop_last=True)
    
    test_loader = data.DataLoader(lovedaDataSet(args.data_dir_test,args.data_label_dir_test, transforms=transform),
                                  batch_size=1,num_workers=1,shuffle=False,pin_memory=True)

    for i_iter in range(args.num_steps):

        trainloader_iter = enumerate(trainloader)
        targetloader_iter = enumerate(targetloader)

        adjust_learning_rate(Trainer.gen_opt,  i_iter, args)
        #print(Trainer.gen_opt.param_groups[0]['lr'])
        # print('\r>>>Current Iter step: %08d, Learning rate: %f'% (i_iter, Trainer.gen_opt.param_groups[0]['lr']), end='')

        for sub_i in range(args.iter_size):

            print('Current Epoch: %08d, Batch: %08d/%08d Learning rate: %f'% (i_iter+1, sub_i+1, args.iter_size, Trainer.gen_opt.param_groups[0]['lr']), end='')
            # print('\r>>>Current Epoch: %08d, Batch: %08d/%08d Learning rate: %f'% (i_iter+1, sub_i+1, args.iter_size, Trainer.gen_opt.param_groups[0]['lr']), end='')

            _, batch = trainloader_iter.__next__()
            _, batch_t = targetloader_iter.__next__()

            images, labels, name = batch
            images = images.cuda()
            labels = labels.long().cuda()
          
            images_t, labels_t, name_t = batch_t
            images_t = images_t.cuda()
            labels_t = labels_t.long().cuda()

            predictions_dicts, loss_dicts, loss_total, val_loss = Trainer.gen_update(images, images_t, labels, labels_t, i_iter)
            print('total loss: ',float(loss_total), 'val loss: ',float(val_loss))

        del predictions_dicts

        if args.tensorboard:
            scalar_info = {
                'total_loss': loss_total,
                'val_loss': val_loss,
            }
            scalar_info.update(loss_dicts)

            if i_iter % args.save_tb_every == 0:
                print('log added')
                for key, val in scalar_info.items():
                    writer.add_scalar(key, val, i_iter)

        del loss_total, val_loss
        del loss_dicts

        if i_iter >= args.num_steps_stop - 1:
            # mIoU = Trainer.evaluate_target(test_loader,args, per_class=True)
            print('save model ...')
            torch.save(Trainer.G.state_dict(), osp.join(args.snapshot_dir, 'rural_to_urban_final_source_free.pth'))
            break



    if args.tensorboard:
        writer.close()


if __name__ == '__main__':
    main()
