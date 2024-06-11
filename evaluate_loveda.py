
import argparse
import scipy
from scipy import ndimage
import numpy as np
import sys
from packaging import version
from multiprocessing import Pool
import torch
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
from torch.utils import data, model_zoo
from model.deeplab_advent_no_p import get_deeplab_v2
from dataset.loveda_dataset import lovedaDataSet
from collections import OrderedDict
import os
from PIL import Image
from utils.tool import fliplr
import matplotlib.pyplot as plt
import torch.nn as nn
import yaml
import time
from albumentations import Compose, Normalize
from viz import VisualizeSegmm
torch.backends.cudnn.benchmark=True


DATA_DIRECTORY = '/home/sanskar/Unsupervised_Domian_Adaptation/LoveDA/Train/Urban/images_png/'
MASK_DIR= '/home/sanskar/Unsupervised_Domian_Adaptation/LoveDA/Train/Urban/masks_png/'
SAVE_PATH = './eval'

IGNORE_LABEL = 255
NUM_CLASSES = 7
NUM_STEPS = 500
BATCH_SIZE = 2
RESTORE_FROM = './pretrained/rural_to_urban_final_source_free.pth'

SET = 'Val'
MODEL = 'DeepLab'

info= {"classes":7, "label":['Background', 'Building', 'Road', 'Water',  'Barren', 'Forest', 'Agricultural']}

TRANSFORMS=dict(transforms= Compose([
        Normalize(mean=(123.675, 116.28, 103.53),
                  std=(58.395, 57.12, 57.375),
                  max_pixel_value=1, always_apply=True),
        # er.preprocess.albu.ToTensor()
    ]))

COLOR_MAP = OrderedDict(
    Background=(255, 255, 255),
    Building=(255, 0, 0),
    Road=(255, 255, 0),
    Water=(0, 0, 255),
    Barren=(159, 129, 183),
    Forest=(0, 255, 0),
    Agricultural=(255, 195, 128),
)
palette = np.asarray(list(COLOR_MAP.values())).reshape((-1,)).tolist()
# palette = [255, 255, 255, 255, 0, 0, 255, 255, 0, 0, 0, 255, 159, 129, 183, 0, 255, 0, 255, 195, 128] # red, yellow, blue, purple, green, orange, white.
# zero_pad = 256 * 3 - len(palette)
# for i in range(zero_pad):
#     palette.append(0)


def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="Model Choice (DeeplabMulti/DeeplabVGG/Oracle).")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the Cityscapes dataset.")
    parser.add_argument("--mask-dir", type=str, default=MASK_DIR,
                        help="Path to the directory containing the urban dataset mask.")
    parser.add_argument("--transform", type=dict, default=TRANSFORMS,
                        help="dictionary containing data transformations.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--restore_from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--gpu", type=int, default=0,
                        help="choose gpu device.")
    parser.add_argument("--batchsize", type=int, default=BATCH_SIZE,
                        help="choose gpu device.")
    parser.add_argument("--set", type=str, default=SET,
                        help="choose evaluation set.")
    parser.add_argument("--save", type=str, default=SAVE_PATH,
                        help="Path to save result.")
    return parser.parse_args()

def save(output_name):
    output, name = output_name
    # output_col = colorize_mask(output)
    output = Image.fromarray(output)
    output.save('%s' % (name))
    # output_col.save('%s_color.png' % (name_col.split('.png')[0]))
    return


def main():
    args = get_arguments()
    print('ModelType:%s'%args.model)

    gpu0 = args.gpu
    batchsize = args.batchsize

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    if args.model == 'DeepLab':
        model = get_deeplab_v2(num_classes=7, multi_level=False)

    saved_state_dict = torch.load(args.restore_from)
    model.load_state_dict(saved_state_dict)
    model.eval()
    model.cuda(gpu0)
    transform= args.transform["transforms"]
    testloader = data.DataLoader(lovedaDataSet(args.data_dir, args.mask_dir, set=args.set,transforms=transform),batch_size=batchsize, shuffle=False, pin_memory=True)

    interp = nn.Upsample(size=(1024, 1024), mode='bilinear', align_corners=True)

    for index, img_data in enumerate(testloader):
        image, _,name = img_data
        name= list(name)
        inputs = image.cuda()

        print('\r>>>>Extracting feature...%03d'%(index*batchsize), end='')
        if args.model == 'DeepLab':
            with torch.no_grad():
                output1, output2 = model(inputs)
                output_batch = interp((output2))
                del output1, output2, inputs
                output_batch = output_batch.cpu().data.numpy()
        output_batch = output_batch.transpose(0,2,3,1)
        output_batch = np.asarray(np.argmax(output_batch, axis=3), dtype=np.uint8)
        output_iterator = []

        for i in range(output_batch.shape[0]):
            output_iterator.append(output_batch[i,:,:])
            name[i] = '%s/%s' % (args.save, name[i])
        
        with Pool(4) as p:
            p.map(save, zip(output_iterator, name) )

        del output_batch

    return args.save

if __name__ == '__main__':
    tt = time.time()
    with torch.no_grad():
        save_path = main()
    print('Time used: {} sec'.format(time.time()-tt))
    # os.system('python compute_iou.py /home/sanskar/Unsupervised_Domian_Adaptation/LoveDA/Train/Urban/masks_png/  %s'%save_path)
