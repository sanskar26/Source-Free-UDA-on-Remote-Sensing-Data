import argparse
import numpy as np
import torch
import os
from PIL import Image
from torch.utils import data
import random

from dataset.loveda_dataset import lovedaDataSet
from loveda_source_model.source_model_loveda import Deeplabv2 
from albumentations import HorizontalFlip, VerticalFlip, RandomRotate90, Normalize, RandomCrop, RandomScale
from albumentations import OneOf, Compose
from collections import OrderedDict
from viz import VisualizeSegmm

torch.backends.cudnn.benchmark = True

# target_dir = dict(
#     image_dir=[
#         './LoveDA/Val/Urban/images_png/',
#     ],
#     mask_dir=[
#         './LoveDA/Val/Urban/masks_png/',
#     ],
# )

# TARGET_DATA_CONFIG = dict(
#     image_dir=target_dir['image_dir'],
#     mask_dir=target_dir['mask_dir'],
#     transforms=Compose([
#         RandomCrop(512, 512),
#         OneOf([
#             HorizontalFlip(True),
#             VerticalFlip(True),
#             RandomRotate90(True)
#         ], p=0.75),
#         Normalize(mean=(123.675, 116.28, 103.53),
#                   std=(58.395, 57.12, 57.375),
#                   max_pixel_value=1, always_apply=True),
#         er.preprocess.albu.ToTensor()

#     ]),
#     batch_size=8, 
#     num_workers=0, # CHANGED FROM 2
# )

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

DATA_DIRECTORY = '/home/sanskar/Unsupervised_Domian_Adaptation/LoveDA/Train/Urban/images_png/'
SAVE_PATH = './data/Urban/Pseudo_labels/train'
SAVE_PATH_COLOR = './data/Urban/Pseudo_labels/train_color'
TRANSFORMS=dict(transforms= Compose([
        Normalize(mean=(123.675, 116.28, 103.53),
                  std=(58.395, 57.12, 57.375),
                  max_pixel_value=1, always_apply=True),
        # er.preprocess.albu.ToTensor()
    ]))

if not os.path.isdir(SAVE_PATH[:-6]):
    os.makedirs(SAVE_PATH[:-6])
    os.mkdir(SAVE_PATH)
    os.mkdir(SAVE_PATH_COLOR)

IGNORE_LABEL = 255
NUM_CLASSES = 7 # (building,road,water,barren,forest agriculture,background)
BATCH_SIZE = 2
RESTORE_FROM = "/home/sanskar/Unsupervised_Domian_Adaptation/log/cbst/2urban/URBAN32000.pth"
SET = 'train'
MODEL = 'DeepLab'
MASK_DIR= "/home/sanskar/Unsupervised_Domian_Adaptation/LoveDA/Train/Urban/masks_png/"

# palette = [255, 255, 255, 255, 0, 0, 255, 255, 0, 0, 0, 255, 159, 129, 183, 0, 255, 0, 255, 195, 128] # red, yellow, blue, purple, green, orange, white.
# zero_pad = 256 * 3 - len(palette)
# for i in range(zero_pad):
#     palette.append(0)

def colorize_mask(mask):
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask

def get_arguments():
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--model", type=str, default=MODEL, help="Model Choice (Deeplabv2/DeeplabVGG).")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the urban dataset.")
    parser.add_argument("--mask-dir", type=str, default=MASK_DIR,
                        help="Path to the directory containing the urban dataset mask.")
    parser.add_argument("--transform", type=dict, default=TRANSFORMS,
                        help="dictionary containing data transformations.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM, help="Where restore model parameters from.")
    parser.add_argument("--gpu", type=int, default=0, help="choose gpu device.")
    parser.add_argument("--batchsize", type=int, default=BATCH_SIZE, help="choose gpu device.")
    parser.add_argument("--set", type=str, default=SET, help="choose evaluation set.")
    parser.add_argument("--save", type=str, default=SAVE_PATH, help="Path to save result.")
    parser.add_argument("--save-color", type=str, default=SAVE_PATH_COLOR, help="Path to save color results.")
    return parser.parse_args()


def main():
    """Load the source-trained weights on the source model and start the plabel generation process on target dataset."""

    args = get_arguments()
    gpu0 = args.gpu
    batchsize = args.batchsize
    weights= args.restore_from
    set= args.set
    save_path_color= args.save_color
    transform= args.transform["transforms"]

    model = Deeplabv2(dict(
        backbone=dict(
                resnet_type='resnet50',
                output_stride=16,
                pretrained=True,
            ),
        multi_layer=False,
        cascade=False,
        use_ppm=True,
        ppm=dict(
            num_classes=7,
            use_aux=False,
        ),
        inchannels=2048,
        num_classes=7
    ))

    model.load_state_dict(torch.load(weights),strict=True)
    model.eval()
    model.cuda(gpu0)

    testloader = data.DataLoader(lovedaDataSet(args.data_dir, args.mask_dir, mirror=False, set=set, transforms=transform), batch_size=batchsize)

    for i, batch in enumerate(testloader):
        image, _,name= batch
        input = image.cuda()
       

        print('\r>>>>Extracting feature...%04d' % (i* batchsize), end='')
        
        with torch.no_grad():
            output= model(input)
            output= output.cpu().data.numpy()

        output = output.transpose(0, 2, 3, 1)
        output = np.asarray(np.argmax(output, axis=3), dtype=np.uint8)

        for i in range(output.shape[0]):
            out = output[i, :, :]
            # out_col = colorize_mask(out)
            viz_op = VisualizeSegmm(save_path_color, palette)
            viz_op(out,name[i].replace('tif', 'png'))
            out = Image.fromarray(out)
            save_path = args.save
            out.save('%s/%s' % (save_path,name[i]))
            # out_col.save('%s/color_%s' % (save_path_color,name[i]))

    return out


if __name__ == '__main__':
    seed=2333
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) 
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.enabled = True
    with torch.no_grad():
        save_path = main()
    # os.system('python compute_iou.py /home/sanskar/Unsupervised_Domian_Adaptation/LoveDA/Train/Urban/masks_png %s' % SAVE_PATH)
    
