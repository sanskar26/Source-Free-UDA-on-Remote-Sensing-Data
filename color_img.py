from PIL import Image
import numpy as np
from dataset.loveda_dataset import lovedaDataSet
from torch.utils import data
from collections import OrderedDict
from viz import VisualizeSegmm
import os

DATA_DIRECTORY = '/home/sanskar/Unsupervised_Domian_Adaptation/LoveDA/Train/Urban/images_png/'
SAVE_PATH = './color_mask_pred'
MASK_DIR= "/home/sanskar/GTA5/LOVEDA/eval"

if not os.path.isdir(SAVE_PATH):
    os.makedirs(SAVE_PATH)

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
    new_mask = Image.fromarray(mask).convert('P')
    new_mask.putpalette(palette)
    return new_mask

if __name__ == '__main__':
    testloader = data.DataLoader(lovedaDataSet(DATA_DIRECTORY, MASK_DIR), batch_size=1)
    for i, batch in enumerate(testloader):
        image,label,name= batch
        viz_op = VisualizeSegmm(SAVE_PATH, palette)
        viz_op(np.array(label[0,:,:]),name[0].replace('tif', 'png'))
        # out_col = colorize_mask(np.array(label[0,:,:],dtype= np.uint8 ))
        # out_col.save('%s/color_%s' % (SAVE_PATH,name[0]))