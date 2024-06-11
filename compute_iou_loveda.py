import numpy as np
import argparse
import json
from PIL import Image
from os.path import join
import glob
import os


def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

info= {
  "classes":7,
  "label":["Background", "Building", "Road", "Water",  "Barren", "Forest", "Agricultural"],
  "palette":[[255, 255, 255], [255, 0, 0], [255, 255, 0], [0, 0, 255], [159, 129, 183], [0, 255, 0], [255, 195, 128]]
}

def compute_mIoU(gt_dir, pred_dir):
    """
    Compute IoU given the predicted colorized images and 
    """
    num_classes = int(info['classes'])
    print(('Num classes', num_classes))
    name_classes = np.array(info['label'], dtype=str)
    hist = np.zeros((num_classes, num_classes))

    gt_list = []
    pred_list=[]
    
    gt_list += glob.glob(os.path.join(gt_dir, '*.png'))
    pred_list += glob.glob(os.path.join(pred_dir, '*.png'))
    
    for ind in range(len(gt_list)):
        pred = np.array(Image.open(pred_list[ind]))
        label = np.array(Image.open(gt_list[ind]))
        if len(label.shape) == 3 and label.shape[2]==4:
            label = label[:,:,0]
        if len(label.flatten()) != len(pred.flatten()):
            print(('Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(len(label.flatten()), len(pred.flatten()), gt_list[ind], pred_list[ind])))
            continue
        hist += fast_hist(label.flatten(), pred.flatten(), num_classes)
        if ind > 0 and ind % 100 == 0:
            print(('{:d} / {:d}: {:0.1f}'.format(ind, len(gt_list), 100*np.mean(per_class_iu(hist)))))
    
    mIoUs = per_class_iu(hist)
    for ind_class in range(num_classes):
        print(('===>' + name_classes[ind_class] + ':\t' + str(round(mIoUs[ind_class] * 100, 1))))
    print(('===> mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 1))))
    return mIoUs

def main(args):
    compute_mIoU(args.gt_dir, args.pred_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('gt_dir', type=str, help='directory which stores gt images')
    parser.add_argument('pred_dir', type=str, help='directory which stores pred images')
    args = parser.parse_args()
    main(args)
