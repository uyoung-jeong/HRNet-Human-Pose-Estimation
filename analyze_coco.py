# simple performance analysis and visualization
from pycocotools.coco import COCO
import numpy as np
import cv2

import json
import os
from os import path
import argparse

import ipdb

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--coco_base_dir',
                        help='coco base directory',
                        type=str,
                        default='/syn_mnt/uyoung/human/coco')

    args = parser.parse_args()
    return args

def calculate_vis(coco_obj):
    num_vis_dict = dict()
    for i in range(18):
        num_vis_dict[str(i)] = 0

    for k,v in coco_obj.anns.items():
        v.keys() # ['segmentation', 'num_keypoints', 'area', 'iscrowd', 'keypoints', 'image_id', 'bbox', 'category_id', 'id']
        keypoints = np.array(v['keypoints']).reshape((17,3))
        num_vis = np.sum(keypoints[:,2]>0)
        num_vis_dict[str(num_vis)] += 1
    return num_vis_dict

def main():
    args = parse_args()

    train_annot_path = '{}/annotations/person_keypoints_{}.json'.format(args.coco_base_dir,'train2017')
    coco_train=COCO(train_annot_path)
    """
    num_vis_train = calculate_vis(coco_train)
    print("number of visible keypoints in train dataset:")
    print(num_vis_train)
    """
    #{'0': 112652, '1': 3270, '2': 4030, '3': 3541, '4': 4758, '5': 4452, '6': 6976, '7': 6765, '8': 7577, '9': 8481, '10': 10327, '11': 11169, '12': 13995, '13': 14374, '14': 12117, '15': 13858, '16': 15648, '17': 8475}

    val_annot_path = '{}/annotations/person_keypoints_{}.json'.format(args.coco_base_dir,'val2017')
    coco_val=COCO(val_annot_path)
    """
    num_vis_val = calculate_vis(coco_train)
    print("number of visible keypoints in val dataset:")
    print(num_vis_val)
    """
    #{'0': 112652, '1': 3270, '2': 4030, '3': 3541, '4': 4758, '5': 4452, '6': 6976, '7': 6765, '8': 7577, '9': 8481, '10': 10327, '11': 11169, '12': 13995, '13': 14374, '14': 12117, '15': 13858, '16': 15648, '17': 8475}

    print("Finished")


if __name__ == '__main__':
    main()
