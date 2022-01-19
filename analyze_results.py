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

    parser.add_argument('--json_path',
                        help='results json file path',
                        type=str,
                        default='output/coco/pose_hrnet/w48_384x288_adam_lr1e-3/results/keypoints_val2017_results_0.json')

    parser.add_argument('--vis_num',
                        help='number of instances to visualize (applied for low, high cases respectively)',
                        type=int,
                        default=20)

    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    annot_path = '{}/annotations/person_keypoints_{}.json'.format(args.coco_base_dir,'val2017')
    coco=COCO(annot_path)

    json_data = None
    with open(args.json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    print(f'len(json_data):{len(json_data)}')
    
    json_sorted = sorted(json_data, key=lambda e: e['score'])

    score_dict = {'0-20':0, '20-40':0, '40-60':0, '60-80':0, '80-100':0}

    for e in json_sorted:
        score = e['score']
        if score>0.8:
            score_dict['80-100'] += 1
        elif score > 0.6:
            score_dict['60-80'] += 1
        elif score > 0.4:
            score_dict['40-60'] += 1
        elif score > 0.2:
            score_dict['20-40'] += 1
        else:
            score_dict['0-20'] += 1
    print(score_dict)

    img_base_dir = path.join(args.coco_base_dir, 'val2017')

    def draw_keypoints(reverse_order=False):
        rng = range(-args.vis_num,0) if reverse_order else range(args.vis_num)
        circle_color = [0, 0, 255] # red
        for vis_i in rng:
            img_dict = coco.loadImgs(json_sorted[vis_i]['image_id'])
            gt_joint_dict = coco.getAnnIds(imgIds=json_sorted[vis_i]['image_id'], catIds=[1])
            gt_joints = [e['keypoints'] for e in coco.loadAnns(gt_joint_dict)] # could contain multiple instances

            filename = img_dict[0]['file_name']
            order = 'high' if reverse_order else 'low'
            print(f'{vis_i}th {order} img file: {filename}, score: {json_sorted[vis_i]["score"]}')
            img_path = path.join(img_base_dir, filename)

            img = cv2.imread(img_path)
            gt_img = np.copy(img)
            pred_img = np.copy(img)

            # draw gt keypoints
            gt_joints = np.array(gt_joints).reshape(-1,3)

            for joint in gt_joints:
                if int(joint[-1]): # only draw the valid joints
                    cv2.circle(gt_img, (int(joint[0]), int(joint[1])), 2, circle_color, 2)

            cv2.imwrite(path.join('/'.join(args.json_path.split('/')[:-1]), filename.replace('.jpg', '_gt.jpg')), gt_img)

            # draw pred keypoints
            pred_joints = np.array(json_sorted[vis_i]['keypoints']).reshape(17,3)

            for joint in pred_joints:
                #if joint[-1]>1.0e-2: # only draw the valid joints
                cv2.circle(pred_img, (int(joint[0]), int(joint[1])), 2, circle_color, 2)

            cv2.imwrite(path.join('/'.join(args.json_path.split('/')[:-1]), filename.replace('.jpg', f'_pred_{json_sorted[vis_i]["score"]:.2f}.jpg')), pred_img)

    draw_keypoints(reverse_order=False) # lowest scores
    draw_keypoints(reverse_order=True) # highest scores

    print("Finished")


if __name__ == '__main__':
    main()
