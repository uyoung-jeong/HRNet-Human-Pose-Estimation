# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.transforms import get_affine_transform
from utils.transforms import affine_transform
from utils.transforms import fliplr_joints

import random

logger = logging.getLogger(__name__)


class JointsDataset(Dataset):
    def __init__(self, cfg, root, image_set, is_train, transform=None, args=None):
        self.num_joints = 0
        self.pixel_std = 200
        self.flip_pairs = []
        self.parent_ids = []

        self.is_train = is_train
        self.root = root
        self.image_set = image_set

        self.output_path = cfg.OUTPUT_DIR
        self.data_format = cfg.DATASET.DATA_FORMAT

        self.scale_factor = cfg.DATASET.SCALE_FACTOR
        self.rotation_factor = cfg.DATASET.ROT_FACTOR
        self.flip = cfg.DATASET.FLIP
        self.num_joints_half_body = cfg.DATASET.NUM_JOINTS_HALF_BODY
        self.prob_half_body = cfg.DATASET.PROB_HALF_BODY
        self.color_rgb = cfg.DATASET.COLOR_RGB

        self.target_type = cfg.MODEL.TARGET_TYPE
        self.image_size = np.array(cfg.MODEL.IMAGE_SIZE)
        self.heatmap_size = np.array(cfg.MODEL.HEATMAP_SIZE)
        self.sigma = cfg.MODEL.SIGMA
        self.use_different_joints_weight = cfg.LOSS.USE_DIFFERENT_JOINTS_WEIGHT
        self.joints_weight = 1

        self.transform = transform
        self.db = []

        self.cfg = cfg
        self.args = args

        # occllusion
        self.mask_size_range = [0.6, 1.4]
        self.center_jitter = [-0.5, 0.5]
        self.mask_shapes = ['triangle', 'rectangle', 'ellipse']

    def _get_db(self):
        raise NotImplementedError

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        raise NotImplementedError

    def half_body_transform(self, joints, joints_vis):
        upper_joints = []
        lower_joints = []
        for joint_id in range(self.num_joints):
            if joints_vis[joint_id][0] > 0:
                if joint_id in self.upper_body_ids:
                    upper_joints.append(joints[joint_id])
                else:
                    lower_joints.append(joints[joint_id])

        if np.random.randn() < 0.5 and len(upper_joints) > 2:
            selected_joints = upper_joints
        else:
            selected_joints = lower_joints \
                if len(lower_joints) > 2 else upper_joints

        if len(selected_joints) < 2:
            return None, None

        selected_joints = np.array(selected_joints, dtype=np.float32)
        center = selected_joints.mean(axis=0)[:2]

        left_top = np.amin(selected_joints, axis=0)
        right_bottom = np.amax(selected_joints, axis=0)

        w = right_bottom[0] - left_top[0]
        h = right_bottom[1] - left_top[1]

        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio

        scale = np.array(
            [
                w * 1.0 / self.pixel_std,
                h * 1.0 / self.pixel_std
            ],
            dtype=np.float32
        )

        scale = scale * 1.5

        return center, scale

    def __len__(self,):
        return len(self.db)

    def __getitem__(self, idx):
        db_rec = copy.deepcopy(self.db[idx])

        image_file = db_rec['image']
        filename = db_rec['filename'] if 'filename' in db_rec else ''
        imgnum = db_rec['imgnum'] if 'imgnum' in db_rec else ''

        if self.data_format == 'zip':
            from utils import zipreader
            data_numpy = zipreader.imread(
                image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
            )
        else:
            data_numpy = cv2.imread(
                image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
            )

        if self.color_rgb:
            data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)

        if data_numpy is None:
            logger.error('=> fail to read {}'.format(image_file))
            raise ValueError('Fail to read {}'.format(image_file))

        joints = db_rec['joints_3d']
        joints_vis = db_rec['joints_3d_vis']

        c = db_rec['center']
        s = db_rec['scale']
        score = db_rec['score'] if 'score' in db_rec else 1
        r = 0

        if self.is_train:
            if (np.sum(joints_vis[:, 0]) > self.num_joints_half_body
                and np.random.rand() < self.prob_half_body):
                c_half_body, s_half_body = self.half_body_transform(
                    joints, joints_vis
                )

                if c_half_body is not None and s_half_body is not None:
                    c, s = c_half_body, s_half_body

            # simulate occlusion
            if self.cfg.DATASET.OCC == True:
                if np.random.random_sample() > 0.5: # apply occlusion with 50% probability
                    if self.cfg.DATASET.OCC_TYPE == 'Circles':
                        data_numpy = self.circles_aug(data_numpy,self.cfg.DATASET.max_number_of_obj, self.cfg.DATASET.min_number_of_obj, image_file)
                    elif self.cfg.DATASET.OCC_TYPE == 'Rectangles':
                        data_numpy = self.rectangles_aug(data_numpy,self.cfg.DATASET.max_number_of_obj, self.cfg.DATASET.min_number_of_obj, image_file)
                    elif self.cfg.DATASET.OCC_TYPE == 'Bars':
                        data_numpy = self.bars_aug(data_numpy, self.cfg.DATASET.max_number_of_obj, self.cfg.DATASET.min_number_of_obj, image_file)
                    elif self.cfg.DATASET.OCC_TYPE == 'Mixed':
                        data_numpy = self.mixed_aug(data_numpy, image_file)
            sf = self.scale_factor
            rf = self.rotation_factor
            s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
            r = np.clip(np.random.randn()*rf, -rf*2, rf*2) \
                if random.random() <= 0.6 else 0

            if self.flip and random.random() <= 0.5:
                data_numpy = data_numpy[:, ::-1, :]
                joints, joints_vis = fliplr_joints(
                    joints, joints_vis, data_numpy.shape[1], self.flip_pairs)
                c[0] = data_numpy.shape[1] - c[0] - 1

        trans = get_affine_transform(c, s, r, self.image_size)
        input = cv2.warpAffine(
            data_numpy,
            trans,
            (int(self.image_size[0]), int(self.image_size[1])),
            flags=cv2.INTER_LINEAR)

        if self.transform:
            input = self.transform(input)

        for i in range(self.num_joints):
            if joints_vis[i, 0] > 0.0:
                joints[i, 0:2] = affine_transform(joints[i, 0:2], trans)

        target, target_weight = self.generate_target(joints, joints_vis)

        target = torch.from_numpy(target)
        target_weight = torch.from_numpy(target_weight)

        meta = {
            'image': image_file,
            'filename': filename,
            'imgnum': imgnum,
            'joints': joints,
            'joints_vis': joints_vis,
            'center': c,
            'scale': s,
            'rotation': r,
            'score': score
        }

        return input, target, target_weight, meta

    def select_data(self, db):
        db_selected = []
        for rec in db:
            num_vis = 0
            joints_x = 0.0
            joints_y = 0.0
            for joint, joint_vis in zip(
                    rec['joints_3d'], rec['joints_3d_vis']):
                if joint_vis[0] <= 0:
                    continue
                num_vis += 1

                joints_x += joint[0]
                joints_y += joint[1]
            if num_vis == 0:
                continue

            joints_x, joints_y = joints_x / num_vis, joints_y / num_vis

            area = rec['scale'][0] * rec['scale'][1] * (self.pixel_std**2)
            joints_center = np.array([joints_x, joints_y])
            bbox_center = np.array(rec['center'])
            diff_norm2 = np.linalg.norm((joints_center-bbox_center), 2)
            ks = np.exp(-1.0*(diff_norm2**2) / ((0.2)**2*2.0*area))

            metric = (0.2 / 16) * num_vis + 0.45 - 0.2 / 16
            if ks > metric:
                db_selected.append(rec)

        logger.info('=> num db: {}'.format(len(db)))
        logger.info('=> num selected db: {}'.format(len(db_selected)))
        return db_selected

    def generate_target(self, joints, joints_vis):
        '''
        :param joints:  [num_joints, 3]
        :param joints_vis: [num_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        '''
        target_weight = np.ones((self.num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints_vis[:, 0]

        assert self.target_type == 'gaussian', \
            'Only support gaussian map now!'

        if self.target_type == 'gaussian':
            target = np.zeros((self.num_joints,
                               self.heatmap_size[1],
                               self.heatmap_size[0]),
                              dtype=np.float32)

            tmp_size = self.sigma * 3

            for joint_id in range(self.num_joints):
                feat_stride = self.image_size / self.heatmap_size
                mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
                mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
                # Check that any part of the gaussian is in-bounds
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] \
                        or br[0] < 0 or br[1] < 0:
                    # If not, just return the image as is
                    target_weight[joint_id] = 0
                    continue

                # # Generate gaussian
                size = 2 * tmp_size + 1
                x = np.arange(0, size, 1, np.float32)
                y = x[:, np.newaxis]
                x0 = y0 = size // 2
                # The gaussian is not normalized, we want the center value to equal 1
                g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))

                # Usable gaussian range
                g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
                g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
                # Image range
                img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
                img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

                v = target_weight[joint_id]
                if v > 0.5:
                    target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                        g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        if self.use_different_joints_weight:
            target_weight = np.multiply(target_weight, self.joints_weight)

        return target, target_weight

    # method=='anchor': anchor on keypoint
    # method=='random': random positioning
    def occ_aug(self, data_numpy, joints, joints_vis, method='anchor'):
        data_numpy_occ = np.copy(data_numpy)
        n_joint = len(joints)
        vis_mask = joints_vis[:,0].astype(bool)
        visible_idxs = np.arange(n_joint)[vis_mask]
        occ_idxs = np.random.choice(visible_idxs,self.cfg.DATASET.OCC_HIDE_NUM) # select among visible joints

        for occ_idx in occ_idxs:
            dist = np.sqrt(np.sum((joints[occ_idx,:2] - np.delete(joints, occ_idx,axis=0)[:,:2])**2, axis=1))
            mask_dist = np.min(dist)
            mask_size = np.random.uniform(*self.mask_size_range) * mask_dist
            mask_shape = np.random.choice(self.mask_shapes)
            cur_pt = joints[occ_idx,:2].reshape((2,1))

            # mask the image
            center = [data_numpy.shape[0]//2, data_numpy.shape[1]//2]
            if method == 'anchor':
                center = cur_pt.flatten() + np.random.uniform(*self.center_jitter,2) * mask_dist
            elif method == 'random':
                center = np.array([np.random.randint(mask_dist//2,data_numpy.shape[0]-mask_dist//2),
                                   np.random.randint(mask_dist//2,data_numpy.shape[1]-mask_dist//2)])
            color = (0,0,0) if self.cfg.DATASET.OCC_COLOR=='black' else np.random.randint(255,size=3)
            if mask_shape == 'ellipse': # draw ellipse
                axes = np.random.uniform(*self.mask_size_range, 2) * mask_dist
                angle = np.random.randint(45)
                thickness = -1

                data_numpy_occ = cv2.ellipse(data_numpy_occ, center.astype(np.int32), axes.astype(np.int32), angle, 0, 360, color, thickness)
            else: # draw polygon
                num_vertices = 4 if mask_shape == 'rectangle' else 3
                vector = np.random.random_sample((2,1))
                vector = vector / np.linalg.norm(vector)
                vertices = [center + vector * mask_size]
                angle_acc = 0
                for v_i in range(num_vertices-1):
                    angle = 210 - angle_acc
                    if (v_i+1 < num_vertices) or (angle_acc > 180):
                        angle = np.radians(np.random.randint(30, 150))
                    rot_mat = np.array([[np.cos(angle), -np.sin(angle)],[np.sin(angle), np.cos(angle)]])
                    vector = np.dot(rot_mat, vector) # get next vertex by rotation
                    vertices.append(center + vector * mask_size)
                vertices = np.array(vertices).astype(np.int32).reshape((1,-1,2)) # need to expand dimension

                data_numpy_occ = cv2.fillPoly(data_numpy_occ, vertices, color)

        #cv2.imwrite(f'output/{idx}_occ.jpg',cv2.cvtColor(data_numpy_occ, cv2.COLOR_BGR2RGB))
        #cv2.imwrite(f'output/{idx}.jpg',cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB))
        return data_numpy_occ

    def circles_aug(self,data_numpy,max_number_of_obj, min_number_of_obj, image_file):
        data_numpy_occ = np.copy(data_numpy)
        number_of_obj = max_number_of_obj if max_number_of_obj == min_number_of_obj else np.random.randint(min_number_of_obj, max_number_of_obj)
        for i in range(number_of_obj):
            mask_dist = np.random.randint(data_numpy.shape[0] // 5)
            # mask the image
            center = np.array([np.random.randint(mask_dist // 2, data_numpy.shape[0] - mask_dist // 2),
                               np.random.randint(mask_dist // 2, data_numpy.shape[1] - mask_dist // 2)])
            color = [0, 0, 0] if self.cfg.DATASET.OCC_COLOR == 'black' else np.random.randint(255, size=3)
            color = tuple(int(i) for i in color)
            thickness = -1

            data_numpy_occ = cv2.circle(data_numpy_occ, center.astype(np.int32), mask_dist, color, thickness)

        # cv2.imwrite(f'output/Circles/{image_file[27:-4]}{i}_occ.jpg',cv2.cvtColor(data_numpy_occ, cv2.COLOR_BGR2RGB))
        # cv2.imwrite(f'output/Circles/{image_file[27:-4]}{i}.jpg',cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB))
        return data_numpy_occ

    def rectangles_aug(self,data_numpy,max_number_of_obj,min_number_of_obj,  image_file):
        data_numpy_occ = np.copy(data_numpy)
        number_of_obj = max_number_of_obj if max_number_of_obj == min_number_of_obj else np.random.randint(min_number_of_obj, max_number_of_obj)
        for i in range(number_of_obj):
            # finding distance
            height = np.random.randint(data_numpy.shape[0] // 5)
            width = np.random.randint(data_numpy.shape[1] // 5)

            # findig points
            _angle = np.random.randint(180) * np.pi / 180.0
            b = np.cos(_angle) * 0.5
            a = np.sin(_angle) * 0.5
            # Choosing Random Point
            x0 = np.random.randint(width // 2, data_numpy.shape[1] - width // 2)
            y0 = np.random.randint(height // 2, data_numpy.shape[0] - height // 2)
            # four points of the rectangle
            pt0 = (int(x0 - a * height - b * width),
                   int(y0 + b * height - a * width))
            pt1 = (int(x0 + a * height - b * width),
                   int(y0 - b * height - a * width))
            pt2 = (int(2 * x0 - pt0[0]), int(2 * y0 - pt0[1]))
            pt3 = (int(2 * x0 - pt1[0]), int(2 * y0 - pt1[1]))

            pts = np.array([pt0, pt1, pt2, pt3])
            pts = pts.reshape((-1, 1, 2))
            # mask the image

            color = [0, 0, 0] if self.cfg.DATASET.OCC_COLOR == 'black' else np.random.randint(255, size=3)
            color = tuple(int(i) for i in color)

            thickness = -1

            data_numpy_occ = cv2.fillPoly(data_numpy_occ, [pts], color)

        # cv2.imwrite(f'output/Rectangles/{image_file[27:-4]}{i}_occ.jpg',cv2.cvtColor(data_numpy_occ, cv2.COLOR_BGR2RGB))
        # cv2.imwrite(f'output/Rectangles/{image_file[27:-4]}{i}.jpg',cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB))
        return data_numpy_occ

    def bars_aug(self,data_numpy, max_number_of_obj, min_number_of_obj, image_file):
        def get_point(img,side):
            if side == 'N':
                y = 0
                x =  np.random.randint(0, (img.shape[1] - 1))
            elif side == 'S':
                y = img.shape[0] - 1
                x = np.random.randint(0, (img.shape[1] - 1))
            elif side == 'E':
                x = img.shape[1] - 1
                y = np.random.randint(0, (img.shape[0] - 1))
            else:
                x = 0
                y = np.random.randint(0, (img.shape[0] - 1))
            return (x, y)
        
        
        data_numpy_occ = np.copy(data_numpy)
        number_of_obj = max_number_of_obj if max_number_of_obj == min_number_of_obj else np.random.randint(min_number_of_obj, max_number_of_obj)
        
        sides = ['W', 'E', 'N', 'S']
        
        for i in range(number_of_obj):
            choosen_sides = random.sample(sides, 2)
            start_point = get_point(data_numpy_occ, choosen_sides[0])
            end_point = get_point(data_numpy_occ, choosen_sides[1])
            thickness = np.random.randint(10, 50)
            color = [0, 0, 0] if self.cfg.DATASET.OCC_COLOR == 'black' else np.random.randint(255, size=3)
            color = tuple(int(i) for i in color)
            
            data_numpy_occ = cv2.line(data_numpy_occ, start_point, end_point, color, thickness)

        # cv2.imwrite(f'output/Bars/{image_file[27:-4]}{i}_occ.jpg',cv2.cvtColor(data_numpy_occ, cv2.COLOR_BGR2RGB))
        # cv2.imwrite(f'output/Bars/{image_file[27:-4]}{i}.jpg',cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB))
        return data_numpy_occ

    def mixed_aug(self, data_numpy, image_file):
        data_numpy_occ = np.copy(data_numpy)
        number_of_obj = self.cfg.DATASET.max_number_of_obj if self.cfg.DATASET.max_number_of_obj == self.cfg.DATASET.min_number_of_obj else np.random.randint(
            self.cfg.DATASET.min_number_of_obj, self.cfg.DATASET.max_number_of_obj)
        aug_methods = ['Circles', 'Rectangles', 'Bars']
        for i in range(number_of_obj):
            aug_method = np.random.choice(aug_methods)
            if aug_method == 'Circles':
                data_numpy_occ = self.circles_aug(data_numpy_occ, 1, 1, image_file)
            if aug_method == 'Rectangles':
                data_numpy_occ = self.rectangles_aug(data_numpy_occ, 1, 1, image_file)
            if aug_method == 'Bars':
                data_numpy_occ = self.bars_aug(data_numpy_occ, 1, 1, image_file)

        # cv2.imwrite(f'output/Mixes_color/{image_file[27:-4]}{i}_occ.jpg',cv2.cvtColor(data_numpy_occ, cv2.COLOR_BGR2RGB))
        # cv2.imwrite(f'output/Mixes_color/{image_file[27:-4]}{i}.jpg',cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB))
        return data_numpy_occ