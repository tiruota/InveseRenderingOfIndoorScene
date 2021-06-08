import os
import os.path as osp
import json
import argparse

import numpy as np
import cv2
import matplotlib.pyplot as plt
from numpy.lib.type_check import imag
from torch.utils.data import Dataset
from shapely.geometry import Polygon
from descartes.patch import PolygonPatch

from utils import get_corners_of_bb3d_no_index, project_3d_points_to_2d, parse_camera_info

class BatchLoader(Dataset):
    def __init__(self, dataRoot, imHeight, imWidth, phase='TRAIN', rseed = None, cascadeLevel = 0):
        self.dataRoot = dataRoot
        self.imHeight = imHeight
        self.imWidth = imWidth
        self.room_list = []
        self.albedo_list = []
        self.depth_list = []
        self.normal_list = []
        self.rgb_list = []
        self.semantic_list = []
        self.instance_list = []

        self.camera_list = []
        self.layout_list = []

        if phase.upper() == 'TRAIN':
            self.sceneFile = osp.join(dataRoot, 'train.txt')
        elif phase.upper() == 'TEST':
            self.sceneFile = osp.join(dataRoot, 'test.txt') 
        else:
            print('Unrecognized phase for data loader')
            assert(False ) 

        folder_list = [folder_name for folder_name in os.listdir(dataRoot) if not folder_name.startswith('.')]
        for folder_id in [folder_name for folder_name in os.listdir(dataRoot) if not folder_name.startswith('.')]:
            folder_path = os.path.join(dataRoot, folder_id)
            for scene_id in [scene_name for scene_name in os.listdir(folder_path) if not scene_name.startswith('.')]:
                scene_path = os.path.join(folder_path, scene_id, "2D_rendering")
                for room_id in [room_name for room_name in os.listdir(scene_path) if not room_name.startswith('.')]:
                    room_path = os.path.join(scene_path, room_id, "perspective", "full")
                    for position_id in [position_name for position_name in os.listdir(room_path) if not position_name.startswith('.')]:
                        posittion_path = os.path.join(room_path, position_id)
                        self.room_list.append(posittion_path)
                        self.albedo_list.append(os.path.join(posittion_path, "albedo.png"))
                        self.depth_list.append(os.path.join(posittion_path, "depth.png"))
                        self.normal_list.append(os.path.join(posittion_path, "normal.png"))
                        self.rgb_list.append(os.path.join(posittion_path, "rgb_rawlight.png"))
                        self.semantic_list.append(os.path.join(posittion_path, "semantic.png"))
                        self.instance_list.append(os.path.join(posittion_path, "instance.png"))
                        self.camera_list.append(os.path.join(posittion_path, "camera_pose.txt"))
                        self.layout_list.append(os.path.join(posittion_path, "layout.json"))

    def __len__(self):
        return len(self.room_list)

    def __getitem__(self, idx):
        # print(self.albedo_list[idx], self.depth_list[idx], self.depth_list[idx], self.rgb_list[idx],self.semantic_list[idx],self.instance_list[idx])
        albedo = self.load_image(self.albedo_list[idx])
        depth = self.load_depth(self.depth_list[idx])
        normal = self.load_image(self.normal_list[idx])
        rgb = self.load_image(self.rgb_list[idx])
        semantic = self.load_image(self.semantic_list[idx])
        instance = self.load_instance(self.instance_list[idx])
        instance = instance[0:1, :, :]
        segArea = np.logical_and(instance > 0.49, instance < 0.51 ).astype(np.float32)
        segEnv = (instance < 0.1).astype(np.float32)
        segObj = (instance > 0.9).astype(np.float32)
        # segEnv = (rgb & instance).astype(np.float32)
        # segObj = (rgb & ~instance).astype(np.float32)
        camera_info = self.load_camera_info(self.camera_list[idx], height=self.imHeight, width=self.imWidth)
        layout = self.load_json(self.layout_list[idx])

        sample = {'albedo': albedo,
                'depth': depth,
                'normal': normal,
                'rgb': rgb,
                'semantic':semantic,
                'segArea':segArea,
                'segEnv':segEnv,
                'segObj':segObj,
                # 'camera_info':camera_info,
                # 'layout': layout
                }
        return sample

    def load_image(self, img_path, isGamma=False):
        image = cv2.imread(img_path)
        image = cv2.resize(image, (self.imWidth, self.imHeight))
        image = np.asarray(image, dtype=np.float32)
        if isGamma:
            image = (image / 255.0) ** 2.2
            image = 2*image - 1
        else:
            image = (image - 127.5) / 127.5
        image = np.transpose(image, [2,0,1])
        return image

    def load_json(self, json_path):
        with open(json_path) as f:
            annos = json.load(f)
        return annos

    def load_camera_info(self, txt_path, height, width):
        camera_info = np.loadtxt(txt_path)
        rot, trans, K = parse_camera_info(camera_info, height=height, width=width)
        return [rot, trans, K]

    def load_depth(self, img_path):
        # Depth image takes a range of [0:65535].
        # It　is normalized to a range of [0:1] or [-1:1]
        image = cv2.imread(img_path, -1)
        image = cv2.resize(image, (self.imWidth, self.imHeight))
        image = np.asarray(image, dtype=np.float32)
        image = (image - 32767.5) / 32767.5
        image = image[np.newaxis, :, :]
        # image = np.concatenate([image, image, image], axis=0)
        return image

    def load_instance(self, img_path):
        # Depth image takes a range of [0:65535].
        # It　is normalized to a range of [0:1] or [-1:1]
        image = cv2.imread(img_path, -1)
        image = cv2.resize(image, (self.imWidth, self.imHeight))
        image = np.asarray(image, dtype=np.float32)
        image = (image - 32767.5) / 32767.5
        image = image[np.newaxis, :, :]
        image = np.concatenate([image, image, image], axis=0)
        return image