# encoding: utf-8
"""This file includes necessary params, info."""
import os
import mmcv
import os.path as osp

import numpy as np
import yaml

# ---------------------------------------------------------------- #
# ROOT PATH INFO
# ---------------------------------------------------------------- #
cur_dir = osp.abspath(osp.dirname(__file__))
root_dir = osp.normpath(osp.join(cur_dir, ".."))
# directory storing experiment data (result, model checkpoints, etc).
output_dir = osp.join(root_dir, "output")

data_root = osp.join(root_dir, "datasets")
bop_root = osp.join(data_root, "BOP_DATASETS/")

# ---------------------------------------------------------------- #
# YCBV DATASET
# ---------------------------------------------------------------- #
dataset_root = osp.join(bop_root, "ycbv_random_texture")

train_real_dir = osp.join(dataset_root, "train_real")
train_render_dir = osp.join(dataset_root, "train_synt")
train_pbr_dir = osp.join(dataset_root, "train_pbr")

test_dir = osp.join(dataset_root, "test")

test_scenes = [i for i in range(48, 59 + 1)]
train_real_scenes = [i for i in range(0, 91 + 1) if i not in test_scenes]
train_synt_scenes = [i for i in range(0, 79 + 1)]
train_pbr_scenes = [i for i in range(0, 49 + 1)]

model_dir = osp.join(dataset_root, "models")
fine_model_dir = osp.join(dataset_root, "models_fine")
model_eval_dir = osp.join(dataset_root, "models_eval")
model_scaled_simple_dir = osp.join(dataset_root, "models_rescaled")  # m, .obj
vertex_scale = 0.001

# object info
object_id_file = osp.join(dataset_root, 'data/ycbv.yaml')
with open(object_id_file, 'r') as stream:
    data_loaded = yaml.safe_load(stream)
id2obj = data_loaded["names"]
objects = list(id2obj.values())

obj_num = len(id2obj)
obj2id = {_name: _id for _id, _name in id2obj.items()}

model_paths = [osp.join(model_dir, "obj_{:06d}.ply").format(_id) for _id in id2obj]  # TODO: check this
texture_paths = [osp.join(model_dir, "obj_{:06d}.png".format(_id)) for _id in id2obj]
model_colors = [((i + 1) * 10, (i + 1) * 10, (i + 1) * 10) for i in range(obj_num)]  # for renderer

# yapf: disable
diameters = np.array([172.063, 269.573, 198.377, 120.543, 196.463,
                      89.797,  142.543, 114.053, 129.540, 197.796,
                      259.534, 259.566, 161.922, 124.990, 226.170,
                      237.299, 203.973, 121.365, 174.746, 217.094,
                      102.903]) / 1000.0
# yapf: enable
# Camera info
width = 640
height = 480
zNear = 0.25
zFar = 6.0
center = (height / 2, width / 2)
# default: 0000~0059 and synt
camera_matrix = uw_camera_matrix = np.array([[1066.778, 0.0, 312.9869], [0.0, 1067.487, 241.3109], [0.0, 0.0, 1.0]])
# 0060~0091
cmu_camera_matrix = np.array([[1077.836, 0.0, 323.7872], [0.0, 1078.189, 279.6921], [0.0, 0.0, 1.0]])

depth_factor = 10000.0


def get_models_info():
    """key is str(obj_id)"""
    models_info_path = osp.join(model_dir, "models_info.json")
    assert osp.exists(models_info_path), models_info_path
    models_info = mmcv.load(models_info_path)  # key is str(obj_id)
    return models_info


def get_fps_points():
    """key is str(obj_id) generated by
    core/gdrn_modeling/tools/ycbv/ycbv_1_compute_fps.py."""
    fps_points_path = osp.join(model_dir, "fps_points.pkl")
    assert osp.exists(fps_points_path), fps_points_path
    fps_dict = mmcv.load(fps_points_path)
    return fps_dict


def get_keypoints_3d():
    """key is str(obj_id) generated by
    core/roi_pvnet/tools/ycbv/ycbv_1_compute_keypoints_3d.py."""
    keypoints_3d_path = osp.join(model_dir, "keypoints_3d.pkl")
    assert osp.exists(keypoints_3d_path), keypoints_3d_path
    kpts_dict = mmcv.load(keypoints_3d_path)
    return kpts_dict
