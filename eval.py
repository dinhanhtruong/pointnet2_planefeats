"""
Author: Benny
Date: Nov 2019
"""

import os
import sys
import torch
import numpy as np
import wandb
import datetime
import logging
import provider
import importlib
import random
import shutil
import argparse
import pytorch3d.loss
import numpy as np
from PIL import Image
import imageio
import struct
import os
from scipy.spatial import distance

from pathlib import Path
from tqdm import tqdm
from data_utils.ModelNetDataLoader import ModelNetDataLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
    parser.add_argument('--model', default='pointnet_cls', help='model name [default: pointnet_cls]')
    parser.add_argument('--num_category', default=40, type=int, choices=[10, 40],  help='training on ModelNet10/40')
    parser.add_argument('--epoch', default=400, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--num_point', type=int, default=2048, help='Point Number')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    return parser.parse_args()


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def write_pointcloud(filename,xyz_points,rgb_points=None):
    """ creates a .pkl file of the point clouds generated
    """
    assert xyz_points.shape[1] == 3,'Input XYZ points should be Nx3 float array'
    if rgb_points is None:
        rgb_points = np.ones(xyz_points.shape).astype(np.uint8)*255
    assert xyz_points.shape == rgb_points.shape,'Input RGB colors should be Nx3 float array and have same size as input XYZ points'

    # Write header of .ply file
    fid = open(filename,'wb')
    fid.write(bytes('ply\n', 'utf-8'))
    fid.write(bytes('format binary_little_endian 1.0\n', 'utf-8'))
    fid.write(bytes('element vertex %d\n'%xyz_points.shape[0], 'utf-8'))
    fid.write(bytes('property float x\n', 'utf-8'))
    fid.write(bytes('property float y\n', 'utf-8'))
    fid.write(bytes('property float z\n', 'utf-8'))
    fid.write(bytes('property uchar red\n', 'utf-8'))
    fid.write(bytes('property uchar green\n', 'utf-8'))
    fid.write(bytes('property uchar blue\n', 'utf-8'))
    fid.write(bytes('end_header\n', 'utf-8'))

    # Write 3D points to .ply file
    for i in range(xyz_points.shape[0]):
        fid.write(bytearray(struct.pack("fffccc",xyz_points[i,0],xyz_points[i,1],xyz_points[i,2],
                                        rgb_points[i,0].tostring(),rgb_points[i,1].tostring(),
                                        rgb_points[i,2].tostring())))
    fid.close()

def get_point_cloud(model, loader):
    model.eval()

    for j, (points, shape_names) in tqdm(enumerate(loader), total=len(loader)):
        if not args.use_cpu:
            points = points.cuda()

        points = points.transpose(2, 1)
        pred, latent_feat = model(points)

        # save PLY
        for shape_pc, name in zip(pred, shape_names):
            print("shape name: ", name)
            write_pointcloud(f"plane_feats/point_clouds/{name}_recon.ply", shape_pc.cpu().numpy())
        
def extract_feats(model, loader):
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    model.eval()
    name_to_latent_feat = {} # string ID to np feat array
    for j, (points, name) in tqdm(enumerate(loader), total=len(loader)):
        if not args.use_cpu:
            points = points.cuda()
        points = points.transpose(2, 1)
        pred, latent_feat = model(points)
        for feat_instance, name_instance in zip(latent_feat, name):
            name_to_latent_feat[name_instance] = feat_instance.detach().cpu().numpy()
    
    latent_feats = np.stack(list(name_to_latent_feat.values()), axis=0)
    pairwise_dists = distance.cdist(latent_feats,latent_feats)
    print("pairwise dists: ")
    print(pairwise_dists)
    print("saving feats: ", latent_feats.shape)
    np.save("plane_feats/spaghetti_gt_plane_feats.npy", latent_feats)

    np.savez("plane_feats/spaghetti_id_to_pointnet_feat", **name_to_latent_feat)
    
    np.save("plane_feats/pairwise_dists.npy", pairwise_dists)


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    experiment_dir = 'log/reconstruction/' + args.log_dir

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    data_path = '../shapenet_all_planes'# 'data/modelnet40_normal_resampled/'

    full_dataset = ModelNetDataLoader(root=data_path, args=args, split='spaghetti', process_data=False)
    fullDataLoader = torch.utils.data.DataLoader(full_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    '''MODEL LOADING'''
    model = importlib.import_module(args.model)
    model = model.get_model(args.num_point, normal_channel=args.use_normals)
    model.apply(inplace_relu)

    if not args.use_cpu:
        model = model.cuda()
    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])


    with torch.no_grad():
        # get_point_cloud(model.eval(), fullDataLoader)
        extract_feats(model.eval(), fullDataLoader)



if __name__ == '__main__':
    args = parse_args()
    main(args)
