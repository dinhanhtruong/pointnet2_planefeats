'''
@author: Xu Yan
@file: ModelNet.py
@time: 2021/3/19 15:51
'''
import os
import numpy as np
import warnings
import pickle
import json
from tqdm import tqdm
from torch.utils.data import Dataset

warnings.filterwarnings('ignore')


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


class ModelNetDataLoader(Dataset):
    def __init__(self, root, args, split='train', process_data=False):
        self.root = root
        self.npoints = args.num_point
        self.process_data = process_data
        self.uniform = args.use_uniform_sample
        self.use_normals = args.use_normals
        self.num_category = args.num_category

        # if self.num_category == 10:
        #     self.catfile = os.path.join(self.root, 'modelnet10_shape_names.txt')
        # else:
        #     self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')

        # self.cat = [line.rstrip() for line in open(self.catfile)]
        # self.classes = dict(zip(self.cat, range(len(self.cat))))

        # shape_ids = {}
        # if self.num_category == 10:
        #     shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_train.txt'))]
        #     shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_test.txt'))]
        # else:
        #     shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))]
        #     shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]

        # assert (split == 'train' or split == 'test')
        # shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        # self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt') for i
        #                  in range(len(shape_ids[split]))]
        # print('The size of %s data is %d' % (split, len(self.datapath)))


        filepaths = []
        # filter + sort by spaghetti train set (maybe loop over spaghetti ids first, then find corresp npz) --------------------
        shapenet_ordered_ids = json.load(open("plane_feats/shapenet2spaghetti.json")).keys()
        for id in shapenet_ordered_ids:
            filepaths.append(f"{root}/{id}/models/surface_points.npz")
            
        print("num shapes in entire dataset: {}".format(len(filepaths)))

        self.list_of_points = []
        for filepath in filepaths:
            print("loading npz")
            self.list_of_points.append(np.load(filepath)['surface_points'] )

        if self.uniform:
            self.save_path = os.path.join(root, 'modelnet%d_%s_%dpts_fps.dat' % (self.num_category, split, self.npoints))
        else:
            self.save_path = os.path.join(root, 'modelnet%d_%s_%dpts.dat' % (self.num_category, split, self.npoints))


    def __len__(self):
        return len(self.list_of_points)

    def _get_item(self, index):
        assert not self.use_normals
        if self.process_data:
            point_set, label = self.list_of_points[index], self.list_of_labels[index]
        else:
            # fn = self.datapath[index]
            # cls = self.classes[self.datapath[index][0]]
            label = np.array([-1]).astype(np.int32)  # each line of txt has pos + normal
            point_set = self.list_of_points[index].astype(np.float32) # np.loadtxt(fn[1], delimiter=',').astype(np.float32) # TODO: use shapenet npz (see old pointnet 2 repo)

            if self.uniform:
                point_set = farthest_point_sample(point_set, self.npoints)
            else:
                point_set = point_set[0:self.npoints, :]
                
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        if not self.use_normals:
            point_set = point_set[:, 0:3]

        return point_set, label[0]

    def __getitem__(self, index):
        return self._get_item(index)


if __name__ == '__main__':
    import torch

    # data = ModelNetDataLoader('/data/modelnet40_normal_resampled/', split='train')
    data = ModelNetDataLoader('../shapenet_all_planes', split='test')
    DataLoader = torch.utils.data.DataLoader(data, batch_size=12, shuffle=True)
    for point, label in DataLoader:
        print(point.shape)
        print(label.shape)
