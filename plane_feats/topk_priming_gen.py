import numpy as np
import torch
import json
k=4

dists = np.load("pairwise_dists.npy") #[n,n]. row i = dists vs. i-th plane
print("orig: ", dists.shape)
# # remove diagonal  [n, n-1]
# dists = dists[~np.eye(dists.shape[0],dtype=bool)].reshape(dists.shape[0],-1)
print("after: ", dists.shape)
# get top k indices
topk_indices = np.argsort(dists, axis=1)[:,:k+1]
print("dists: ", np.take_along_axis(dists, topk_indices, axis=1))
# get top k corresp ids
shape_ids = list(json.load(open("shapenet2spaghetti.json")).keys())
shape_ids = np.array(shape_ids)
obj = {}
for i, curr_row_indices in enumerate(topk_indices):
    topk_ids = shape_ids[curr_row_indices].tolist()
    # ignore closest match (always same shape as GT)
    topk_ids = topk_ids[1:]
    GT = shape_ids[i]
    # print("GT: ", shape_ids[i])
    # print(topk_ids)
    obj[GT] = [topk_ids]

with open("pointnet_priming_tuples.json", "w") as out:
    out.write(json.dumps(obj, indent=4))