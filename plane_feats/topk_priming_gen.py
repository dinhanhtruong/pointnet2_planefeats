import numpy as np
import torch
import json
k=4 #num priming shapes


dists = np.load("pairwise_dists.npy") #[n,n]. row i = dists vs. i-th plane
print("orig: ", dists.shape)
# # remove diagonal  [n, n-1]. NOTE: handled below
# dists = dists[~np.eye(dists.shape[0],dtype=bool)].reshape(dists.shape[0],-1)
print("after: ", dists.shape)
# get top k indices
topk_indices = np.argsort(dists, axis=1)[:,:k+1] # [n, k+1]
print("ordered dists: ", np.take_along_axis(dists, topk_indices, axis=1))
# get top k corresp ids
shape_ids = list(json.load(open("shapenet2spaghetti.json")).keys())
shape_ids = np.array(shape_ids)
print('spaghetti ordered: ', shape_ids)
obj = {}
for i, curr_row_indices in enumerate(topk_indices):
    # assert i == curr_row_indices[0]
    print("top k idx: ", curr_row_indices)
    # print("top k dists: ", dists[i, curr_row_indices])
    topk_ids = shape_ids[curr_row_indices].tolist()
    # ignore closest match (always same shape as GT)
    topk_ids = topk_ids[1:]
    GT = shape_ids[i]
    print("GT: ", shape_ids[i])
    print(topk_ids)
    obj[GT] = [topk_ids]

with open("pointnet_autoencoder_priming_tuples.json", "w") as out:
    out.write(json.dumps(obj, indent=4))