import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import numpy as np
import pdb
import sys
sys.path.append('../auxiliary')
import visualization as vis

dataset_path = '../data/dataset_shapenet/03001627'

items = [f for f in os.listdir(dataset_path) if '.txt' in f]

for item in items:
    points = np.loadtxt(os.path.join(dataset_path, item))

    image_file = os.path.join(dataset_path, item.replace('.txt', '.png'))

    if os.path.exists(image_file):
        continue
    
    pts = points[:,:3]
    c = points[:,6]

    vis.visualize_points(pts, bound=1.0, c=c, out_file=image_file)
