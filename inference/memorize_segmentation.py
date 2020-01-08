from __future__ import print_function
import sys

sys.path.append("./auxiliary/")
sys.path.append("./extension/")
sys.path.append("./trainer/")
sys.path.append("./scripts/")
sys.path.append("./inference/")
sys.path.append("./training/")
sys.path.append("./")
import argument_parser
import dataset_shapenet
import trainer
from auxiliary.my_utils import *
from functools import reduce
from auxiliary.model_atlasnet import *
from auxiliary.visualization import visualize_points
import miou_shape
import useful_losses as loss
import my_utils
from my_utils import Max_k, Min_k
import ICP
import tqdm
from save_mesh_from_points_and_labels import *
import figure_2_3
import pdb
import os

opt = argument_parser.parser()

figures_folder = 'figures/memorize_tests'

# =============DEFINE CHAMFER LOSS======================================== #
# TODO: Fix GPU version.
# import extension.dist_chamfer_idx as ext
# distChamfer = ext.chamferDist()

import extension.chamfer_python as chamf_python
distChamfer = chamf_python.distChamfer
# ========================================================== #

# Load the Cycle Consistency Model:
trainer = trainer.Trainer(opt)
trainer.build_dataset_train_for_matching()
trainer.build_dataset_test_for_matching()
trainer.build_network()
trainer.network.eval()

# Load our memorization dataset.
dataset = dataset_shapenet.ShapeNetSeg(mode="MEMORIZE",
                                       class_choice="Chair",
                                       npoints=opt.number_points,
                                       get_single_shape=True)
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
#                                          shuffle=False, num_workers=int(opt.workers),
#                                          drop_last=True)
len_dataset_test = len(dataset)

def get_dataset_item(i):
    if i >= len_dataset_test or i < 0:
        return None

    elem = dataset[i]

    points = elem[0][:,:3]
    normals = elem[0][:,3:6]
    labels = elem[0][:,6]

    return points, normals, labels

# We can try to transfer from any one point cloud here to the other 3:
with torch.no_grad():
    for i in range(1): # range(len_dataset_test)

        # Load the base element to transfer from.
        src_points, _, src_labels = get_dataset_item(i)
        visualize_points(src_points.cpu().numpy(), bound=1.0, c=src_labels.cpu().numpy(), out_file=os.path.join(figures_folder, 'src_' + str(i) + '.png'))
                         
        for j in range(len_dataset_test):
            if i == j:
                continue

            # Load the other instance.
            tgt_points, _, tgt_labels = get_dataset_item(j)
            visualize_points(tgt_points.cpu().numpy(), bound=1.0, c=tgt_labels.cpu().numpy(), out_file=os.path.join(figures_folder, 'tgt_' + str(j) + '.png'))
            
            # Nearest Neighbor (NN):
            # identity_nn = distChamfer(src_points.unsqueeze(0).cuda(), tgt_points.unsqueeze(0).cuda())
            # identity_pred_labels = src_labels[identity_nn[3].view(-1).cpu().data.long()]
            # visualize_points(tgt_points.cpu().numpy(), bound=1.0, c=identity_pred_labels.cpu().numpy(), out_file=os.path.join(figures_folder, 'tgt_' + str(j) + '_nn.png'))

            # ICP + NN:
            # icp_src_points = ICP.ICP(src_points.unsqueeze(0), tgt_points.unsqueeze(0))
            # visualize_points(icp_src_points.cpu().numpy(), bound=1.0, c=src_labels.cpu().numpy(), out_file=os.path.join(figures_folder, 'tgt_' + str(j) + '_icp_inter.png'))
            # icp_nn = distChamfer(icp_src_points.unsqueeze(0).cuda(), tgt_points.unsqueeze(0).cuda())
            # icp_pred_labels = src_labels[icp_nn[3].view(-1).cpu().data.long()]
            # visualize_points(tgt_points.cpu().numpy(), bound=1.0, c=icp_pred_labels.cpu().numpy(), out_file=os.path.join(figures_folder, 'tgt_' + str(j) + '_icp.png'))

            # Cycle Consistency model:
            cc_forward = loss.forward_chamfer(trainer.network, src_points.unsqueeze(0).cuda(), tgt_points.unsqueeze(0).cuda(), local_fix=None, distChamfer=distChamfer)
            cc_src_points = cc_forward[0].squeeze(0)
            visualize_points(cc_src_points.cpu().numpy(), bound=1.0, c=src_labels.cpu().numpy(), out_file=os.path.join(figures_folder, 'tgt_' + str(j) + '_cc_inter.png'), show=True)
            cc_pred_labels = src_labels[cc_forward[4].view(-1).cpu().data.long()]
            visualize_points(tgt_points.cpu().numpy(), bound=1.0, c=cc_pred_labels, out_file=os.path.join(figures_folder, 'tgt_' + str(j) + '_cc.png'), show=True)
            
            pdb.set_trace()
            print("Wait.")
