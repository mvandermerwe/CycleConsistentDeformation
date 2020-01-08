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
import trainer
from auxiliary.my_utils import *
from functools import reduce
from auxiliary.model_atlasnet import *
import miou_shape
import useful_losses as loss
import my_utils
from my_utils import Max_k, Min_k
import ICP
import tqdm
from auxiliary.visualization import visualize_points, visualize_points_overlay
from save_mesh_from_points_and_labels import *
import pdb

opt = argument_parser.parser()

trainer = trainer.Trainer(opt, test=True)
# trainer.build_dataset_train_for_matching()
trainer.build_dataset_train()
# trainer.build_dataset_test_for_matching()
trainer.build_dataset_test()
trainer.build_network()
trainer.network.eval()

target_idx = 0
canon_idx = 0

target_mesh = os.path.join(args.deform_dataset, trainer.dataset_test.models[target_idx]['model'], 'mesh.obj')
canon_mesh = os.path.join(args.deform_dataset, trainer.dataset_test.models[target_idx]['model'], 'mesh.obj')

# =============DEFINE CHAMFER LOSS======================================== #
import extension.dist_chamfer_idx as ext

distChamfer = ext.chamferDist()
# ========================================================== #

# ========Loop on test examples======================== #
iterator = trainer.dataloader_test.__iter__()

# P1, _, _, _ = trainer.dataset_train[10]
# P1 = P1[:,:3].unsqueeze(0).contiguous().cuda().float()

with torch.no_grad():
    for i in tqdm.tqdm(range(trainer.len_dataset_test)):
        
        try:
            P1, _, _, _, P2, _, _, _ = trainer.dataset_test[i]
            # P2, _, _, _ = trainer.dataset_test[i]
            P1 = torch.tensor(P1[:, :3]).unsqueeze(0).contiguous().cuda().float()
            P2 = torch.tensor(P2[:, :3]).unsqueeze(0).contiguous().cuda().float()
        except:
            break

        visualize_points(P1[0].cpu().numpy(), bound=0.5, show=True)
        visualize_points(P2[0].cpu().numpy(), bound=0.5, show=True)

        # pdb.set_trace()
        
        P1_2 = loss.forward_chamfer(trainer.network, P1, P2, local_fix=None, distChamfer=distChamfer)

        visualize_points(P1_2[0][0].cpu().numpy(), bound=0.5, show=True)

        visualize_points_overlay([P1_2[0][0].cpu().numpy(), P2[0].cpu().numpy()], bound=0.5, show=True)
