#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH -o ddn_test.out
#SBATCH --nodelist=leto50

source $HOME/.bashrc

module load cuda/10.0

# 1. Load your environment
conda activate sensei

# 2. Run
python ./training/train_shapenet.py --ddn_config ../occupancy_flow/configs/representation_power/cycle_match.yaml --logdir test_ddn_memorize_sym --nepoch 1000 --batch_size 8 --chamfer_loss_type SYM
python ./training/train_shapenet.py --ddn_config ../occupancy_flow/configs/representation_power/cycle_match.yaml --logdir test_ddn_memorize_assym1 --nepoch 1000 --batch_size 8 --chamfer_loss_type ASSYM_1
python ./training/train_shapenet.py --ddn_config ../occupancy_flow/configs/representation_power/cycle_match.yaml --logdir test_ddn_memorize_assym2 --nepoch 1000 --batch_size 8 --chamfer_loss_type ASSYM_2
