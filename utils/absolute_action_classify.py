"""Actions was classified to different primitive actions"""

import os
import h5py
import argparse
from pathlib import Path
import numpy as np
from collections import Counter

#from lotus.libero import benchmark
#from lotus.libero.envs import OffScreenRenderEnv
from torch.utils.data import DataLoader
#from datasets.waypoint_maniskill_dataset import ManiskillWaypointDataset

"""
PRIMITIVES = [
    "idle",
    "push",
    "pull",
    "grasp",
    "release",
    "rotation",
    "push+rotation",
    "pull+rotation",
]

def classify_primitive(prev_action, curr_action,
                    trans_thresh=1e-3, rot_thresh=1e-2, gripper_thresh=1e-3):
    '''
    Classify action vector into primitive.
    Assumed action format: [dx, dy, dz, rx, ry, rz, gripper]
    - dx,dy,dz : translation deltas
    - rx,ry,rz : rotation deltas
    - gripper  : scalar, >0 means open, <=0 means closed
    '''

    if prev_action is None:
        return "idle"

    # Extract translation, rotation, and gripper values
    delta_trans = curr_action[:3]
    delta_rot = curr_action[3:6]
    gripper_prev = prev_action[-1]
    gripper_curr = curr_action[-1]

    # --- Check idle (very small movement and no gripper change) ---
    if (np.linalg.norm(delta_trans) < trans_thresh and
        np.linalg.norm(delta_rot) < rot_thresh and
        abs(gripper_curr - gripper_prev) < gripper_thresh):
        return "idle"

    # --- Gripper primitives ---
    if gripper_prev > 0 and gripper_curr <= 0:
        return "grasp"
    if gripper_prev <= 0 and gripper_curr > 0:
        return "release"

    # --- Hybrid and pure motion primitives ---
    trans_mag = np.linalg.norm(delta_trans)
    rot_mag = np.linalg.norm(delta_rot)

    # If both translation and rotation are significant -> hybrid
    if trans_mag >= trans_thresh and rot_mag >= rot_thresh:
        if delta_trans[0] > 0:
            return "push+rotation"
        elif delta_trans[0] < 0:
            return "pull+rotation"

    # If mostly rotation -> rotation
    if rot_mag > trans_mag:
        return "rotation"

    # If mostly translation → push or pull
    if delta_trans[0] > 0:
        return "push"
    elif delta_trans[0] < 0:
        return "pull"

    # Fallback
    return "idle"
    

def classify_batch(prev_actions, curr_actions):
    primitives = []
    for i in range(len(prev_actions)):
        primitive = classify_primitive(prev_actions[i], curr_actions[i])
        primitives.append(primitive)
        #print(f"Step {i}: action={curr_actions[i]}, primitive={primitive}")

    return primitives

"""

# Primitive labels
PRIMITIVES = [
    0,  # idle
    1,  # push
    2,  # pull
    3,  # grasp
    4,  # release
    5,  # rotation
    6,  # push+rotation
    7,  # pull+rotation
]


def classify_primitive(prev_action, curr_action,
                       trans_thresh=2e-3, rot_thresh=2e-2, gripper_thresh=2e-3):
    '''
    Classify action vector into primitive ID (0–7).
    Action format: [dx, dy, dz, rx, ry, rz, gripper]
    '''

    if prev_action is None:
        return 0  # idle

    delta_trans = curr_action[:3]
    delta_rot = curr_action[3:6]
    gripper_prev = prev_action[-1]
    gripper_curr = curr_action[-1]

    # Idle
    if (np.linalg.norm(delta_trans) < trans_thresh and
        np.linalg.norm(delta_rot) < rot_thresh and
        abs(gripper_curr - gripper_prev) < gripper_thresh):
        return 0

    # Grasp
    if gripper_prev > 0 and gripper_curr <= 0:
        return 3

    # Release
    if gripper_prev <= 0 and gripper_curr > 0:
        return 4

    # Magnitudes
    trans_mag = np.linalg.norm(delta_trans)
    rot_mag = np.linalg.norm(delta_rot)

    '''
    # Hybrid
    if trans_mag >= trans_thresh and rot_mag >= rot_thresh:
        if delta_trans[0] > 0:
            return 6  # push+rotation
        elif delta_trans[0] < 0:
            return 7  # pull+rotation
    '''
    # Hybrid
    if trans_mag >= trans_thresh and rot_mag >= rot_thresh:
        return 5  # translation+rotation
    '''
    # Rotation
    if rot_mag > trans_mag:
        return 5

    # Push / Pull
    if delta_trans[0] > 0:
        return 1
    elif delta_trans[0] < 0:
        return 2
    '''
    # Rotation
    if rot_mag > trans_mag:
        return 2

    return 1


def classify_batch(prev_actions, curr_actions):
    '''Classify batch of actions into primitive IDs'''
    return [classify_primitive(prev_actions[i], curr_actions[i]) for i in range(len(prev_actions))]



'''
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='libero_90', help="'libero_object','libero_spatial','libero_goal', 'libero_10', 'libero_90'")
    parser.add_argument('--save_path', type=str, default='labels.npy', help='Output path for labels')
    parser.add_argument('--batch_size', type=int, default=64,help='batch size for training')
    cfg = parser.parse_args()

    init_action = np.array([[0,0,0,0,0,0,1]])
    
    exp_dir = os.path.join("./LIBERO/libero/datasets/", cfg.dataset_name)
    dataset_name_list = [str(f) for f in Path(exp_dir).glob("*.hdf5")]
    for data_name in dataset_name_list:
        with h5py.File(data_name, "a") as f:  # "a" = append mode
            demos = list(f['data'].keys())
            for demo_key in demos:
                actions = np.array(f['data'][demo_key]['actions'])
                #actions = np.array(h5py_file['data'][key]['actions'])
                print("actions.dim : ", actions.shape)
                print("actions.max : ", actions.max())
                print("actions.min : ", actions.min())
                prev_actions = np.concatenate([init_action, actions[:-1, :]], axis=0)
                primitives = classify_batch(prev_actions, actions)
                # Save to the same hdf5 file under demo
                if 'primitives' in f['data'][demo_key]:
                    del f['data'][demo_key]['primitives']  # overwrite if exists
                f['data'][demo_key].create_dataset('primitives', data=primitives)

                print(f"  {demo_key}: saved {len(primitives)} primitives, counts={dict(Counter(primitives))}")

'''

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='libero_10', 
                        help="'libero_object','libero_spatial','libero_goal', 'libero_10', 'libero_90'")
    parser.add_argument('--output_root', type=str, default='./datas/no_pushpull/', 
                        help='Directory to save processed dataset (same structure as input)')
    cfg = parser.parse_args()

    input_root = os.path.join("./LIBERO/libero/datasets/", cfg.dataset_name)
    output_root = os.path.join(cfg.output_root, cfg.dataset_name)
    os.makedirs(output_root, exist_ok=True)

    dataset_files = list(Path(input_root).glob("*.hdf5"))
    init_action = np.array([[0, 0, 0, 0, 0, 0, 1]])

    print(f"Input dir:  {input_root}")
    print(f"Output dir: {output_root}")
    print(f"Found {len(dataset_files)} files")

    for data_path in dataset_files:
        print(f"\nProcessing {data_path.name} ...")

        out_path = Path(output_root) / data_path.name

        with h5py.File(data_path, "r") as fin:
            with h5py.File(out_path, "w") as fout:
                fin.copy("data", fout)
                demos = list(fin["data"].keys())

                for demo_key in demos:
                    actions = np.array(fin["data"][demo_key]["actions"])
                    prev_actions = np.concatenate([init_action, actions[:-1, :]], axis=0)
                    primitives = classify_batch(prev_actions, actions)

                    if "primitives" in fout["data"][demo_key]:
                        del fout["data"][demo_key]["primitives"]

                    fout["data"][demo_key].create_dataset("primitives", data=primitives)

                    print(f"  {demo_key}: saved {len(primitives)} primitives, counts={dict(Counter(primitives))}")
                    
                    print(f"fout[demo_key][primitives].shape: ", fout["data"][demo_key]["primitives"].shape)
                    print(f"fout[demo_key][actions].shape: ", fout["data"][demo_key]["actions"].shape)
                    print(f"fout[demo_key]['obs']['agentview_rgb'].shape: ", fout["data"][demo_key]['obs']['agentview_rgb'].shape)

        print(f"✅ Saved processed file to {out_path}")

if __name__ == "__main__":
    main()

