import os
import h5py
import argparse
from pathlib import Path
import numpy as np
from collections import Counter
from typing import List

# Primitive labels (ints)
PRIMITIVE_NAMES = [
    "idle",          # 0
    "push",          # 1
    "pull",          # 2
    "grasp",         # 3
    "release",       # 4
    "rotation",      # 5
    "push+rotation", # 6
    "pull+rotation", # 7
]

def classify_primitive(action: np.ndarray,
                       trans_thresh: float = 1e-3,
                       rot_thresh: float = 1e-2,
                       gripper_thresh: float = 1e-3,
                       forward_axis: int = 0) -> int:
    """
    Classify a single *relative* action into primitive ID (0-7).
    Action format: [dx, dy, dz, rx, ry, rz, gripper]
      - (dx,dy,dz): relative translation increments in current end-effector frame
      - (rx,ry,rz): relative rotation increments (e.g., Euler increments)
      - gripper   : scalar command (>0 open, <=0 close)
    Returns:
      int in {0,...,7} corresponding to PRIMITIVE_NAMES.
    """

    a = np.asarray(action, dtype=float)
    if a.size < 7:
        raise ValueError("action must have length >= 7: [dx,dy,dz,rx,ry,rz,gripper]")

    delta_trans = a[:3]
    delta_rot = a[3:6]
    gripper_cmd = a[6]

    trans_mag = float(np.linalg.norm(delta_trans))
    rot_mag = float(np.linalg.norm(delta_rot))

    # 1) Idle: negligible motion and negligible gripper command
    if (trans_mag < trans_thresh and rot_mag < rot_thresh and abs(gripper_cmd) < gripper_thresh):
        return 0  # idle

    # 2) Gripper intentions (use current gripper command only)
    #    we interpret the gripper value
    #    as intention: <=0 -> close (grasp intention), >0 -> open (release intention).
    if abs(gripper_cmd) >= gripper_thresh:
        if gripper_cmd <= 0:
            return 3  # grasp (close intent)
        else:
            return 4  # release (open intent)

    # 3) Hybrid (translation + rotation)
    if trans_mag >= trans_thresh and rot_mag >= rot_thresh:
        # determine forward sign along chosen axis
        if delta_trans[forward_axis] > 0:
            return 6  # push + rotation
        elif delta_trans[forward_axis] < 0:
            return 7  # pull + rotation

    # 4) Pure rotation (rotation dominates)
    if rot_mag > trans_mag:
        return 5  # rotation

    # 5) Pure translation -> push / pull by forward axis sign
    if delta_trans[forward_axis] > 0:
        return 1  # push
    elif delta_trans[forward_axis] < 0:
        return 2  # pull

    # Fallback
    return 0  # idle


def classify_batch(actions: np.ndarray,
                   trans_thresh: float = 2e-1,
                   rot_thresh: float = 5e-2,
                   gripper_thresh: float = 2e-1,
                   forward_axis: int = 0) -> List[int]:
    """
    Classify a batch of relative actions into primitive IDs.
    actions: array-like of shape (B, 7)
    Returns: list of ints length B
    """
    arr = np.asarray(actions, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] < 7:
        raise ValueError("actions must have shape (B, >=7)")

    results = []
    for i in range(arr.shape[0]):
        results.append(classify_primitive(arr[i],
                                          trans_thresh=trans_thresh,
                                          rot_thresh=rot_thresh,
                                          gripper_thresh=gripper_thresh,
                                          forward_axis=forward_axis))
    return results
    
    
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='libero_object', help="'libero_object','libero_spactial','libero_goal', 'libero_10', 'libero_90'")
    parser.add_argument('--save_path', type=str, default='labels.npy', help='Output path for labels')
    cfg = parser.parse_args()

    exp_dir = os.path.join("./LIBERO/libero/datasets/", cfg.dataset_name)
    dataset_name_list = [str(f) for f in Path(exp_dir).glob("*.hdf5")]
    for data_name in dataset_name_list:
        with h5py.File(data_name, "a") as f:  # "a" = append mode
            demos = list(f['data'].keys())
            for demo_key in demos:
                actions = np.array(f['data'][demo_key]['actions'])
                #'''
                actio = actions[:,6]
                print("actio : ", actio)
                #print("actions.max : ", actio.max())
                #print("actions.min : ", actio.min())
                #'''
                primitives = classify_batch(actions)
                # Save to the same hdf5 file under demo
                if 'primitives' in f['data'][demo_key]:
                    del f['data'][demo_key]['primitives']  # overwrite if exists
                f['data'][demo_key].create_dataset('primitives', data=primitives)

                print(f"  {demo_key}: saved {len(primitives)} primitives, counts={dict(Counter(primitives))}")

        #break


if __name__ == "__main__":
    main()

