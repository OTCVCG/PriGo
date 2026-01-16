import torch
import torch.nn.functional as F

PRIMITIVES = [
    "idle",          # 0
    "push",          # 1
    "pull",          # 2
    "grasp",         # 3
    "release",       # 4
    "rotation",      # 5
    "push+rotation", # 6
    "pull+rotation", # 7
]

def classify_actions_torch(prev_actions, curr_actions,
                           trans_thresh=1e-3, rot_thresh=1e-2, gripper_thresh=1e-3):
    """
    Torch batch rule-based primitive classifier.
    
    Args:
        prev_actions (torch.Tensor): [B, 7]
        curr_actions (torch.Tensor): [B, 7]
    
    Returns:
        torch.Tensor: [B] predicted primitive labels (0–7)
    """
    B = curr_actions.shape[0]

    # Extract components
    delta_trans = curr_actions[:, :3]
    delta_rot = curr_actions[:, 3:6]
    gripper_prev = prev_actions[:, -1]
    gripper_curr = curr_actions[:, -1]

    trans_mag = torch.linalg.norm(delta_trans, dim=1)
    rot_mag = torch.linalg.norm(delta_rot, dim=1)

    # Start with idle
    pred_labels = torch.zeros(B, dtype=torch.long, device=curr_actions.device)

    # Grasp
    mask_grasp = (gripper_prev > 0) & (gripper_curr <= 0)
    pred_labels[mask_grasp] = 3

    # Release
    mask_release = (gripper_prev <= 0) & (gripper_curr > 0)
    pred_labels[mask_release] = 4

    # Hybrid: push+rotation
    mask_push_rot = (trans_mag >= trans_thresh) & (rot_mag >= rot_thresh) & (delta_trans[:, 0] > 0)
    pred_labels[mask_push_rot] = 6

    # Hybrid: pull+rotation
    mask_pull_rot = (trans_mag >= trans_thresh) & (rot_mag >= rot_thresh) & (delta_trans[:, 0] < 0)
    pred_labels[mask_pull_rot] = 7

    # Pure rotation (if not already assigned)
    mask_rot = (rot_mag > trans_mag) & (pred_labels == 0)
    pred_labels[mask_rot] = 5

    # Push
    mask_push = (delta_trans[:, 0] > 0) & (pred_labels == 0)
    pred_labels[mask_push] = 1

    # Pull
    mask_pull = (delta_trans[:, 0] < 0) & (pred_labels == 0)
    pred_labels[mask_pull] = 2

    # Idle (already 0 by default)
    mask_idle = (trans_mag < trans_thresh) & (rot_mag < rot_thresh) & \
                (torch.abs(gripper_curr - gripper_prev) < gripper_thresh)
    pred_labels[mask_idle] = 0

    return pred_labels


def classify_actions_soft(prev_actions, curr_actions, temperature=0.1):
    """
    Differentiable primitive classifier (soft version).
    Returns probabilities [B, 8].
    """
    delta_trans = curr_actions[:, :3]
    delta_rot = curr_actions[:, 3:6]
    gripper_prev = prev_actions[:, -1]
    gripper_curr = curr_actions[:, -1]

    trans_mag = torch.linalg.norm(delta_trans, dim=1)
    rot_mag = torch.linalg.norm(delta_rot, dim=1)

    score_idle    = -(trans_mag + rot_mag + torch.abs(gripper_curr - gripper_prev))
    score_push    = delta_trans[:, 0]
    score_pull    = -delta_trans[:, 0]
    score_grasp   = (gripper_prev - gripper_curr)
    score_release = (gripper_curr - gripper_prev)
    score_rot     = (rot_mag - trans_mag)
    score_pushrot = delta_trans[:, 0] + rot_mag
    score_pullrot = -delta_trans[:, 0] + rot_mag

    scores = torch.stack([
        score_idle, score_push, score_pull, score_grasp,
        score_release, score_rot, score_pushrot, score_pullrot
    ], dim=1)  # [B, 8]

    probs = F.softmax(scores / temperature, dim=1)  # [B, 8]
    return probs

def PSLoss(actions, labels):
    """
    Compute batch loss between rule-based predictions and ground-truth labels.
    
    Args:
        prev_actions (torch.Tensor): [B, 7]
        curr_actions (torch.Tensor): [B, 7]
        labels (torch.Tensor): [B], values in [0–7]
    
    Returns:
        torch.Tensor: scalar loss
    """
    init_action = torch.tensor([[0,0,0,0,0,0,1]]).to(actions.device, dtype=actions.dtype)
    prev_actions = torch.cat([init_action, actions[:-1, :]], dim=0)
    #pred_labels = classify_actions_torch(prev_actions, actions)  # [B]
    #pred_logits = F.one_hot(pred_labels, num_classes=len(PRIMITIVES)).float()
    pred_logits = classify_actions_soft(prev_actions, actions)
    return F.cross_entropy(pred_logits, labels)


# Example usage
if __name__ == "__main__":

    curr_actions = torch.tensor([
        [0.1, 0, 0, 0, 0, 0, 1],     # push
        [-0.2, 0, 0, 0, 0, 0, 1],    # pull
        [0, 0, 0, 0.1, 0.2, 0, 1],   # rotation
        [0, 0, 0, 0, 0, 0, -1],      # grasp
    ], dtype=torch.float32)

    labels = torch.tensor([1, 2, 5, 3], dtype=torch.long)

    loss = PSLoss(curr_actions, labels)
    print("Loss:", loss.item())
