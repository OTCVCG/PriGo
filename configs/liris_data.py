import numpy as np
import pinocchio as pin
from datasets import load_dataset


ds = load_dataset("lirislab/franka_stash_plate", split="train")
model, collision_model, visual_model = pin.buildModelsFromUrdf("./configs/panda_v2.urdf")
data = model.createData()
ee_frame = model.getFrameId("panda_hand")


def joint2task(q, dq):
    # q: (n,)
    # dq:  (n,)
    pin.framesForwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    # 
    J = pin.computeFrameJacobian(model, data, q, ee_frame, pin.ReferenceFrame.LOCAL)  # 6×n
    # v = J * dq
    v = J.dot(dq)
    return v  # shape (6,) 

for entry in ds:
    q = np.array(entry["observation.state"])
    dq = np.array(entry["action"])
    v_ee = joint2task(q, dq)

    print("Joint dq:", dq, "-> End-effector v:", v_ee)
    break
