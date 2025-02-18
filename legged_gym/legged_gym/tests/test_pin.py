# %%
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
import os
import numpy as np

urdf_path = "/home/hightorque/Desktop/DreamWaQ-Pi-Sim2sim/legged_gym/resources/robots/hi_cl_23_240925/urdf/hi_cl_23_240925_rl.urdf"
model_path = "/home/hightorque/Desktop/DreamWaQ-Pi-Sim2sim/legged_gym/resources/robots/hi_cl_23_240925/meshes"

robot = RobotWrapper.BuildFromURDF(urdf_path, model_path, pin.JointModelFreeFlyer())
model = robot.model
data = model.createData()
pin.computeTotalMass(model, data)

# %%
robot = RobotWrapper.BuildFromURDF(urdf_path, [model_path])
model = robot.model
data = model.createData()
q = pin.neutral(robot.model)
print(q)

# %%
def inverse_kinematics(model, data, end_effector_id, target_transform, q_init, max_iter=1000, eps=1e-4):
    q = q_init.copy()
    for i in range(max_iter):
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacement(model, data, end_effector_id)
        current_transform = data.oMf[end_effector_id]

        error_se3 = pin.log6(current_transform.inverse() * target_transform)
        error = error_se3.vector

        if np.linalg.norm(error) < eps:
            return q, True

        J = pin.computeFrameJacobian(model, data, q, end_effector_id, pin.ReferenceFrame.LOCAL)
        J_pseudo_inv = np.linalg.pinv(J)
        dq = J_pseudo_inv.dot(error)
        q = pin.integrate(model, q, dq)
    return q, False

def compute_joint_positions(model, data, joint_names, q=None):
    if q is None:
        q = pin.neutral(model)
    
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    
    positions = {}
    for joint_name in joint_names:
        try:
            frame_id = model.getFrameId(joint_name)
            frame_placement = data.oMf[frame_id]
            position = frame_placement.translation
            positions[joint_name] = position
        except pin.FrameDoesNotExist:
            print(f"Frame '{joint_name}' does not exist in the model.")
            exit(1)
    
    return positions

# %%
R_left_tar = pin.utils.rpyToMatrix(0.0, 0.0, 0.0)
R_right_tar = pin.utils.rpyToMatrix(0.0, 0.0, 0.0)

p_left_tar  = np.array([0.075,  0.08, -0.60])
p_right_tar = np.array([0.075, -0.08, -0.60])

target_transform_left = pin.SE3(R_left_tar, p_left_tar)
target_transform_right = pin.SE3(R_right_tar, p_right_tar)

q_init = pin.neutral(model)

# %%
left_end_effector_frame = "l_ankle_roll_link"
left_end_effector_id = model.getFrameId(left_end_effector_frame)
q_ik_left, success_left = inverse_kinematics(model, data, left_end_effector_id, target_transform_left, q_init)

right_end_effector_frame = "r_ankle_roll_link"
right_end_effector_id = model.getFrameId(right_end_effector_frame)
q_ik_right, success_right = inverse_kinematics(model, data, right_end_effector_id, target_transform_right, q_init)

print(f"l_ankle_roll_link position {compute_joint_positions(model, data, ['l_ankle_roll_link'])}")
print(f"r_ankle_roll_link position {compute_joint_positions(model, data, ['r_ankle_roll_link'])}")


# %%
q = pin.neutral(robot.model)
q[:6] = q_ik_left[:6]  # 左腿
q[6:12] = q_ik_right[6:12]  # 右腿

pin.forwardKinematics(model, data, q)
pin.updateFramePlacements(model, data)

# %%
if success_left:
    left_ankle_index = model.getFrameId("l_ankle_roll_link")
    left_ankle_position = data.oMf[left_ankle_index].translation
    print("left leg:",q_ik_left)
    print("tar left ankle roll:", p_left_tar)
    print("calc left ankle roll:", left_ankle_position)
else:
    print("Inverse kinematics for left leg did not converge.")

# %%
if success_right:
    right_ankle_index = model.getFrameId("r_ankle_roll_link")
    right_ankle_position = data.oMf[right_ankle_index].translation
    print("right leg:",q_ik_right)
    print("tar right ankle roll:", p_right_tar)
    print("calc right ankle roll:", right_ankle_position)
else:
    print("Inverse kinematics for right leg did not converge.")

# %%
from pinocchio.visualize import MeshcatVisualizer
if success_left and success_right:
    viz = MeshcatVisualizer(robot.model, robot.collision_model, robot.visual_model)
    viz.initViewer(open=True)
    viz.loadViewerModel()
    for name in enumerate(model.names):
        print(f"Joint Name: {name}")
    viz.display(q)


