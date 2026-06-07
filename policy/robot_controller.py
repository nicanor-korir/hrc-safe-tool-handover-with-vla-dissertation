"""
Low-level robot controller for the Franka Emika Panda in PyBullet.

Responsibilities:
  - Move the gripper to a target pose using PyBullet's inverse kinematics
  - Open and close the parallel-jaw gripper
  - Report the current gripper pose

This module is intentionally task-agnostic. It does not know what a
handover is or what a tool is. The handover policy in handover_policy.py
calls these primitives to construct the full task behaviour.
"""

import numpy as np
import pybullet as p

# Franka Panda joint layout in the standard PyBullet URDF
ARM_JOINT_INDICES = [0, 1, 2, 3, 4, 5, 6]
FINGER_JOINT_INDICES = [9, 10]
END_EFFECTOR_LINK_INDEX = 11

# Resting joint configuration for the arm (radians)
# This pose puts the arm in a comfortable forward-leaning posture
HOME_JOINT_POSITIONS = [0.0, -0.5, 0.0, -2.2, 0.0, 1.6, 0.785]

# Gripper open and closed widths (each finger moves this far)
GRIPPER_OPEN_WIDTH = 0.04   # 4cm per finger -> 8cm total opening
GRIPPER_CLOSED_WIDTH = 0.0

# Force applied by motors when commanding positions
ARM_MOTOR_FORCE = 200.0
FINGER_MOTOR_FORCE = 20.0

# How close the gripper must be to a target pose to count as "arrived"
POSITION_TOLERANCE = 0.01     # 1 cm
ORIENTATION_TOLERANCE = 0.1   # quaternion difference threshold


class RobotController:
    """
    Wraps the Franka Panda for high-level pose commands.

    Usage:
        controller = RobotController(robot_id)
        controller.reset_to_home()
        controller.move_gripper_to_pose(target_pos, target_orn)
        controller.close_gripper()
    """

    def __init__(self, robot_id):
        """
        Parameters
        ----------
        robot_id : int
            PyBullet body ID for the Franka Panda, returned by build_workspace.
        """
        self.robot_id = robot_id
        self.reset_to_home()

    def reset_to_home(self):
        """
        Instantly snap the arm to the home pose. Used at the start
        of every trial to ensure a deterministic starting state.
        """
        for joint_idx, target_pos in zip(ARM_JOINT_INDICES, HOME_JOINT_POSITIONS):
            p.resetJointState(self.robot_id, joint_idx, target_pos)
        # Start with the gripper open
        for finger_idx in FINGER_JOINT_INDICES:
            p.resetJointState(self.robot_id, finger_idx, GRIPPER_OPEN_WIDTH)

    def get_gripper_pose(self):
        """
        Return the current world pose of the gripper centre.

        Returns
        -------
        position : numpy array of 3 floats
        orientation : numpy array of 4 floats (quaternion x, y, z, w)
        """
        state = p.getLinkState(self.robot_id, END_EFFECTOR_LINK_INDEX)
        position = np.array(state[0])
        orientation = np.array(state[1])
        return position, orientation

    def get_arm_joint_positions(self):
        """Return the seven arm joint angles in radians."""
        return np.array([
            p.getJointState(self.robot_id, idx)[0]
            for idx in ARM_JOINT_INDICES
        ])

    def get_arm_joint_velocities(self):
        """Return the seven arm joint velocities in radians per second."""
        return np.array([
            p.getJointState(self.robot_id, idx)[1]
            for idx in ARM_JOINT_INDICES
        ])

    def command_gripper_to_pose(self, target_position, target_orientation):
        """
        Send a single timestep command moving the gripper toward target.

        Does NOT block or wait. The caller must step the simulation
        and call this repeatedly until the gripper arrives.

        Parameters
        ----------
        target_position : list or array of 3 floats
        target_orientation : list or array of 4 floats (quaternion)
        """
        # Use PyBullet's IK solver to find joint angles for the target pose
        joint_targets = p.calculateInverseKinematics(
            self.robot_id,
            END_EFFECTOR_LINK_INDEX,
            list(target_position),
            list(target_orientation),
            maxNumIterations=50,
            residualThreshold=1e-4,
        )

        # The IK solver returns a value for every controllable joint,
        # including the fingers. We only want the first seven (the arm).
        arm_targets = joint_targets[:7]

        # Command each arm joint motor toward its target
        for joint_idx, target in zip(ARM_JOINT_INDICES, arm_targets):
            p.setJointMotorControl2(
                bodyUniqueId=self.robot_id,
                jointIndex=joint_idx,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target,
                force=ARM_MOTOR_FORCE,
            )

    def move_gripper_to_pose(
        self, target_position, target_orientation, max_steps=480
    ):
        """
        Drive the gripper toward target_pose and wait for arrival.

        Blocks (in the sense of stepping the simulation many times) until
        the gripper is within tolerance or max_steps is exceeded.

        Returns
        -------
        success : bool
            True if the gripper arrived within tolerance.
        steps_taken : int
            Number of simulation steps used.
        """
        for step in range(max_steps):
            self.command_gripper_to_pose(target_position, target_orientation)
            p.stepSimulation()

            current_pos, current_orn = self.get_gripper_pose()
            pos_error = np.linalg.norm(current_pos - np.array(target_position))

            if pos_error < POSITION_TOLERANCE:
                return True, step + 1

        return False, max_steps

    def open_gripper(self, steps=60):
        """
        Open the gripper fingers over the given number of simulation steps.
        """
        for finger_idx in FINGER_JOINT_INDICES:
            p.setJointMotorControl2(
                bodyUniqueId=self.robot_id,
                jointIndex=finger_idx,
                controlMode=p.POSITION_CONTROL,
                targetPosition=GRIPPER_OPEN_WIDTH,
                force=FINGER_MOTOR_FORCE,
            )
        for _ in range(steps):
            p.stepSimulation()

    def close_gripper(self, steps=60, target_width=0.005):
        """
        Close the gripper fingers around an object.

        Parameters
        ----------
        steps : int
            Simulation steps to run while closing.
        target_width : float
            How far each finger should travel inward. A small positive
            value (rather than zero) allows the fingers to press against
            a grasped object rather than colliding with each other.
        """
        for finger_idx in FINGER_JOINT_INDICES:
            p.setJointMotorControl2(
                bodyUniqueId=self.robot_id,
                jointIndex=finger_idx,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target_width,
                force=FINGER_MOTOR_FORCE,
            )
        for _ in range(steps):
            p.stepSimulation()

    def get_gripper_width(self):
        """
        Return the current opening of the gripper, summed across both fingers.
        Useful for detecting whether the gripper is empty or holding something.
        """
        f1 = p.getJointState(self.robot_id, FINGER_JOINT_INDICES[0])[0]
        f2 = p.getJointState(self.robot_id, FINGER_JOINT_INDICES[1])[0]
        return f1 + f2


if __name__ == "__main__":
    # Manual test: build the workspace, move the gripper around a bit
    import time
    from environment.workspace import build_workspace

    ids = build_workspace(gui=True)
    controller = RobotController(ids["robot_id"])

    print("Letting things settle for 1 second...")
    for _ in range(240):
        p.stepSimulation()
        time.sleep(1.0 / 240.0)

    # Read the starting gripper pose
    start_pos, start_orn = controller.get_gripper_pose()
    print(f"Starting gripper position: {start_pos}")

    # Move to a point above the screwdriver
    print("\nMoving gripper above screwdriver...")
    target_pos = [0.30, 0.10, 0.95]
    # Gripper pointing straight down (rotated 180 degrees around X axis)
    target_orn = p.getQuaternionFromEuler([np.pi, 0, 0])
    success, steps = controller.move_gripper_to_pose(target_pos, target_orn)
    print(f"  Success: {success}, steps used: {steps}")

    end_pos, _ = controller.get_gripper_pose()
    print(f"  Final position: {end_pos}")
    print(f"  Error: {np.linalg.norm(end_pos - np.array(target_pos)):.4f} m")

    # Demonstrate open and close
    print("\nClosing gripper...")
    controller.close_gripper(steps=120)
    print(f"  Gripper width after closing: {controller.get_gripper_width():.4f}")

    print("Opening gripper...")
    controller.open_gripper(steps=120)
    print(f"  Gripper width after opening: {controller.get_gripper_width():.4f}")

    # Move to a point above the handover target
    print("\nMoving gripper above handover zone...")
    target_pos = [0.55, 0.0, 0.95]
    success, steps = controller.move_gripper_to_pose(target_pos, target_orn)
    print(f"  Success: {success}, steps used: {steps}")

    print("\nHolding for 3 seconds...")
    for _ in range(240 * 3):
        p.stepSimulation()
        time.sleep(1.0 / 240.0)

    p.disconnect()
    print("Done.")
