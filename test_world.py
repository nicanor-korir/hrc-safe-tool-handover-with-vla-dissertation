"""
Smoke test for the simulation environment.
Opens a PyBullet window, loads the ground and a Franka Panda,
and runs the physics for 10 seconds so you can see it.
"""

import pybullet as p
import pybullet_data
import time

# Connect to PyBullet with a GUI window (use p.DIRECT for headless mode later)
physics_client = p.connect(p.GUI)

# Tell PyBullet where to find the built-in URDF files
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Set gravity
p.setGravity(0, 0, -9.81)

# Load a flat ground plane
plane_id = p.loadURDF("plane.urdf")

# Load the Franka Emika Panda at the origin, base fixed in place
robot_id = p.loadURDF(
    "franka_panda/panda.urdf",
    basePosition=[0, 0, 0],
    useFixedBase=True
)

# Print some basic info about the robot so we know it loaded
num_joints = p.getNumJoints(robot_id)
print(f"Loaded Panda with {num_joints} joints")

# Step the physics simulation for 10 seconds at 240 Hz
# (PyBullet's default timestep is 1/240 seconds)
for _ in range(240 * 10):
    p.stepSimulation()
    time.sleep(1.0 / 240.0)

# Clean disconnect
p.disconnect()
print("Done.")
