"""
Workspace builder for the handover simulation.

Builds the physical setup described in Section 3.2 of the methodology:
- a flat table one metre wide
- a Franka Emika Panda mounted at one end
- three tools placed at fixed starting positions
  (screwdriver, rubber mallet, pencil)
- a target zone where the human hand will appear

This module does not run the simulation. It only constructs the world.
Other modules (policy, experiment) drive the simulation forward.
"""

import pybullet as p
import pybullet_data


# Fixed positions and dimensions, in metres
TABLE_HEIGHT = 0.65
TABLE_LENGTH = 1.20
TABLE_WIDTH = 0.60

ROBOT_BASE_POSITION = [0.0, -TABLE_WIDTH / 2 + 0.05, TABLE_HEIGHT]

# Tool starting positions on the table surface
SCREWDRIVER_START = [0.30, 0.10, TABLE_HEIGHT + 0.02]
MALLET_START = [0.30, 0.00, TABLE_HEIGHT + 0.04]
PENCIL_START = [0.30, -0.10, TABLE_HEIGHT + 0.01]

# Where the human hand will reach to during handover trials
HANDOVER_TARGET = [0.55, 0.0, TABLE_HEIGHT + 0.25]


def build_workspace(gui=True):
    """
    Construct the full workspace in PyBullet.
    Returns a dict of object IDs so other modules can reference them.

    Parameters
    ----------
    gui : bool
        If True, opens a PyBullet GUI window. If False, runs headless
        (used later for batch experiments).
    """
    # Connect to PyBullet
    if gui:
        physics_client = p.connect(p.GUI)
        # Set a sensible default camera angle for the workspace
        p.resetDebugVisualizerCamera(
            cameraDistance=1.6,
            cameraYaw=50,
            cameraPitch=-30,
            cameraTargetPosition=[0.3, 0.0, TABLE_HEIGHT],
        )
    else:
        physics_client = p.connect(p.DIRECT)

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)

    # Ground plane
    plane_id = p.loadURDF("plane.urdf")

    # Build the table from a simple box collision shape
    # Half-extents are half the length, half the width, half the height
    table_collision = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=[TABLE_LENGTH / 2, TABLE_WIDTH / 2, TABLE_HEIGHT / 2],
    )
    table_visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[TABLE_LENGTH / 2, TABLE_WIDTH / 2, TABLE_HEIGHT / 2],
        rgbaColor=[0.7, 0.6, 0.5, 1.0],
    )
    table_id = p.createMultiBody(
        baseMass=0,  # mass zero means fixed in place
        baseCollisionShapeIndex=table_collision,
        baseVisualShapeIndex=table_visual,
        basePosition=[TABLE_LENGTH / 2 - 0.1, 0.0, TABLE_HEIGHT / 2],
    )

    # Mount the Franka Panda at one short end of the table
    robot_id = p.loadURDF(
        "franka_panda/panda.urdf",
        basePosition=ROBOT_BASE_POSITION,
        useFixedBase=True,
    )

    # Tools as simple primitive shapes for now.
    # A more detailed mesh is not needed for the ablation, since the
    # monitoring streams operate on policy state and trajectory data,
    # not on visual fidelity.
    screwdriver_id = _make_tool(
        SCREWDRIVER_START,
        size=[0.075, 0.008, 0.008],  # 15cm long, thin
        colour=[0.9, 0.2, 0.2, 1.0],  # red
    )
    mallet_id = _make_tool(
        MALLET_START,
        size=[0.05, 0.025, 0.04],  # heavier, chunkier
        colour=[0.3, 0.3, 0.3, 1.0],  # dark grey
        mass=1.2,
    )
    pencil_id = _make_tool(
        PENCIL_START,
        size=[0.085, 0.0035, 0.0035],  # 17cm long, very thin
        colour=[1.0, 0.85, 0.2, 1.0],  # yellow
        mass=0.01,
    )

    # Visual marker showing where the human hand will appear
    # No collision shape, so it cannot interact with anything
    target_visual = p.createVisualShape(
        p.GEOM_SPHERE,
        radius=0.04,
        rgbaColor=[0.2, 0.8, 0.2, 0.3],  # translucent green
    )
    target_id = p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=target_visual,
        basePosition=HANDOVER_TARGET,
    )

    return {
        "physics_client": physics_client,
        "plane_id": plane_id,
        "table_id": table_id,
        "robot_id": robot_id,
        "screwdriver_id": screwdriver_id,
        "mallet_id": mallet_id,
        "pencil_id": pencil_id,
        "target_id": target_id,
    }


def _make_tool(position, size, colour, mass=0.1):
    """Helper to build a tool as a coloured box with collision."""
    collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=size)
    visual = p.createVisualShape(
        p.GEOM_BOX, halfExtents=size, rgbaColor=colour
    )
    return p.createMultiBody(
        baseMass=mass,
        baseCollisionShapeIndex=collision,
        baseVisualShapeIndex=visual,
        basePosition=position,
    )


if __name__ == "__main__":
    # Running this file directly opens the workspace for visual inspection
    import time

    ids = build_workspace(gui=True)
    print("Workspace built. Object IDs:")
    for name, obj_id in ids.items():
        if name != "physics_client":
            print(f"  {name}: {obj_id}")

    print("\nKeeping the window open for 30 seconds. Use mouse to orbit.")
    for _ in range(240 * 30):
        p.stepSimulation()
        time.sleep(1.0 / 240.0)

    p.disconnect()
    print("Done.")
