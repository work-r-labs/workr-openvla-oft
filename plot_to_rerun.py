import pickle
import numpy as np
import cv2
import rerun as rr
from pathlib import Path

def main():
    # Initialize Rerun
    rr.init("OpenVLA Demo Visualization", spawn=True)
    
    # Load the pickle file to get the original observation data
    with open("experiments/robot/libero/sample_libero_spatial_observation.pkl", "rb") as file:
        observation = pickle.load(file)

    # Print keys
    print(observation.keys())
    
    # Use the original image data from the pickle file instead of the JPG
    print(f"Image shape: {observation['full_image'].shape}, dtype: {observation['full_image'].dtype}, range: [{observation['full_image'].min()}, {observation['full_image'].max()}]")
    # Load the generated actions
    actions_path = "experiments/robot/libero/sample_libero_spatial_actions.txt"
    actions = []
    with open(actions_path, "r") as file:
        for line in file:
            if line.strip():
                action_str = line.strip().strip('[]')
                action = np.array([float(x) for x in action_str.split()])
                actions.append(action)
    
    actions = np.array(actions)
    
    # Get initial state from observation
    initial_state = observation["state"]
    initial_pos = initial_state[:3]  # xyz
    initial_rot = initial_state[3:6]  # axis-angle rotation
    
    # Compute cumulative trajectory
    positions = [initial_pos]
    rotations = [initial_rot]
    current_pos = initial_pos
    current_rot = initial_rot
    
    for action in actions:
        # Update position
        current_pos = current_pos + action[:3]
        positions.append(current_pos)
        
        # Update rotation (simple addition for visualization)
        current_rot = current_rot + action[3:6]
        rotations.append(current_rot)
    
    positions = np.array(positions)
    rotations = np.array(rotations)
    
    # Log trajectory
    rr.log("trajectory/positions", rr.LineStrips3D(
        strips=[positions],
        colors=[[0, 0, 255]]  # Blue
    ))
    
    # Log orientation arrows along trajectory
    for pos, rot in zip(positions, rotations):
        # Create arrow showing orientation
        arrow_length = 0.1
        arrow_end = pos + arrow_length * rot / np.linalg.norm(rot)
        rr.log("trajectory/orientation", rr.Arrows3D(
            origins=[pos],
            vectors=[arrow_end - pos],
            colors=[[0, 255, 0]]  # Green
        ))
    
    # Log images
    rr.log("full_image", rr.Image(observation["full_image"]))
    rr.log("wrist_image", rr.Image(observation["wrist_image"]))
    
    print(f"Trajectory length: {len(positions)} points")
    print(f"Position range: {np.ptp(positions, axis=0)}")

if __name__ == "__main__":
    main()
