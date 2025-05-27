import pickle
import requests
import json_numpy
import numpy as np

# Patch json to handle numpy arrays
json_numpy.patch()

def test_server_client():
    """Test the OpenVLA-OFT server by sending the same observation from demo.py"""
    
    # Server configuration
    SERVER_URL = "http://localhost:8000"
    
    print("Loading sample observation from demo.py...")
    
    # Load sample observation (same as demo.py)
    with open("experiments/robot/libero/sample_libero_spatial_observation.pkl", "rb") as file:
        observation = pickle.load(file)
    
    print(f"Observation: {observation}")
    
    # Extract components
    payload = {
        "full_image": observation["full_image"],
        "task_description": observation["task_description"]
    }
    if "wrist_image" in observation:
        payload["wrist_image"] = observation["wrist_image"]
    if "state" in observation:
        payload["state"] = observation["state"]

    print(f"Payload: {payload}")
    
    # Test server connection
    print(f"\nTesting server connection to {SERVER_URL}...")
    response = requests.post(f"{SERVER_URL}/act", json=payload, timeout=30)
    print(f"Response: {response.json()}")

def main():
    test_server_client()

if __name__ == "__main__":
    main()