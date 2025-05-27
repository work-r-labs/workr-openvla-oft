"""
deploy.py

Provide a lightweight server/client implementation for deploying OpenVLA models (through the HF AutoClass API) over a
REST API. This script implements *just* the server, with specific dependencies and instructions below.

Note that for the *client*, usage just requires numpy/json-numpy, and requests; example usage below!

Dependencies:
    => Server (runs OpenVLA model on GPU): `pip install uvicorn fastapi json-numpy`
    => Client: `pip install requests json-numpy`

Client (Standalone) Usage (assuming a server running on 0.0.0.0:8000):

```
import requests
import json_numpy
json_numpy.patch()
import numpy as np

action = requests.post(
    "http://0.0.0.0:8000/act",
    json={"image": np.zeros((256, 256, 3), dtype=np.uint8), "instruction": "do something"}
).json()

Note that if your server is not accessible on the open web, you can use ngrok, or forward ports to your client via ssh:
    => `ssh -L 8000:localhost:8000 ssh USER@<SERVER_IP>`
"""

import os.path

# ruff: noqa: E402
import json_numpy

json_numpy.patch()
import json
import logging
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pickle
from experiments.robot.libero.run_libero_eval import GenerateConfig
from experiments.robot.openvla_utils import get_action_head, get_processor, get_proprio_projector, get_vla, get_vla_action
from prismatic.vla.constants import NUM_ACTIONS_CHUNK, PROPRIO_DIM
import cv2

import draccus
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from PIL import Image


# === Server Interface ===
class OpenVLAOFTServer:
    def __init__(self, num_images_in_input):
        # Instantiate config (see class GenerateConfig in experiments/robot/libero/run_libero_eval.py for definitions)
        self.cfg = GenerateConfig(
            pretrained_checkpoint = "moojink/openvla-7b-oft-finetuned-libero-spatial",
            use_l1_regression = True,
            use_diffusion = False,
            use_film = False,
            num_images_in_input = num_images_in_input,
            use_proprio = False,
            load_in_8bit = False,
            load_in_4bit = False,
            center_crop = True,
            num_open_loop_steps = NUM_ACTIONS_CHUNK,
            unnorm_key = "libero_spatial_no_noops",
        )

        # Load OpenVLA-OFT policy and inputs processor
        self.vla = get_vla(self.cfg)
        self.processor = get_processor(self.cfg)

        # Load MLP action head to generate continuous actions (via L1 regression)
        self.action_head = get_action_head(self.cfg, llm_dim=self.vla.llm_dim)

        # Load proprio projector to map proprio to language embedding space
        # self.proprio_projector = get_proprio_projector(self.cfg, llm_dim=self.vla.llm_dim, proprio_dim=PROPRIO_DIM)

    def predict_action(self, payload: Dict[str, Any]) -> str:
        # Convert payload to observation dict
        observation = {
            "full_image": payload["full_image"],
            "task_description": payload["task_description"]
        }
        if "wrist_image" in payload:
            observation["wrist_image"] = payload["wrist_image"]
        if "state" in payload:
            observation["state"] = payload["state"]
        
        # Generate robot action chunk (sequence of future actions)
        actions = get_vla_action(self.cfg, self.vla, self.processor, observation, observation["task_description"], self.action_head)
        print("Generated action chunk:")
        for act in actions:
            print(act)

        return JSONResponse(actions)

    def run(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        self.app = FastAPI()
        self.app.post("/act")(self.predict_action)
        uvicorn.run(self.app, host=host, port=port)

@dataclass
class DeployConfig:
    # fmt: off

    # Server Configuration
    host: str = "0.0.0.0"                                               # Host IP Address
    port: int = 8000                                                    # Host Port

    # fmt: on


@draccus.wrap()
def deploy(cfg: DeployConfig) -> None:
    server = OpenVLAOFTServer(num_images_in_input=2)
    server.run(cfg.host, port=cfg.port)


if __name__ == "__main__":
    # server = OpenVLAOFTServer(num_images_in_input=2)
    # with open("experiments/robot/libero/sample_libero_spatial_observation.pkl", "rb") as file:
    #     payload = pickle.load(file)
    # server.predict_action(payload)

    deploy()
