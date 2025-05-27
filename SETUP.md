# Setup Instructions

## Set Up Conda Environment

```bash
# Create and activate conda environment
conda create -n openvla-oft python=3.10 -y
conda activate openvla-oft

# Install PyTorch
# Use a command specific to your machine: https://pytorch.org/get-started/locally/
pip3 install torch torchvision torchaudio

# Clone openvla-oft repo and pip install to download dependencies
git clone https://github.com/moojink/openvla-oft.git
cd openvla-oft
pip install -e .

# Install Flash Attention 2 for training (https://github.com/Dao-AILab/flash-attention)
#   =>> If you run into difficulty, try `pip cache remove flash_attn` first
pip install packaging ninja
ninja --version; echo $?  # Verify Ninja --> should return exit code "0"
pip install "flash-attn==2.5.5" --no-build-isolation

# Install LIBERO simulation benchmark (required for LIBERO evaluations)
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
pip install -e LIBERO
pip install -r experiments/robot/libero/libero_requirements.txt
```

## Set Up Environment with uv

```bash
# Create and activate virtual environment with uv
uv venv --python 3.10
source .venv/bin/activate

# Install PyTorch
# Use a command specific to your machine: https://pytorch.org/get-started/locally/
uv pip install torch torchvision torchaudio

# Clone openvla-oft repo and install dependencies
git clone https://github.com/moojink/openvla-oft.git
cd openvla-oft
uv pip install -e .

# Install Flash Attention 2 for training (https://github.com/Dao-AILab/flash-attention)
#   =>> If you run into difficulty, try `uv cache clean` first
uv pip install packaging ninja
ninja --version; echo $?  # Verify Ninja --> should return exit code "0"
uv pip install "flash-attn==2.5.5" --no-build-isolation

# Install LIBERO simulation benchmark (required for LIBERO evaluations)
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
uv pip install -e LIBERO
uv pip install -r experiments/robot/libero/libero_requirements.txt
```