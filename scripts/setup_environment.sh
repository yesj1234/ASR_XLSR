# 1. install gcc and cmake
sudo apt install -y gcc
sudo apt install -y cmake

# 2. install cuda
wget wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update 
sudo apt-get install -y cuda

# 3. install pip
sudo apt install -y python3-pip 

# 4. install python libraries 
pip install mecab-python3
pip install unidic-lite
pip install transformers 
pip install datasets 
pip install torch 
pip install torchaudio 
pip install numpy 
pip install jiwer 
pip install soundfile 
pip install librosa 
pip install evaluate
sudo apt-get install -y sox 
pip install sox 
sudo apt update 
sudo apt install -y ffmpeg 
pip install -U accelerate 
pip install wandb

