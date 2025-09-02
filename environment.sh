conda create -n DynamicNeRF python=3.12 numpy=1.26.4 -y
conda activate DynamicNeRF

pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128
# pip3 install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
# pip3 install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip3 install matplotlib tensorboard scipy opencv-python pandas
pip install imageio scikit-image configargparse timm lpips open3d
pip install ipdb 
sudo apt install colmap -y
sudo apt install graphicsmagick-imagemagick-compat -y