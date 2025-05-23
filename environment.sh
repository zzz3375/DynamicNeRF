conda create -n DynamicNeRF python=3.12 numpy=1.26.4 -y
conda activate DynamicNeRF

pip3 install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip3 install matplotlib tensorboard scipy opencv-python
pip install imageio scikit-image configargparse timm lpips
 pip install ipdb
sudo apt install colmap
sudo apt install graphicsmagick-imagemagick-compat