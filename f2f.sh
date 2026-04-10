#!/bin/bash

# -----------------------------
# FACE2FACE AUTO SETUP SCRIPT
# -----------------------------

# 1️⃣ Create virtual environment
echo "Creating virtual environment..."
python3 -m venv f2f_env
source f2f_env/bin/activate

# 2️⃣ Upgrade pip and basic tools
echo "Upgrading pip, setuptools, wheel..."
pip install --upgrade pip setuptools wheel

# 3️⃣ Install core dependencies
echo "Installing dependencies..."
pip install numpy==1.25.0 scipy==1.11.0 matplotlib pillow tqdm imageio imageio-ffmpeg opencv-python dlib ffmpeg-python
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install scikit-learn==1.2.2

# 4️⃣ Clone stable FaceSwap repo
echo "Cloning FaceSwap repo..."
git clone https://github.com/deepfakes/faceswap.git
cd faceswap || exit

# 5️⃣ Download pre-trained model
echo "Downloading pre-trained model..."
mkdir -p weights
cd weights || exit
curl -L -O https://github.com/deepfakes/faceswap-models/releases/download/v1.0/64_64_model.zip
unzip 64_64_model.zip
cd ..

# 6️⃣ Prepare test video
echo "Downloading test video..."
curl -L -o input_video.mp4 https://sample-videos.com/video123/mp4/720/big_buck_bunny_720p_1mb.mp4

# 7️⃣ Run a test face swap
echo "Running Face2Face swap demo..."
python faceswap.py convert -i input_video.mp4 -o output_video.mp4 -m weights/

echo "✅ Setup complete! Check output_video.mp4 for results."
