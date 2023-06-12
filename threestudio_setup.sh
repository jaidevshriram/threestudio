# bash
python3 -m virtualenv venv
. venv/bin/activate

git clone -b dreambooth https://github.com/jaidevshriram/threestudio.git
cd threestudio/

pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

pip install ninja

pip install -r requirements.txt

# Extras - add anything extra that was needed 
pip install transformers
pip install nerfacc==0.5.0
pip install -U 'tensorboardX'
pip install -U 'tensorboard'
pip install trimesh
pip install git+https://github.com/NVlabs/nvdiffrast/
pip install accelerate
pip install imageio-ffmpeg


