## Installation
```bash
conda create --name pevl python=3.7.11
conda activate pevl

conda install ruamel_yaml
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge

pip install opencv-python 
pip install timm
pip install transformers==4.8.1

export INSTALL_DIR=$PWD
#install apex
git clone https://github.com/NVIDIA/apex.git
cd apex
python setup.py install --cuda_ext --cpp_ext

unset INSTALL_DIR