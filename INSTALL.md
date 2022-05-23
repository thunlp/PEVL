## Installation
Most of the requirements of this projects are exactly the same as [ALBEF](https://github.com/salesforce/ALBEF). If you have any problem of your environment, you should check their [issues page](https://github.com/salesforce/ALBEF/issues) first. Hope you will find the answer.

### Requirements
- apex 0.1
- timm 0.5.4
- yaml 0.2.5
- CUDA 11.1
- numpy 1.21.5
- pytorch 1.8.0
- torchvision 0.9.0
- transformers 4.8.1
- Python 3.7.11

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

```