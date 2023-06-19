First, create a conda virtual environment and activate it:
```
conda create -n procedurevrl python=3.7 -y
source activate procedurevrl
```

Then, install the following packages:

- torchvision: `pip install torchvision` or `conda install torchvision -c pytorch`
- [fvcore](https://github.com/facebookresearch/fvcore/): `pip install 'git+https://github.com/facebookresearch/fvcore'`
- simplejson: `pip install simplejson`
- einops: `pip install einops`
- timm: `pip install timm`
- PyAV: `conda install av -c conda-forge`
- psutil: `pip install psutil`
- OpenCV: `pip install opencv-python`
- tensorboard: `pip install tensorboard`
- ffmpeg: `pip install ffmpeg-python`
- pandas: `pip install pandas`
- submitit: `pip install submitit`

Note that the part of this repo also requires ffmpeg installed in the system.

Lastly, build the codebase by running:
```
git clone https://github.com/facebookresearch/ProcedureVRL
cd ProcedureVRL
# way 1
python setup.py build develop
# way 2
export PYTHONPATH="$PYTHONPATH:<Path_To_This_Directory>"
```

As a reference, the code was developped using Python 3.7.13, Pytorch 1.10.1+cu111, torchvision 0.11.2+cu111, and NCCL (2, 10, 3). Each compute node contains 8 A100 GPUs. Other platforms or GPU cards have not been fully tested. 
