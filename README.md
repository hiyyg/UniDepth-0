[![arXiv](https://img.shields.io/badge/arXiv-UniDepth-red)](https://arxiv.org/abs/2403.18913)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Cooming%20Soon-yellow)](https://huggingface.co/spaces/lpiccinelli/UniDepth)

[![KITTI Benchmark](https://img.shields.io/badge/KITTI%20Benchmark-1st%20(at%20submission%20time)-blue)](https://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/unidepth-universal-monocular-metric-depth/monocular-depth-estimation-on-nyu-depth-v2)](https://paperswithcode.com/sota/monocular-depth-estimation-on-nyu-depth-v2?p=unidepth-universal-monocular-metric-depth)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/unidepth-universal-monocular-metric-depth/monocular-depth-estimation-on-kitti-eigen)](https://paperswithcode.com/sota/monocular-depth-estimation-on-kitti-eigen?p=unidepth-universal-monocular-metric-depth)


# UniDepth: Universal Monocular Metric Depth Estimation

![](assets/docs/unidepth-banner.png)

> [**UniDepth: Universal Monocular Metric Depth Estimation**](https://arxiv.org/abs/2403.18913),  
> Luigi Piccinelli, Yung-Hsu Yang, Christos Sakaridis, Mattia Segu, Siyuan Li, Luc Van Gool, Fisher Yu,  
> CVPR 2024 (to appear),  
> *Paper at [arXiv 2403.18913](https://arxiv.org/pdf/2403.18913.pdf)*  


## News and ToDo

- [ ] Release UniDepth as Pip package
- [ ] Release smaller models
- [ ] HuggingFace/Gradio demo
- [ ] Release UniDepthV2  
- [x] `01.04.2024`: Inference and v1 models released  
- [x] `26.02.2024`: UniDepth accepted at CVPR 2024!  


## Zero-Shot Visualization

### YouTube (The Office - Parkour)
<p align="center">
  <img src="assets/docs/theoffice.gif" alt="animated" />
</p>

### NuSceneS stitched cameras
<p align="center">
  <img src="assets/docs/nuscenes_surround.gif" alt="animated" />
</p>


## Installation

Requirements are not in principle hard requirements, but there might be some differences (not tested):
- Linux
- Python 3.10+ 
- CUDA 11.8

Install the environment needed to run UniDepth with:
```shell
export VENV_DIR=<YOUR-VENVS-DIR>
export NAME=Unidepth

python -m venv $VENV_DIR/$NAME
source $VENV_DIR/$NAME/bin/activate

# Install PyTorch
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt

# Install xFormers
pip install xformers==0.0.24 --index-url https://download.pytorch.org/whl/cu118

# Install Pillow-SIMD (Optional)
pip uninstall pillow
CC="cc -mavx2" pip install -U --force-reinstall pillow-simd

export PYTHONPATH="$PWD:$PYTHONPATH"
```

If you use conda, you should change the following: 
```shell
python -m venv $VENV_DIR/$NAME -> conda create -n $NAME python=3.11
source $VENV_DIR/$NAME/bin/activate -> conda activate $NAME
```

*Note*: Make sure that your compilation CUDA version and runtime CUDA version match.  
You can check the supported CUDA version for precompiled packages on the [PyTorch website](https://pytorch.org/).

*Note*: xFormers may raise the the Runtime "error": `Triton Error [CUDA]: device kernel image is invalid`.  
This is related to xFormers mismatching system-wide CUDA and CUDA shipped with torch.  
It may considerably slow down inference.

Run UniDepth on the given assets to test your installation (you can check this script as guideline for further usage):
```shell
python ./scripts/demo.py
```
If everything runs correctly, `demo.py` should print: `ARel: 5.13%`.


## Get Started

After installing the dependencies, you can load the pre-trained models easily through TorchHub. For instance, if you want UniDepth v1 with Dino backbone:
```python
import torch

version="v1"
backbone="ViTL14"
model = torch.hub.load("lpiccinelli-eth/UniDepth", "UniDepth", version=version, backbone=backbone, pretrained=True, trust_repo=True, force_reload=True)
```

or

```python
from unidepth.models import UniDepthV1

model = UniDepthV1.from_pretrained(backbone="ViTL14") # on CPU
```

Then you can generate the metric depth estimation and intrinsics prediction directly from RGB image only as follows:

```python
import numpy as np
from PIL import Image

# Move to CUDA, if any
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Load the RGB image and the normalization will be taken care of by the model
rgb = torch.from_numpy(np.array(Image.open(image_path))).permute(2, 0, 1) # C, H, W

predictions = model.infer(rgb)

# Metric Depth Estimation
depth = predictions["depth"]

# Point Cloud in Camera Coordinate
xyz = predictions["points"]

# Intrinsics Prediction
intrinsics = predictions["intrinsics"]
```

You can use ground truth intrinsics as input to the model as well:
```python
intrinsics_path = "assets/demo/intrinsics.npy"

# Load the intrinsics if available
intrinsics = torch.from_numpy(np.load(intrinsics_path)) # 3 x 3

predictions = model.infer(rgb, intrinsics)
```

To use the forward method you should format the input as:
```python
data = {"image": rgb, "K": intrinsics}
predictions = model(data, {})
```

For easy-to-use, we provide our models via TorchHub and current available versions of UniDepth are with ViT-L and ConvNext-L backbones:

1. UniDepthV1_ViTL14
2. UniDepthV1_ConvNextL

For imporved flexibility, we provide a UniDepth wrapper where you need to specifiy the version and the backbone to torch.hub.load() call, such as:
```python
torch.hub.load("lpiccinelli-eth/UniDepth", "UniDepth", version=version, backbone=backbone, pretrained=True, trust_repo=True, force_reload=True)
```

or you can directly embed in the TorchHub model-string:
```python
torch.hub.load("lpiccinelli-eth/UniDepth", "UniDepthV1_ViTL14", pretrained=True, trust_repo=True, force_reload=True)
```

Please visit our [HuggingFace](https://huggingface.co/lpiccinelli/UniDepth) to access models weights.

## Results

### Metric Depth Estimation
The performance reported is d1 (higher is better) on zero-shot evaluation. The common split between SUN-RGBD and NYUv2 is removed from SUN-RGBD validation set for evaluation. 
*: non zero-shot on NYUv2 and KITTI.

| Model | NYUv2 | SUN-RGBD | ETH3D | Diode (In) | IBims-1 | KITTI | Nuscenes | DDAD | 
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| BTS* | 88.5 | 76.1 | 26.8 | 19.2 | 53.1 | 96.2 | 33.7 | 43.0 |
| AdaBins* | 90.1 | 77.7 | 24.3 | 17.4 | 55.0 | 96.3 | 33.3 | 37.7 |
| NeWCRF* | 92.1 | 75.3 | 35.7 | 20.1 | 53.6 | 97.5 | 44.2 | 45.6 | 
| iDisc* | 93.8 | 83.7 | 35.6 | 23.8 | 48.9 | 97.5 | 39.4 | 28.4 |
| ZoeDepth* | 95.2 | 86.7 | 35.0 | 36.9 | 58.0 | 96.5 | 28.3 | 27.2 |
| Metric3D | 92.6 | 15.4 | 45.6 | 39.2 | 79.7 | 97.5 | 72.3 | - |
| UniDepth_ConvNext | 97.2| 94.8 | 49.8 | 60.2 | 79.7 | 97.2 | 83.3 | 83.2 |
| UniDepth_ViT | 98.4 | 96.6 | 32.6 | 77.1 | 23.9 | 98.6 | 86.2 | 86.4 |


## Contributions

If you find any bug in the code, please report to Luigi Piccinelli (lpiccinelli@ethz.ch)


## Citation

If you find our work useful in your research please consider citing our publication:
```bibtex
@inproceedings{piccinelli2024unidepth,
    title={UniDepth: Universal Monocular Metric Depth Estimation},
    author = {Piccinelli, Luigi and Yang, Yung-Hsu and Sakaridis, Christos and Segu, Mattia and Li, Siyuan and Van Gool, Luc and Yu, Fisher},
    booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2024}
}
```


## License

This software is released under Creatives Common BY-NC 4.0 license. You can view a license summary [here](LICENSE).


## Acknowledgement

This work is funded by Toyota Motor Europe via the research project [TRACE-Zurich](https://trace.ethz.ch) (Toyota Research on Automated Cars Europe).
