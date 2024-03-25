> [**UniDepth: Universal Monocular Metric Depth Estimation**](),  
> Luigi Piccinelli, Yung-Hsu Yang, Christos Sakaridis, Mattia Segu, Siyuan Li, Luc Van Gool, Fisher Yu,  
> CVPR 2024 (to appear),  
<!-- > *Paper ([arXiv 2304.06334](https://arxiv.org/pdf/2304.06334.pdf))*   -->


## Installation

Install the environment needed to run UniDepth with:
```shell
export VENV_DIR=<YOUR-VENVS-DIR>
export NAME=Unidepth

python -m venv $VENV_DIR/$NAME
source $VENV_DIR/$NAME/bin/activate
pip install -r requirements.txt
export PYTHONPATH="$PWD:$PYTHONPATH"
```

Note: Make sure that your compilation CUDA version and runtime CUDA version match.
You can check the supported CUDA version for precompiled packages on the [PyTorch website](https://pytorch.org/).

Run UniDepth on the given assets to test your installation:
```shell
python ./scripts/demo.py
```


## Get Started

For easy-to-use, we provide our models via torchhub, the following script is an example on how to use it from your code.
The intrinsics can be either passed (will be used) or not. The infer method takes care of the pre- and post-processing. 
```python
import torch
import numpy as np
from PIL import Image

unidepth_version = "UniDepthV1_ViTL14"
model = torch.hub.load("lpiccinelli-eth/UniDepth", unidepth_version, pretrained=True, trust_repo=True)
rgb = torch.from_numpy(np.array(Image.open(...))).permute(2,0,1) # do not normalize
intrinsics = torch.from_numpy(np.load(...)) if exists else None

predictions = model.infer(rgb, intrinsics)

depth = predictions["depth"]
xyz = predictions["points"]
intrinsics = predictions["intrinsics"]
```

To use the forward method you should format the input as:
```python
data = {"image": rgb.unsqueeze(0), "K": intrinsics.unsqueeze(0)}
predictions = model(data, {})
```

Current available versions of UniDepth on TorchHub:

1. UniDepthV1_ViTL14
2. UniDepthV1_ConvNextL

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


## Zero-Shot Visualization

<!-- ### YouTube (The Office)
<p align="center">
  <img src="docs/kitti_example.gif" alt="animated" />
</p> -->


### Nuscenes
<p align="center">
  <img src="assets/docs/nuscenes_surround.gif" alt="animated" />
</p>


## License

This software is released under Creatives Common BY-NC 4.0 license. You can view a license summary [here](LICENSE).


## Contributions

If you find any bug in the code, please report to <br>
Luigi Piccinelli (lpiccinelli_at_ethz.ch)


## Acknowledgement

This work is funded by Toyota Motor Europe via the research project [TRACE-Zurich](https://trace.ethz.ch) (Toyota Research on Automated Cars Europe).
