<div align="center">

# SynergyNeRF 

<p align="center">
  <!-- [<a href="https://arxiv.org/pdf/2306.16928.pdf"><strong>Paper</strong></a>] -->
  [<a href="https://arxiv.org/abs/2405.07857"><strong>ArXiv</strong></a>]  
  [<a href="https://mingyukim87.github.io/SynergyNeRF/"><strong>Project</strong></a>] 
  <!-- [<a href="#citation"><strong>BibTeX</strong></a>] -->
</p>

</div>

<!-- <a href="https://arxiv.org/abs/2402.03898"><img src="https://img.shields.io/badge/Paper-arXiv:2402.03898-Green"></a>
<a href=#bibtex><img src="https://img.shields.io/badge/Paper-BibTex-yellow"></a> -->

Official PyTorch implementation of **SynergyNeRF**, as presented in our paper: \
\
**Synergistic Integration of Coordinate Network and Tensorial Feature for Improving Neural Radiance Fields from Sparse Inputs (ICML2024)** \
*[Mingyu Kim](https://mingyukim87.github.io/)<sup>1</sup>, [Jun-Seong Kim](https://ami.postech.ac.kr/members/kim-jun-seong/)<sup>2</sup>, 
[Se-Young Yun](https://fbsqkd.github.io/)<sup>1†</sup>
and [Jin-Hwa Kim](http://wityworks.com/)<sup>3†</sup>* \
<sup>1</sup>KAIST AI, <sup>2</sup>POSTECH EE, <sup>3</sup>NAVER AI Lab.  
(<sup>†</sup> indicates corresponding authors)

<img src="https://mingyukim87.github.io/SynergyNeRF/img/2_Overview_2.png" width="100%">  

## Update
- [x] Training code.
- [x] Inference code.
- [x] Datasets.

## News

- (24.05.01) Our paper has been accepted in [**ICML 2024**](https://icml.cc/virtual/2024/poster/34866). 
- (23.10.21) Our paper has been presented at [**NeurIPS2023 DeepInverse Workshop**](https://openreview.net/forum?id=28zXoRIcZd&referrer=%5Bthe%20profile%20of%20Mingyu%20Kim%5D(%2Fprofile%3Fid%3D~Mingyu_Kim2)).

## Environment Setup
```bash
# create conda environment
conda create --name SynergyNeRF python=3.9

# activate env
conda activate SynergyNeRF

# install pytorch >= 1.12 (e.g cu116)
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116

# install packages
pip install -r requirements.txt
```

This implementation utilizes the code bases of [TensoRF](https://github.com/apchenstu/TensoRF) and [HexPlane](https://github.com/Caoang327/HexPlane).

## Dataset
For static NeRFs, this implementation utilizes [NeRF Synthetic](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1) and [TankandTemples](https://dl.fbaipublicfiles.com/nsvf/dataset/TanksAndTemple.zip). 
```bash
# NeRF Synthetic: download and extract nerf_synthetic.zip
cd SynergyNeRF_3D
mkdir data
cd data
unzip nerf_synthetic.zip 

# tankandtemples: download and extract TankAndTemple.zip
unzip TankAndTemple.zip 
```
For dynamic NeRFs, this implementation utilizes [D-NeRF Dataset](https://www.dropbox.com/s/0bf6fl0ye2vz3vr/data.zip?e=1&dl=0). 
```bash
# D-NeRF dataset: download and extract data.zip
cd SynergyNeRF_4D
mkdir data
cd data
unzip data.zip 
```


## Quick Start
We provide training scripts based on config files.  
First, we illustrate training static NeRFs for nerf_synthetic and tankandtemples.  
To run this code on GPUs with 24GB of VRAM, you should use the configuration files located in `config/SynergyNeRF/revised_cfg/`.  
If you want to run the code as described in the original paper,  
please use the configuration files found in `config/SynergyNeRF_disentangled/official/`.  
```bash
# training nerf_synthetic
cd SynergyNeRF_3D 
# scenes : {chair, drums, ficus, lego, hotdog, materials, mic, ship}
python main.py --config=config/SynergyNeRF/revised_cfg/8_views/{scene}.yaml

# training tankandtemples
# scenes : {barn, caterpillar, family, truck}
python main.py --config=config/SynergyNeRF/tankandtemples_cfgs/{scene}.yaml
```
Second, we illustrate training dynamic NeRFs for the D-NeRF dataset.
```bash
# training nerf_synthetic
cd SynergyNeRF_4D 
# scenes : {bouncingballs, hellwarrior, hook, jumpingjacks, lego, mutant, standup, trex}
python main.py --config=config/SynergyNeRF/official/{scene}.yaml
```

## Bibliography
```bibtext
@InProceedings{kim2024synergistic,
  author    = {Kim, Mingyu and Kim, Jun Seong and Yun, Se Young and Kim, Jin Hwa},  
  title     = {Synergistic Integration of Coordinate Network and Tensorial Feature for Improving NeRFs from Sparse Inputs},  
  booktitle = {Proceedings of the 41th International Conference on Machine Learning},
  year      = {2024},
  series    = {Proceedings of Machine Learning Research},
  publisher = {PMLR},  
}
```

## Acknowledgement
This work was supported by Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government(MSIT) [No.2022-0-00641, XVoice: Multi-Modal Voice Meta Learning]. 
A portion of this work was carried out during an internship at <a href="https://naver-career.gitbook.io/en/teams/clova-cic/ai-lab" target="_blank">NAVER AI Lab</a>.
We also extend our gratitude to <a href="https://actnova.io" target="_blank">ACTNOVA</a> for providing the computational resources required.
