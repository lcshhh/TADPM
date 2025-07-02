# TADPM：Automatic Tooth Arrangement with Joint Features of Point and Mesh Representations via Diffusion Probabilistic Models

This is the official PyTorch implementation of our CAGD paper *"Automatic Tooth Arrangement with Joint Features of Point and Mesh Representations via Diffusion Probabilistic Models."*

### Installation

First create a conda environment:

```shell
conda create --name tadpm
conda activate tadpm
```

Pytorch / Python combination that was verified to work is:

- Python 3.10, Pytorch 2.3.1, CUDA 11.8

To install python requirements:

```shell
pip install -r requirements.txt
```

To install chamfer distance:

```shell
cd chamfer_dist
python setup.py install
```

To install manifold, please refer to https://github.com/ZhaoHengJiang/MeshReconstruction/tree/main/Manifold



## Dataset

See the [Data Use Agreement](./Data-Use-Agreement.pdf) for details.



### Data pre-process

- First you need to extract single tooth meshes from dental models, run:

```shell
bash scripts/get_mesh.sh
```

Note that this script automatically **centers and normalizes** the mesh. You may adjust the normalization scale within the script as needed.

- To get pointcloud files:

```shell
bash scripts/get_pointcloud.sh
```

This script extracts **corresponding** points from individual teeth before and after orthodontic treatment.

- To get remeshed files:

```shell
bash scripts/remesh.sh
```

- To get **ground truth transformation parameters**, you can run:

```shell
bash scripts/register.sh
```

You need to **adjust the data directory parameters** in all the scripts mentioned above accordingly.



### Pretraining

To pretrain MeshMAE:

```shell
bash scripts/pretrain.sh
```
You can also refer to https://github.com/liang3588/MeshMAE for more details.



### Training

To train the TADPM model:

```shell
bash scripts/train.sh
```

When training TADPM, you should specify the path to the **pretrained MeshMAE model checkpoint** in this script.



## Evaluation

Once training is complete, you can run:

```
bash scripts/get_result.sh
```



## BibTeX

```
@article{lei2024automatic,
  title={Automatic tooth arrangement with joint features of point and mesh representations via diffusion probabilistic models},
  author={Lei, Changsong and Xia, Mengfei and Wang, Shaofeng and Liang, Yaqian and Yi, Ran and Wen, Yu-Hui and Liu, Yong-Jin},
  journal={Computer Aided Geometric Design},
  volume={111},
  pages={102293},
  year={2024},
  publisher={Elsevier}
}
```

