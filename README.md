This is the official repository for our paper "GRSV-2DGS : Geometry-Reliable Sparse-View 2D Gaussian Splatting for Surface-Aware Novel View Synthesis"

![result](assets/0.png)

## Installation

```bash
# download
git clone https://github.com/Thyme-21/GRSV-2DGS.git --recursive

# if you have an environment used for 3dgs, use it
# if not, create a new environment
conda env create --file environment.yml
conda activate GRSV-2DGS
```
### Data Preparation

In the data preparation stage, we first reconstruct sparse-view inputs using **Structure-from-Motion (SfM)** with the provided camera poses from the datasets. Then, we perform dense stereo matching using COLMAP’s `patch_match_stereo` function, followed by `stereo_fusion` to generate the dense stereo point cloud.

```bash
mkdir dataset
cd dataset

# Download MipNeRF-360 dataset and RawNeRF dataset
wget http://storage.googleapis.com/gresearch/refraw360/360_v2.zip
unzip -d mipnerf360 360_v2.zip

wget http://storage.googleapis.com/gresearch/refraw360/raw.zip
unzip -d mipnerf360 360_v2.zip

# Generate sparse point cloud using COLMAP (limited views) for MipNeRF-360
python tools/colmap_360.py

# Generate sparse point cloud using COLMAP (limited views) for RawNeRF
python tools/colmap_llff.py
```

## Training
To train a scene, simply use
```bash
python train.py -s <path to COLMAP or NeRF Synthetic dataset> -m <output> --n_views 12 --eval
```
Commandline arguments for regularizations
```bash
--lambda_normal  # hyperparameter for normal consistency
--lambda_distortion # hyperparameter for depth distortion
--depth_ratio # 0 for mean depth and 1 for median depth, 0 works for most cases
```
**Tips for adjusting the parameters on your own dataset:**
- For unbounded/large scenes, we suggest using mean depth, i.e., ``depth_ratio=0``,  for less "disk-aliasing" artifacts.

## Testing
### Bounded Mesh Extraction
To export a mesh within a bounded volume, simply use
```bash
python render.py -m <path to pre-trained model> -s <path to COLMAP dataset> 
```
Commandline arguments you should adjust accordingly for meshing for bounded TSDF fusion, use
```bash
--depth_ratio # 0 for mean depth and 1 for median depth
--voxel_size # voxel size
--depth_trunc # depth truncation
```
If these arguments are not specified, the script will automatically estimate them using the camera information.

### Unbounded Mesh Extraction
To export a mesh with an arbitrary size, we devised an unbounded TSDF fusion with space contraction and adaptive truncation.
```bash
python render.py -m <path to pre-trained model> -s <path to COLMAP dataset> --mesh_res 1024 --n_views 12 --eval

**Tips for adjusting the parameters on your own dataset:**
- For unbounded/large scenes, we suggest using mean depth, i.e., ``depth_ratio=0``,  for less "disk-aliasing" artifacts.

## Quick Examples
Assuming you have downloaded [MipNeRF360](https://jonbarron.info/mipnerf360/), simply use
```bash
python train.py -s <path to m360>/<garden> -m output/m360/garden --n_views 12 --eval
# use our unbounded mesh extraction!!
python render.py -s <path to m360>/<garden> -m output/m360/garden --unbounded --skip_test --skip_train --mesh_res 1024 --eval --n_views 12
# or use the bounded mesh extraction if you focus on foreground
python render.py -s <path to m360>/<garden> -m output/m360/garden --skip_test --skip_train --mesh_res 1024 --eval --n_views 12
```

## Full evaluation
We provide scripts to evaluate our method of novel view synthesis and geometric reconstruction.

#### Novel View Synthesis
For novel view synthesis on MipNeRF360 (which also works for other colmap datasets), use
python scripts/m360_eval.py -m360 <path to the MipNeRF360 dataset>

#### Geometry reconstruction

For geometry reconstruction on TnT dataset, please download the preprocessed [TnT_data](https://huggingface.co/datasets/ZehaoYu/gaussian-opacity-fields/tree/main). You also need to download the ground truth [TnT_GT](https://www.tanksandtemples.org/download/), including ground truth point cloud, alignments and cropfiles.

> [!IMPORTANT]  
> Due to historical issue, you should use open3d==0.10.0 for evaluating TNT.

```bash
# use open3d 0.18.0, skip metrics
python scripts/tnt_eval.py --TNT_data <path to the preprocessed TNT dataset>   \
     --TNT_GT <path to the official TNT evaluation dataset> --skip_metrics

# use open3d 0.10.0, skip traing and rendering
python scripts/tnt_eval.py --TNT_data <path to the preprocessed TNT dataset>   \
     --TNT_GT <path to the official TNT evaluation dataset> --skip_training --skip_rendering
```

## Acknowledgements
This project is built upon [2DGS](https://github.com/hbb1/2d-gaussian-splatting). The TSDF fusion for extracting mesh is based on [Open3D](https://github.com/isl-org/Open3D). The rendering script for MipNeRF360 is adopted from [Multinerf](https://github.com/google-research/multinerf/), while the evaluation scripts for Tanks and Temples dataset are taken from [TanksAndTemples](https://github.com/isl-org/TanksAndTemples/tree/master/python_toolbox/evaluation), respectively. The fusing operation for accelerating the renderer is inspired by [Han's repodcue](https://github.com/Han230104/2D-Gaussian-Splatting-Reproduce). We would also like to thank [FSGS](https://github.com/VITA-Group/FSGS). for the inspiration.We thank all the authors for their great repos. 
