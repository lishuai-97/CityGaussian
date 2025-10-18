## CityGaussian Experimental Instructions

- [CityGaussian Experimental Instructions](#citygaussian-experimental-instructions)
  - [1. Environment Setup](#1-environment-setup)
  - [2. Data Preparation](#2-data-preparation)
    - [2.1 Downloaded COLMAP Results](#21-downloaded-colmap-results)
    - [2.2 Prepare for Tanks and Temple style geometry evaluation](#22-prepare-for-tanks-and-temple-style-geometry-evaluation)
    - [2.3 Prepare GauU-Scene dataset](#23-prepare-gauu-scene-dataset)
    - [2.4 Prepare MatrixCity dataset](#24-prepare-matrixcity-dataset)
    - [2.4 Prepare Mill19 \& UrbanScene3D datasets](#24-prepare-mill19--urbanscene3d-datasets)
    - [2.5 Data Preprocessing](#25-data-preprocessing)
  - [3. Run \& Evaluate](#3-run--evaluate)
    - [A. Train coarse model](#a-train-coarse-model)
    - [B. Model Partition and Data Assignment](#b-model-partition-and-data-assignment)
    - [C. Finetune model parallelly and merge](#c-finetune-model-parallelly-and-merge)
    - [D. Evaluation Rendering Performance](#d-evaluation-rendering-performance)
    - [E. Mesh extraction and evaluation](#e-mesh-extraction-and-evaluation)
    - [F. Compression](#f-compression)

---
### 1. Environment Setup

- `Python Version: 3.9.24`
- `PyTorch Version: 2.0.1`
- `PyTorch-Lightning Version: 2.0.9.post0`
- `Torch-Scatter Version: 2.1.2`
- `CUDA Version: 11.8`

```bash
# clone repository
git clone https://github.com/DekuLiuTesla/CityGaussian.git
cd CityGaussian

# create virtual environment
conda create -n citygs python=3.9
conda activate citygs

# install pytorch
pip install -r requirements/pyt201_cu118.txt

# install requirements
pip install -r requirements.txt

# install additional package for CityGaussian
pip install -r requirements/CityGS.txt

mkdir data
```

---
### 2. Data Preparation

**Note: Due to limited disk space, all data and results are stored in the path `/data3/lishuai_datasets/3dgs/citygs_data`. Please make sure to update the path accordingly when processing the data.**

---
#### 2.1 Downloaded COLMAP Results

For COLMAP, we recommend to directly use our generated results:

- **Google Drive**: https://drive.google.com/file/d/1Uz1pSTIpkagTml2jzkkzJ_rglS_z34p7/view?usp=sharing
- **Baidu Netdisk**: https://pan.baidu.com/s/1zX34zftxj07dCM1x5bzmbA?pwd=1t6r

Suppose that you have downloaded and unzip the COLMAP results to `./data` folder like

```
├── data
│   ├── colmap_results
│   │   ├── matrix_city_aerial
│   │   │   ├── train
│   │   │   │   ├── sparse
│   │   │   │   │   ├── 0
│   │   │   │   │   │   ├── cameras.bin
│   │   │   │   │   │   ├── points3D.bin
│   │   │   │   │   │   ├── images.bin
│   │   │   ├── test
│   │   │   │   ├── sparse
│   │   │   │   │   ├── 0
│   │   │   │   │   │   ├── cameras.bin
│   │   │   │   │   │   ├── images.bin
│   │   │   │   │   │   ├── points3D.bin
│   │   ├── matrix_city_street
│   │   │   ├── train
│   │   │   ├── val
│   │   ├── building
│   │   │   ├── train
│   │   │   ├── val
│   │   ├── residence
│   │   │   ├── train
│   │   │   ├── val
│   │   ├── rubble
│   │   │   ├── train
│   │   │   ├── val
│   │   ├── sciart
│   │   │   ├── train
│   │   │   ├── val
```

---
#### 2.2 Prepare for Tanks and Temple style geometry evaluation

To evaluate surface reconstruction accuracy, please first down load the ground truth point cloud (`.ply`) and crop volume file (`.json`) to the `./data`. The `transform.txt` is used to coarsely align target points to gt points. The links are:

- **Google Drive**: https://drive.google.com/file/d/18L9AEJS2SNva7JgL2-DmqhoNtfDPSSY5/view?usp=sharing
- **Baidu Netdisk**: https://pan.baidu.com/s/1WBJkj42AOsgrNb7YBmcbGg?pwd=in4i

**Note that Mill19 and UrbanScene3D doesn't provide ground-truth point cloud, thus they are not included.** 

In `./scripts/gt_generate.sh`, we provide the script about how we downsample the ground truth point cloud and generate the crop volume. If you need to process the custom dataset, please refer to the script. For custom dataset, you can also refer to this script to generate your crop volume file (.json).

---
#### 2.3 Prepare GauU-Scene dataset

For GauU-Scene dataset, please follow instruction [here](https://saliteta.github.io/CUHKSZ_SMBU/) to download. The data includes RGB images and COLMAP results.

---
#### 2.4 Prepare MatrixCity dataset

1. Download `small_city` version of aerial data of [MatricCity](https://github.com/city-super/MatrixCity) and save to `data/matrix_city/aerial`.

2. Unzip the data to `input` folder for each block in train and test set. You can take `scripts/untar_matrixcity_train.sh` and `scripts/untar_matrixcity_test.sh` as reference.
  
```bash
# Train set
cp scripts/untar_matrixcity_train.sh data/matrix_city/aerial/train
cd data/matrix_city/aerial/train
# comment out the scripts for `street` view data if you only need aerial view
bash untar_matrixcity_train.sh

# Test set
cp scripts/untar_matrixcity_test.sh data/matrix_city/aerial/train
cd data/matrix_city/aerial/test
# comment out the scripts for `street` view data if you only need aerial view
bash untar_matrixcity_test.sh
```

3. Run following command to prepare data with generated COLMAP results.

```bash
bash scripts/data_proc_mc.sh
```

4. [Optional] Run following command to prepare data from scratch (The COLMAP step make take a long time for over 5000 images ).

```bash
bash scripts/data_proc_mc_scratch.sh
```

5. The process above also applies to the street view of MatrixCity's small_city version. The mentioned scripts also contains the required steps for street view data preparation.

---
#### 2.4 Prepare Mill19 & UrbanScene3D datasets

1. Download data of Mill19 and UrbanScene3D according to instruction from [MegaNeRF](https://github.com/cmusatyalab/mega-nerf). Save the data to `data/mill19` and `data/urban_scene_3d` respectively. It is worth noticing that the used `UrbanScene3D-V1` should be downloaded from [here](https://github.com/Linxius/UrbanScene3D).

2. Run following command to prepare data of Mill19 with generated COLMAP results.

```bash
bash scripts/data_proc_mill19.sh
```

3. [Optional] Run following command to prepare data of Mill19 from scratch (The COLMAP step make take a long time for large amount of images ).

```bash
bash scripts/data_proc_mill19_scratch.sh
```

4. Run following command to prepare data of UrbanScene3D with generated COLMAP results.

```bash
bash scripts/data_proc_us3d.sh
```

5. [Optional] Run following command to prepare data of UrbanScene3D from scratch (The COLMAP step make take a long time for large amount of images ).

```bash
bash scripts/data_proc_us3d_scratch.sh
```

---
#### 2.5 Data Preprocessing

The desried dataset folder structure is:

```
├── data
│   ├── your_scene
│   │   ├── images
│   │   ├── sparse
│   │   │   ├── 0
│   │   │   │   ├── cameras.bin
│   │   │   │   ├── points3D.bin
│   │   │   │   ├── images.bin
│   ├── geometry_gt
│   │   ├── your_scene
│   │   │   ├── your_gt_pcd.ply
│   │   │   ├── your_gt_pcd.json
│   │   │   ├── transform.txt [optional]
```

**Firstly**, downsample the images to desired size:

```bash
python utils/image_downsample.py data/your_scene/images --factor $DOWNSAMPLE_RATIO

# aerial view of MatrixCity
python utils/image_downsample.py data/matrix_city/aerial/train/block_all/input --factor 1.2
python utils/image_downsample.py data/matrix_city/aerial/test/block_all_test/input --factor 1.2
```

The `$DOWNSAMPLE_RATIO` is `3.4175` for `GauU-Scene`, `1.2` for `aerial` view of `MatrixCity`, `1.0` for `street` view of `MatrixCity` (no downsample), and `4.0` for `Mill19` and `UrbanScene3D`.

**Secondly**, prepare [Depth Anything V2](https://depth-anything-v2.github.io/) for depth regulatization:

```bash
# clone the repo.
git clone https://github.com/DepthAnything/Depth-Anything-V2 utils/Depth-Anything-V2

# NOTE: do not run `pip install -r utils/Depth-Anything-V2/requirements.txt`

# download the pretrained model `Depth-Anything-V2-Large`
mkdir utils/Depth-Anything-V2/checkpoints
wget -O utils/Depth-Anything-V2/checkpoints/depth_anything_v2_vitl.pth "https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true"
```

The depth can be generated with:

```bash
python utils/estimate_dataset_depths.py data/your_scene --image_dir images -d $DOWNSAMPLE_RATIO

# aerial view of MatrixCity
python utils/estimate_dataset_depths.py data/matrix_city/aerial/train/block_all --image_dir images -d 1.2
python utils/estimate_dataset_depths.py data/matrix_city/aerial/test/block_all_test --image_dir images -d 1.2
```

---
### 3. Run & Evaluate

The detailed setting of each step on GauU-Scene and MatrixCity Dataset can be found in `./scripts/run_citygs_SCENE.sh`. For aerial view of MatrixCity, please refer to [run_citygs_mc_aerial.sh](/scripts/run_citygs_mc_aerial.sh).

- If applying 3DGS, please change model and renderer as done in `configs/citygs_lfls_sh2_trim.yaml` over `configs/citygsv2_lfls_sh2_trim.yaml`.
- To obtain full model, adjust SH degree to `3` and enable `diable_trimming` in renderer.
- You can also run on Mill19 or UrbanScene3D dataset by setting configs according to that of branch [V1-Original](https://github.com/DekuLiuTesla/CityGaussian/tree/V1-original).

---
#### A. Train coarse model

```bash
# $COARSE_NAME is the name of coarse model training config file.
python main.py fit \
        --config configs/$COARSE_NAME.yaml \
        -n $COARSE_NAME \
```

**Tips**: If you want to accelerate this part, you can use 7K iteration or lower resolution (Remember to adjust the resolution of estimated depth accordingly). But the overall performance may drop, as validated in our CityGaussianV2 paper and issue https://github.com/DekuLiuTesla/CityGaussian/issues/124.

---
#### B. Model Partition and Data Assignment

```bash
# $NAME is the name of parallel tuning config file.
# to partition in contracted space, add `--contract`
# to partition after xy plane is aligned to ground plane, add `--reorient`
python utils/partition_citygs.py --config_path configs/$NAME.yaml --force
```

The command also visualizes block division and data partitioning, as like below:
![partition](./assets/analysis.png)

---
#### C. Finetune model parallelly and merge

```bash
python utils/train_citygs_partitions.py -n $NAME
python utils/merge_citygs_ckpts.py outputs/$NAME
```

---
#### D. Evaluation Rendering Performance

```bash
# For finetuned model, since the split mode and eval ratios are changed for per-block tuning, the parameters have to be reappointed. Please see the script for details.
python main.py test \
        --config outputs/$NAME/config.yaml \
        --save_val \
        --test_speed \  # calculate FPS
```

---
#### E. Mesh extraction and evaluation

```bash
python utils/gs2d_mesh_extraction.py \
        outputs/$NAME \
        --voxel_size $VOXEL_SIZE \
        --sdf_trunc $SDF_TRUNC \
        --depth_trunc $DEPTH_TRUNC \

python tools/eval_tnt/run.py \
        --scene your_gt_pcd \
        --dataset-dir data/geometry_gt/your_scene \
        --transform-path data/geometry_gt/your_scene/transform.txt \
        --ply-path "outputs/$NAME/fuse_post.ply"
```

---
#### F. Compression

```bash
python tools/vectree_lightning.py \
        --model_path outputs/$NAME \
        --save_path outputs/$NAME/vectree \
```

If you use SH degreee of 2, please add `--sh_degree 2`. And if you want to decode the quantized result to checkpoint, add `--no_save_ply`. If 3DGS is used, add `--gs_dim 3`.