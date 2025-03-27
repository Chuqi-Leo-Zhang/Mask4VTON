# Mask4VTON

Mask4VTON is a robust pipeline designed for Virtual Try-On (VTON) systems. It leverages a combination of **classification** and **segmentation models** to generate precise garment masks, making it an ideal preliminary component for VTON applications.

## Pipeline Overview
The Mask4VTON pipeline integrates the following key components:
1. **Classification Model (ResNet)**: Trained on the **Fashion-MNIST** dataset to predict the class of the garment.
2. **Segmentation Model (Mask2Former with Tiny SwinTransformer)**: Trained on the **CIHP dataset** to perform human parsing and garment segmentation.
3. **Mask Combination**: Combines segmentation masks based on the predicted garment class to generate the final mask for Virtual Try-On. 

You can download the weight of models at [here](https://drive.google.com/drive/folders/1tGgDTZeWiVsbgkUdzuo3WhAqtCtOZpFV?usp=share_link). After download checkpoints directory you should directly put it under MASK4VTON direcoty.

## Installation
Follow these steps to set up the environment and install dependencies:

### 1. Clone the Repository with Submodules
```bash
git clone --recurse-submodules https://github.com/Chuqi-Leo-Zhang/Mask4VTON.git
```

### 2. Create and Activate the Conda Environment
```bash
conda create --name ssseg python=3.10
conda activate ssseg
```

### 3. Install PyTorch and CUDA
```bash
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.1 -c pytorch -c nvidia
```

### 4. Set Up the Segmentation Framework (sssegmentation)
Navigate to the segmentation directory:
```bash
cd sssegmentation
export SSSEG_WITH_OPS=1
```
Install the required packages:
```bash
pip install -r requirements.txt
python setup.py develop
```
Install **MMCV** for CUDA support:
```bash
pip install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.4/index.html
```

## Directory Structure

After cloning the repository and setting up the environment, your directory structure should look like this:

```
Mask4VTON
├── checkpoints               # Directory to store model weights
├── classification            # Classification model and training scripts
├── clothes                   # Directory to store clothes images
├── persons                   # Directory to store person images
├── sssegmentation            # Segmentation framework submodule           
└── mask4vton.ipynb           # Main notebook to run the pipeline
```

## Training and Models
### Classification Model
The classification model uses **ResNet** trained on the **Fashion-MNIST** dataset. Training scripts and notebooks can be found at [cls_train](classification/cls_train.ipynb).

### Segmentation Model
The segmentation model uses **Mask2Former with Tiny SwinTransformer**, trained on the **CIHP dataset**. More details and configurations can be found at [here](https://github.com/SegmentationBLWX/sssegmentation/tree/main?tab=readme-ov-file).



## Pipeline Workflow
1. **Classification**: Predict the class of the garment using the trained ResNet model.
2. **Segmentation**: Perform human parsing and garment segmentation using Mask2Former.
3. **Mask Generation**: Combine masks based on garment classes to generate the final mask for VTON.

## Tips
1. The clothes images are expected to has a white background.
2. For easy implementation, you can save your clothes images in directory [clothes](clothes) and save the person image in directory [persons](persons). I also provided some clothes and person images for you to test.
3. The clothes type is expected to be one of the following types: 'T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Ankle Boot'.
4. The mask result would be save at person_seg directory. Some details would be mentioned in the [notebook](mask4vton.ipynb)


## Acknowledgments
Special thanks to the authors of [**sssegmentation**](https://github.com/SegmentationBLWX/sssegmentation.git) for providing robust segmentation models and training scripts.






