# NTHU's CV (Computer Vision) final competition - Face Parsing
## Kaggle link
<https://www.kaggle.com/competitions/cs6550-face-parsing/overview>

<https://www.kaggle.com/competitions/cs6550_face_parsing_unseen/overview>

## Training and Testing data 
<https://github.com/switchablenorms/CelebAMask-HQ>

<https://github.com/microsoft/FaceSynthetics>

Please see the [kaggle website](https://www.kaggle.com/competitions/cs6550-face-parsing/overview), it will assist in dividing the training and testing sets.

## Reference paper
[D-LinkNet](https://ieeexplore.ieee.org/document/8575492)

[SegFormer](https://arxiv.org/abs/2105.15203)

## Reference version of packages
```
- Python                3.7.9
- numpy                 1.16.2
- imgaug                0.4.0
- opencv-python         4.5.4.60
- matplotlib            3.5.3
- tensorflow-gpu        2.4.0
- segmentation-models   1.0.1
```

## Directory structure
```
|-- Model_DLinkNet
|    |-- ResNet_Block.py: ResNet block implementation
|    |-- D_LinkNet.py: DLinkNet implementation
|
|-- Model_SegFormer
|    |-- __init__.py: Initialization file
|    |-- utils.py: Utility functions
|    |-- Attention.py: Attention module implementation
|    |-- Head.py: Head module implementation
|    |-- modules.py: Modules implementation
|    |-- segformer.py: SegFormer model implementation
|
|-- CelebAMask-HQ: Dataset for training and testing
|
|-- thresholding.py: Thresholding method implementation, used for post-processing
|-- data_to_json.py: Convert the paths of training data to json file
|-- show_image.py: Helper function for showing image
|-- Load_Data.py: Load data for training and testing
|-- Train_DLinkNet.py: Train DLinkNet
|-- Test_DLinkNet.py: Inference using DLinkNet on CelebAMask-HQ
|-- Train_SegFormer.py: Train SegFormer
|-- Test_SegFormer.py: Inference using SegFormer on CelebAMask-HQ
|-- Test_SegFormer_unseen.py: Inference using SegFormer on unseen data
```







