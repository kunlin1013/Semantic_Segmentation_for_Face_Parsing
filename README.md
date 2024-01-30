## NTHU's CV (Computer Vision) final competition - Face Parsing
### Kaggle link
<https://www.kaggle.com/competitions/cs6550-face-parsing/overview>

<https://www.kaggle.com/competitions/cs6550_face_parsing_unseen/overview>

### Training and Testing data 
<https://github.com/switchablenorms/CelebAMask-HQ>

<https://github.com/microsoft/FaceSynthetics>

Please see the [kaggle website](https://www.kaggle.com/competitions/cs6550-face-parsing/overview), it will assist in dividing the training and testing sets.

### Reference paper
[D-LinkNet](https://ieeexplore.ieee.org/document/8575492)

[SegFormer](https://arxiv.org/abs/2105.15203)

### Reference version of packages
```
- Python                3.7.9
- numpy                 1.16.2
- imgaug                0.4.0
- opencv-python         4.5.4.60
- matplotlib            3.5.3
- tensorflow-gpu        2.4.0
- segmentation-models   1.0.1
```

### Directory structure
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
|-- CSV_process
|    |--csv_utils.py: Output a CSV file in the format requested by the TA
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

### How to run the code

- The code is written in Python and tested on Window OS.
- To run the code, utilize VSCode environment or open a terminal and type `python *file_name*.py` in this directory.
    - E.g., type `python ./Train_DLinkNet.py` in the terminal to train DLinkNet.
- For `Train_DLinkNet.py` and `Train_SegFormer.py`:
    - After the training is done, the weight of model will be saved in the directory same as training script.
    - The file name of the checkpoint is named in the format of `weights-improvement-*epoch*-*validation_loss*.h5`.
    - Note that it may take **about 10 hours** to train the model for 60 epochs.
- For `Test_DLinkNet.py`, `Test_SegFormer.py` and `Test_SegFormer_unseen.py`:
    - The provided code for `Test_DLinkNet.py` and `Test_SegFormer.py` are the version for testing on CelebAMask-HQ, which utilize the **bagging** technique as mentioned in the report.
    - The provided code for `Test_SegFormer_unseen.py` is the version for testing on unseen data. To test for different weight, one should modify the weight path in the code, the location is near **line 73**.
    - Note that one should **delete the .csv file** before inference since the generation of the file is written in a appending way.






