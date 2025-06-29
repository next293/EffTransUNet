# EffTransUNet
This report holds code for EffTransUNet:One Method for Medical Image Tasks Based on TransUNet.
# Usage
### 1. Datset
All datasets can be downloaded on the website.  
In this project, the "BrainTumor" folder contains the dataset for classifying brain tumors. The train_npz and test_h51 in the Synapse folder are the multi-organ segmentation datasets, and train_npz1 and test_h51 are the LeftAtrial datasets.
The multi-organ dataset was obtained from the TransUNet code https://github.com/Beckschen/TransUNet.
The LeftAtrium dataset can be downloaded from this website http://medicaldecathlon.com/, but it needs to be processed before it can be used.
The brain tumor classification dataset can be downloaded here.https://data.mendeley.com/datasets/w4sw3s9f59/1
### 2. Environment
The GPU model used in the experiment is an NVIDIA GPU with 16GB of GPU memory. The experiment was completed using the python language in the PyCharm environment.
The configuration of the Anaconda virtual environment used is that the version of torch is 2.5.0+cu121, the version of torchvision is 0.20.0+cu121, and the version of python used is 3.10.15.
### 3. Train and test
In the parameter modification experiment, the code of the TransUNet model was used. The download link of the code is https://github.com/Beckschen/TransUNet.
In this project, a new dataset, the left atrium dataset, was introduced. Modify the epoch, learning rate and batch size in the TransUNet code. The train.py file is used for training and the test.py file is used for testing.
The EffTransUNet code is modified based on the TransUNet code. In the EffTransUNet experiment, the files with cls are used for the classification dataset experiment, and the rest are used for the segmentation dataset experiment.
It should be noted during the test that due to the different data of different datasets, there are several lines of code in the test_single_volume function in the utils.py file that need to be adjusted. The adjustments that need to be made are marked with annotations.
