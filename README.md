# TredMiL
Truncated Normal Mixture Prior Based Deep Latent Model

### Article: 
S. Mahapatra and P. Maji, "Truncated Normal Mixture Prior Based Deep Latent Model for Color Normalization of Histology Images," in *IEEE Transactions on Medical Imaging*, pp. 1--12, 2023.
doi: 10.1109/TMI.2023.3238425

The codes are written in Python (Python3). The dependencies for the execution of the codes are as follows -

A. DEPENDENCIES:

 1. PyTorch
 2. numpy
 3. scipy
 4. yaml
 5. tensorboard
 6. tensorboardX

B. HARDWARE SETTING:

The model is trained using an NVIDIA RTX A4000 with 6144 CUDA cores and 16GB storage. You can try to use 
fewer GPUs or reduce the batch size if it does not fit in your GPU memory, but training stability and 
image quality cannot be guaranteed.

C. DATA ARRANGEMENT:

The code directory should contain a <Data Set folder>. The <Data Set folder> should contain 4 sub-folders-
 (a) <training folder> - contains training set image patches
 (b) <validation folder>  - contains validation set image patches
 (c) <source image folder> - contains source image patches
 (d) <template image folder> - contains template image patches for mapping

N.B. - Here '<.>' denotes the folder name (user-defined). The folder named 'DataFolder' contains sample 
data arrangement. The sub-folders contain sample image patches.

D. INPUT FILE FORMAT:

 (a) Input image patches : RGB histology images, resolution - 256x256
 (b) Config file : contains information about all hyperparameter setting, data route etc.

E. COMMAND:

 (a) For Training:  
 
     python TrainModel.py -c <Config file> (if executed without interruption)
     python TrainModel.py -c <Config file> -r (if loads a saved model)

 (b) For Mapping:   
 
     python Map_Normalize.py -c <Config file>
