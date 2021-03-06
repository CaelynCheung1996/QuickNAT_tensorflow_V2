# QuickNAT_tensorflow_v2
Tool: QuickNAT: Segmenting MRI Neuroanatomy in 20 seconds
-----------------------------------------------------------

If you use this code please cite:

Guha Roy, A., Conjeti, S., Navab, N., and Wachinger, C. 2018. QuickNAT: Segmenting MRI Neuroanatomy in 20 seconds. Accepted for publication at NeuroImage.

Link to paper: https://arxiv.org/abs/1801.04161

- Tensorflow Implementation is contributed by [IshmeetKaur](https://github.com/IshmeetKaur/QuickNAT_tensorflow)

- Pytorch Implementation is contributed by [Shayan Ahmad Siddiqui](https://github.com/ai-med/quickNAT_pytorch)

You can request the OASIS 30 dataset with manual labels in [MICCAI 2012 Grand Challenge on Multi-Atlas Labeling](https://docs.google.com/forms/d/e/1FAIpQLSfwkdSt7hWo_tjHUDu2stDsxWTaWyLJIUiS_iapbtKaydEMIw/viewform)

### Changes
- Add the matlab code create_wholedataset.m to build imdb structure for any nifti formats dataset. 
- Provide the code to remap the labels of [OASIS dataset](https://www.oasis-brains.org/) in order to reduce the total number of classes.
- Provide the code to calculate the class weights for the weighted_cross_entrophy.
- Add more comments on the original tensorflow code to help understand. 

Simple to run!
----------------------------
### Run in cluster

- Request a interative CHPC GPU node
```
qsub -I -l nodes=1:ppn=1:gpus=1:V100,walltime=1:00:00
```
- Load the module for CUDA, Singularity and run tensorflow
```
module load cuda-8.0
module load singularity-2.4.2
singularity exec --nv /export/tensorflow-1.7.0/test/ubuntu_tf_gpu python3 /home/caelyn/QuickNAT_tensorflow/training.py
```
### Run in local machine

- After activating Tensorflow environment, simply run
```
python3 training.py 
```
- Testing
```
python3 testing.py
```

Troubleshooting
-------------------
- The error was shown as 'Unable to create file (file locking disabled on this file system, use HDF5_USE_FILE_LOCKING environment variable to override)' can be solved by adding the flag:
```
export HDF5_USE_FILE_LOCKING=FALSE
```
