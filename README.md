# QuickNAT_tensorflow_v2
Tool: QuickNAT: Segmenting MRI Neuroanatomy in 20 seconds
-----------------------------------------------------------

If you use this code please cite:

**_Guha Roy, A., Conjeti, S., Navab, N., and Wachinger, C. 2018. QuickNAT: Segmenting MRI Neuroanatomy in 20 seconds. Accepted for publication at NeuroImage._**

Link to paper: https://arxiv.org/abs/1801.04161

- Tensorflow Implementation is contributed by [IshmeetKaur](https://github.com/IshmeetKaur/QuickNAT_tensorflow)

- Pytorch Implementation is contributed by [Shayan Ahmad Siddiqui](https://github.com/ai-med/quickNAT_pytorch)

### Changes
- Add the matlab code create_wholedataset.m to build imdb structure for any nifti formats dataset. 
- Provide the code to remap the labels of [OASIS dataset](https://www.oasis-brains.org/) in order to reduce the total number of classes.
- Provide the code to calculate the class weights for the weighted_cross_entrophy.
- Add more comments on the original tensorflow code to help understand. 
