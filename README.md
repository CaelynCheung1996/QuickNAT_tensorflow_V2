# QuickNAT_tensorflow_v2
Tool: QuickNAT: Segmenting MRI Neuroanatomy in 20 seconds
-----------------------------------------------------------

If you use this code please cite:

Guha Roy, A., Conjeti, S., Navab, N., and Wachinger, C. 2018. QuickNAT: Segmenting MRI Neuroanatomy in 20 seconds. Accepted for publication at NeuroImage.

Link to paper: https://arxiv.org/abs/1801.04161

Tensorflow Implementation is contributed by [IshmeetKaur](https://github.com/IshmeetKaur/QuickNAT_tensorflow)

Pytorch Implementation is contributed by [Shayan Ahmad Siddiqui](https://github.com/ai-med/quickNAT_pytorch)

### Changes
- Add the matlab code create_wholedataset.m to build imdb structure for this project. 
- Provide the code to remap the labels of OASIS data in order to reduce the total number of classes.
- Provide the code to calculate the class weights for the weighted_cross_entrophy.
- Add more comments on the original tensorflow code to help understand. 
