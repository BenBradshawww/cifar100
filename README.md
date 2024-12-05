## Summary

The goal of this project is to benchmark the performance of image classification models on the cifar100 data set.

The models I have tested include:
* ViT (tiny): Vision Image Transfomer (tiny version).


### Training Process
To determine the learning rate for each model, the learning rate test by Leslie N. Smith [1] is performed. This test gradually increases learning rate till the loss diverges and plots the result. A learning rate around the greatest descent of the subsequent plot should then be chosen to train the model. 

For the 



### References
1. Smith, L.N. (2015) ‘Cyclical Learning Rates for Training Neural Networks’, arXiv, 1506.01186 [Preprint]. Available at: https://arxiv.org/abs/1506.01186 (Accessed: 5 December 2024).
2. 