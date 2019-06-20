# Convolutional Generative Stochastic Network for Protein Secondary Structure Prediction
An implementation for Deep Supervised and Convolutional Generative Stochastic Network for Protein Secondary Structure Prediction[1].
Convolutional Generative Stochastic Network is a particular variant of Generative Stochastic Network (Y.Bengio 2013)[2].

## Dataset
The model is evaluated on CullPDB_profile_6133[3] and CB513[4] and are avaiable here: http://www.princeton.edu/~jzthree/datasets/ICML2014/

## Model
![alt text](https://github.com/AlessandroMinervini/Convolutional-Generative-Stochastic-Network-for-Protein-Secondary-Structure-Prediction/blob/master/images/archit.jpg)

Model is structured by:
#### Encoding phase:
- 2 convolutional layers (activaction functions: tanh) and 1 mean pooling layer.

#### Decoding phase:
- 2 convolutional transposed layers, and 1 upsampling layer.

### Walkbacks
Model generates for every input 12 samples, where the last is the prediction. Moreover generates 12 latent state(H_0..H_1_1).

### Weights initialization
Weights are initializated according to Xavier Inizializer [5] and are shared tvhrough the network.

### Regularization
To regularize is used L2 regularization.

### Loss function
Loss function is Binary cross entropy, and Adam the optimizer.

## Performance on CullPDB_profile_6133

Q8 Accuracy                |  Loss
:-------------------------:|:-------------------------:
![](https://github.com/AlessandroMinervini/Convolutional-Generative-Stochastic-Network-for-Protein-Secondary-Structure-Prediction/blob/master/images/acc_6133.png)   |  ![](https://github.com/AlessandroMinervini/Convolutional-Generative-Stochastic-Network-for-Protein-Secondary-Structure-Prediction/blob/master/images/loss_6133.png)

## Performance on CB513

Q8 Accuracy                |  Loss
:-------------------------:|:-------------------------:
![](https://github.com/AlessandroMinervini/Convolutional-Generative-Stochastic-Network-for-Protein-Secondary-Structure-Prediction/blob/master/images/acc_cb513.png)   |  ![](https://github.com/AlessandroMinervini/Convolutional-Generative-Stochastic-Network-for-Protein-Secondary-Structure-Prediction/blob/master/images/loss_cb513.png)

## Example of samples generate on first epoch

![alt text](https://github.com/AlessandroMinervini/Convolutional-Generative-Stochastic-Network-for-Protein-Secondary-Structure-Prediction/blob/master/images/predictions.png)

## References
[1] Jian Zhou, Olga G. Troyanskaya (2014) - Deep Supervised and Convolutional Generative Stochastic Network for Protein Secondary Structure Prediction

[2] Yoshua Bengio, Eric Thibodeau-Laufer, Guillaume Alain, and Jason Yosinski (2013) - Deep Generative Stochastic Networks Trainable by Backprop

[3] Guoli Wang and Roland L. Dunbrack Jr (2003) - PISCES: a protein sequence culling server

[4]  Cuff JA, Barton GJ (1999) - Evaluation and Improvement of Multiple Sequence Methods for Protein Secondary Structure Prediction

[5] Xavier Glorot and Yoshua Bengio (2010) - Understanding the difficulty of training deep feedforward neural networks. International conference on artificial intelligence and statistics.


