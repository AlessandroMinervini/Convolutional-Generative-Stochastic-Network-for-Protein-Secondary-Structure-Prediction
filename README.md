# Convolutional Generative Stochastic Network for Protein Secondary Structure Prediction
An implementation for Deep Supervised and Convolutional Generative Stochastic Network for Protein Secondary Structure Prediction[1].
Convolutional Generative Stochastic Network is a particular variant of Generative Stochastic Network (Y.Bengio 2013)[2].

## Dataset
The model is evaluated on CullPDB_profile_6133[3] and CB513[4] and are avaiable here: http://www.princeton.edu/~jzthree/datasets/ICML2014/

## Model
![alt text](https://github.com/AlessandroMinervini/Convolutional-Generative-Stochastic-Network-for-Protein-Secondary-Structure-Prediction/blob/master/images/archit.jpg)

## Performance on CullPDB_profile_6133

Q8 Accuracy                |  Loss
:-------------------------:|:-------------------------:
![](https://github.com/AlessandroMinervini/Convolutional-Generative-Stochastic-Network-for-Protein-Secondary-Structure-Prediction/blob/master/images/acc_6133.png)   |  ![](https://github.com/AlessandroMinervini/Convolutional-Generative-Stochastic-Network-for-Protein-Secondary-Structure-Prediction/blob/master/images/loss_6133.png)

## Performance on CB513

Q8 Accuracy                |  Loss
:-------------------------:|:-------------------------:
![](https://github.com/AlessandroMinervini/Convolutional-Generative-Stochastic-Network-for-Protein-Secondary-Structure-Prediction/blob/master/images/acc_cb513.png)   |  ![](https://github.com/AlessandroMinervini/Convolutional-Generative-Stochastic-Network-for-Protein-Secondary-Structure-Prediction/blob/master/images/loss_cb513.png)

## Example of samples generate on fist epoch

![alt text](https://github.com/AlessandroMinervini/Convolutional-Generative-Stochastic-Network-for-Protein-Secondary-Structure-Prediction/blob/master/images/predictions.png)

## References
[1] Jian Zhou, Olga G. Troyanskaya (2014) - Deep Supervised and Convolutional Generative Stochastic Network for Protein Secondary Structure Prediction

[2] Yoshua Bengio, Eric Thibodeau-Laufer, Guillaume Alain, and Jason Yosinski (2013) - Deep Generative Stochastic Networks Trainable by Backprop

[3] Guoli Wang and Roland L. Dunbrack Jr (2003) - PISCES: a protein sequence culling server

[4]  Cuff JA, Barton GJ (1999) - Evaluation and Improvement of Multiple Sequence Methods for Protein Secondary Structure Prediction




