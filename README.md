# Intro to Binary Classification 
![sample image horseracing](https://wallpaperaccess.com/full/2107537.jpg)

# Description
The goal of this repo is to give a soft introduction to some important concepts and examples on applying Machine Learning towards predicting the outcome of horse races based on attributes connected to each horse. This particular problem would be referred to as `Binary Classification` since the outcome for each horse can only be a win or loss which we can map to 0 or 1 for use in our models.

We will be using popular DS packages like `scikit-learn` for Linear Regression and `Pytorch` for our Neural Net. 

It a favor to a friend with a dataset and keen interest in ML but limited previous experience and is therefore likely best suited towards similar individuals with little to no background in ML.

## Important Principles
There is a fundamental difference between the traditional way of solving a system and the ML approach of learning a mapping between input and output. A known truth is that a feedforward network with a single layer is sufficient to represent any function, but what if our model learns a convoluted model that perfectly maps training data but fails to perform on new unseen data? Our model would merely be memorizing the training data without actually learning!

### Generalization
How well a model learns the underlying distribution from training data and translates that to new samples is referred to as a models `generalization`. To properly evaluate a model it is therefore common to set aside some data in a separate `test set` (and ideally a `validation set` for NN as the tuning of hyperparameters directly on the test set would introduce a bias). Common splits are roughly `80% Train / 20% Test` (`80% Train / 10% Val / 10% Test`) which is done easily e.g. in scikit-learn [`sklearn.model_selection.train_test_split(args)`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html). We then consider both training and test accuracy and wish to:
1. Make training error small
2. Make the gap between training and test error small

### Capacity, overfitting and underfitting
The previous goals are connected to `underfitting` and `overfitting`, which respectively means that a model is not able to learn from the training set (low training accuracy) and that a model fits training data too perfectly without generalizing well to new data. This is controlled by regulating a models capacity. Capacity describes how a model is able to approximate various functions. Too low capacity means a model likely will not be able fit the training set, while too high leaves it prone to overfitting. More specifically, this explains why all problems aren't immediately targeted with the largest most advanced networks one could think of, as being allowed more functions possibly containing a better approximation doesn't guarantee the model will pick it. As a closing remark, the best performing ML algorithm will often be the one with a capacity close to the true complexity of the problem and adequate to the amount of available training data.

![overfitting explained](https://miro.medium.com/max/1400/1*_7OPgojau8hkiPUiHoGK_w.png)

### Is more data needed?
As a rule of thumb, if unable to obtain decent accuracy on the training data the model is not able to properly learn from the available data and more data is not needed. To remedy this, consider a more complex model (i.e. adding more width or depth for NN) until able to overfit on the training data. If however training accuracy is good but testing set accuracy abysmal more data is often beneficial. Finally, if complex models are not even able to learn the training data it is possible that the input needed to predict output is not there and new or higher quality data is needed.

For further reading, see
```
@Book{GoodBengCour16,
  Title                    = {Deep Learning},
  Author                   = {Ian J. Goodfellow and Yoshua Bengio and Aaron Courville},
  Publisher                = {MIT Press},
  Year                     = {2016},

  Address                  = {Cambridge, MA, USA},
  Note                     = {\url{http://www.deeplearningbook.org}}
}
```

### Choice of model
If Neural Nets are 

If 


Determining wether to gather more data

## Linear Regression

## Neural Net
While knowledge of the math behind a neural net would be preferred this isn't strictly needed to start a beginner Neural Net implementation as most modern high-level ML frameworks such as Pytorch provides functions doing it for us. It is however needed to have some understanding of the building blocks that makes up our network.

In short 

init weights
forward pass, 

### Activation function

prediction,
### Loss Function

### Optimizer
Adam
backward propagation, weights

### 

## List of files
```python

```
