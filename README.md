# Intro to Binary Classification 
![sample image horseracing](https://wallpaperaccess.com/full/2107537.jpg)

# Description
This repo is meant to give a soft introduction to some important concepts and examples on applying Machine Learning towards predicting the outcome of horse races based on attributes connected to each horse. We will be using popular DS packages like scikit-learn and Pytorch for our models. This particular problem would be referred to as `Binary Classification` since the outcome for each horse can only be a win or loss which we can map to 0 or 1 for use in our models.

It is meant as a favor to a friend with a dataset and keen interest in ML but limited previous experience and is therefore likely best suited towards similar individuals with little to no background in ML.

## Important Principles
There is a fundamental difference between the traditional way of solving a system and the ML approach of learning a mapping between input and output. A known truth is that a feedforward network with a single layer is sufficient to represent any function, but what if our model learns a convoluted model that perfectly maps training data but fails to perform on new unseen data? Our model would merely be memorizing the training data without actually learning!

How well a model learns the underlying distribution from training data and translates that to new samples is referred to as a models `generalization`. To properly evaluate a model it is therefore common to set aside some data in a separate `test set` (and ideally a `validation set` for NN as the tuning of hyperparameters directly on the test set would introduce a bias). Common splits are roughly `80% Train / 20% Test` (`80% Train / 10% Val / 10% Test`) which is done easily e.g. in scikit-learn [`sklearn.model_selection.train_test_split(args)`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html). We then consider both training and test accuracy and wish to:
1. Make training error small
2. Make the gap between training and test error small

Capacity, overfitting and underfitting


Cross validation

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
![overfitting explained](https://miro.medium.com/max/1400/1*_7OPgojau8hkiPUiHoGK_w.png)

Determining wether to gather more data

## Linear Regression

# Neural Net

## List of files
