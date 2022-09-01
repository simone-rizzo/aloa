# Membership Inference Attack
- first train the black box model
- second generate the noise dataset with train e test
- then we add the label from the bb on the noise dataset.
- create and train shadow models with the noise labelled dataset.
- train attack model on the data from the shadow models
- evaluate the attack model on the real train and test set of the bb.


## Peculiarities 
- We used disjointed set for training
- Try to change the train,test proporsion (the higher the test the more it will overfit the higher the accuracy of the MIA attack)
- Problem with the DTclassifier with 1,0 confidence.
- Try to use my noise dataset generation with the radius of noise.
- Change the number of shadow models
- Try to use overlapper training set for shadow models.
- Try to overfit and underfit the model to see the effect on the attack.

## Label Only MIA
https://github.com/cchoquette/membership-inference

Given the part of the training set it performs
- binary random perurbation
- continuos random perturbation

## Original LBL only
Source model è il modello trainato su source ds (source training, source test)
Target model è uno shadow model trainato su hold out dataset dal source (target training, target test)

## Differences between my lblonly and the original one
- **the shadow model**: in the original one it uses
 one shadow model for all the classes, instead in my
 implementation i trained multiple shadow models

- **the attack model**: in the original it uses only one attacker model
for all the classes, instead in my implementation i used
one attacker for each class

- **the noise function**: it uses the standard deviation
for gaussian noise on continuos values and bernoulli flipping on 
binary features, instead i used local perturbation on continuos values
and bernoully noise but taking into account also values that are not strictly binary value 0-1
but also 0.25,0.75

- **the attack model**: in the original we find the optimal threshold that
maximize the accuracy, instead in our approach the threshold is automatically
learned by a machine learning model.

- **the training**: in the original paper if the model missclassified a point
the score is automatically 0 and no perturbation is computed. In the calculus of the
score the true label are used, instead in our approach the majority class
of the model is used and if there is a missclassification point we still keep
generating noised examples. 

## Trained models
| Model type     | TR accuracy | Test accuracy | 
| ----------- | ----------- | ----------- | 
| DT | 0.88 | 0.83|
| RF | 1.0 | 0.84 |
| NN | 0.89 | 0.83 |

## MIA attacks on Random forest model
| Attack type     | Precision | Recall | F1 score | 
| ----------- | ----------- | ----------- | ----------- |
| Conf vector | 0.84 | 0.75 | 0.79 |
| My lblonly | 0.79 | 0.71 | 0.75 |
| My lblonly w label | 0.82 | 0.75 | 0.78 |
| Original lblonly | 0.82 | 0.88 | 0.85 |

## MIA attacks on Random forest model balanced DS
| Attack type     | Precision | Recall | F1 score | 
| ----------- | ----------- | ----------- | ----------- |
| Conf vector | 0.57 | 0.75 | 0.65 |
| My lblonly | 0.49 | 0.64 | 0.55 |
| My lblonly w label | 0.56 | 0.42 | 0.48 |
| Original lblonly | 0.53 | 0.89 | 0.66 |

## MIA attacks on Neural Network
| Attack type     | Precision | Recall | F1 score | 
| ----------- | ----------- | ----------- | ----------- |
| Conf vector | 0.81 | 0.49 | 0.60 |
| My lblonly | 0.80 | 0.72 | 0.76 |
| My lblonly w label | 0.74 | 0.05 | 0.10 |
| Original lblonly | 0.81 | 0.76 | 0.78 |