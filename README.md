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
| NN overfitted | 0.98 | 0.81 |

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
| Conf vector(8) | 0.56 | 0.77 | 0.65 |
| My lblonly (8) | 0.53 | 0.24 | 0.33 |
| My lblonly w label (8) | 0.55 | 0.39 | 0.46 |
| Original lblonly | 0.52 | 0.88 | 0.66 |


## MIA attacks on Neural Network balanced
| Attack type     | Precision | Recall | F1 score | 
| ----------- | ----------- | ----------- | ----------- |
| Conf vector(16) | 0.52 | 0.76 | 0.62 |
| My lblonly(16) | 0.50 | 0.77 | 0.60 |
| My lblonly w label(16) | 0.51 | 0.94 | 0.66 |
| Original lblonly | 0.52 | 0.73 | 0.61 |

## Figuring out the important things of an attack
1) Using or not shadow models.
2) Use or not a model to learn the threshold.
3) The way we perturb the data.

What are we doing is to compute all the possible triple in order
to detect which part has the greater impact.

### Configurations:
- no shadow, no model, original perturb
- no shadow, no model, our perturbation
- no shadow, si model, original perturb
- no shadow, si model, our perturbation
- si shadow, no model, original perturb
- si shadow, no model, our perturb
- si shadow, si model, original perturb
- si shadow, si model, our perturb

# Bank dataset
## MIA attacks on Neural Network balanced
| Attack type     | Precision | Recall | F1 score | 
| ----------- | ----------- | ----------- | ----------- |
| Conf vector(2) | 0.53 | 0.66 | 0.58 |
| My lblonly(16) | 0. | 0. | 0. |
| My lblonly w label(16) | 0. | 0. | 0. |
| Original lblonly | 0.60 | 0.83 | 0.69 |