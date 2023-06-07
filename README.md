# ALOA (Agnostic Label Only attack)
This is the repo containing the implementation of:
- **original MIA** (Membership Inference Attack): located in core/confidence_attack.py
- **LBLONLY (original Label only attack)**: located at core/lblonly_attacks.py
- **ALOA (the new attack)**: located at core/lblonly_attacks.py

The main script to execute the attacks is the file:  
**core/lblonly_attacks.py**  

It has various parameters where one is the settings, meaning how we want to configure the attack.
Below a table of explaination of how to set the settings parameter.

### Settings meaning:
| Shadow | Model | Our perturb | 
| ----------- | ----------- | ----------- |
| 0 | 0 | 0 |
| 0 | 0 | 1 |
| 0 | 1 | 0 | 
| 0 | 1 | 1 | 
| 1 | 0 | 0 | 
| 1 | 0 | 1 | 
| 1 | 1 | 0 | 
| 1 | 1 | 1 |
- To use the original LBLONLY use the [0,0,0] setting.
- To use the ALOA use the [0,0,1] setting.

## Repo folders explaination
- **bboxes**: contains the classes for DecisionTree, NeuralNetworkand
 RandomForest classifier black boxes. It works with any possible
  implementations of classifier the important is to extend the class bb_wrapper
  and implement the abtstract methods.  
  
- **core**: contains the main scripts for executing the attacks.
- **data**: contains the 3 datasets(Adult, Bank and Synth). Each of them contains
different csv files. The _data is the raw dataset. The adult_original is the preprocessed
dataset before being used into the models. The train and test are the actual data used to 
train the models while the shadow is a disjointed part of the dataset used only for training
the shadow models.
- **dataset_loading**: contains all the code to load and preprocess the datasets.
- **generate_noise_dataset**: contains the scripts to generate the shadow data from 
a given dataset.
- **models**: contains all the code for training different models overfitted and regularized.
- **results**: contains all the results of the attacks.
- **server_scripts**: contains some script
- **trepan_explainers**: contains the part of training TREPAN explainers for each model. Inside we
have the folder explainers which contains all the models(TREPAN DTs) and performances. The script trepan_main.py
is the script used to train the explainers, it contains also the part of mitigation of the attack
by selecting the propres trade_off_score.

