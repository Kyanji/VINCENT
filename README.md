# VINCENT: Cyber-threat Detection through Vision Transformers and Knowledge Distillation

The repository contains code refered to the work:

_Luca De Rose, Giuseppina Andresini, Annalisa Appice, Donato Malerba_

[VINCENT: Cyber-threat Detection through Vision Transformers and Knowledge Distillation](https://www.sciencedirect.com/science/article/pii/S0167404824002293)
 
Please cite our work if you find it useful for your research and work.


![VINCENT](https://github.com/Kyanji/VINCENT/blob/master/VINCENT.png)

## Code requirements
The code relies on the following python3.7+ libs.
Packages needed are:
* hyperopt==0.2.7
* keras==2.9.0
* keras_cv_attention_models==1.3.9
* matplotlib==3.5.2
* numpy==1.21.5
* opencv_contrib_python==4.7.0.68
* opencv_python==4.5.5.64
* pandas==1.4.4
* scikit_image==0.19.2
* scikit_learn==1.0.2
* scipy==1.9.1
* tensorflow==2.9.3
* tensorflow_addons==0.19.0
* vit_keras==0.1.0
* wandb==0.13.10


## Data
The [DATASETS](https://drive.google.com/file/d/1UeyREoDE28VELDnBgu5bcJ1TauH2Ym-E/view?usp=sharing) used are:
* CICMalDroid20
* CICMalMem22
* NSL-KDD
* UNSW-NB15

* The datasets of images created by using [MAGNETO](https://github.com/Kyanji/MAGNETO/) are uploaded in the folder [IMAGES DATASETS](https://drive.google.com/file/d/1MBMnjk9ipmvQaRe9IgsPTdRZrrGsqO9k/view?usp=sharing)
  
## How to use

The repository contains the following scripts:
* main.py:  script to execute VINCENT 
* config.ini: configuration file
  

## Replicate the experiments
Modify the following code in the config.ini file to change the behaviour of VINCENT.

The models reported in the paper are uploaded in [IMODELS]([https://drive.google.com/file/d/1MBMnjk9ipmvQaRe9IgsPTdRZrrGsqO9k/view?usp=sharing](https://drive.google.com/file/d/1_lD4-SC35hzDPinY_D6DkdOI3k_Z_YTL/view?usp=sharing))

# Parameters
```ini
[SETTINGS]
UseMagnetoEncoding=False : Convert tabular data to Images or load dataset
Dataset = NSL : MALMEM|MALDROID|NSL|UNSW
TrainVIT=False : Train VIT(Teacher) or load the model if false from the VIT_Teacher_Path
TrainVINCENT=False : Train VINCENT(STUDENT) or load the model if false from the VINCENTPath

[VIT_SETTINGS] : Settings of the VIT (Teacher)
[MAGNETO] : Settings about Magneto encoding (e.g. Image size)
[DISTILLATION] : Settings for the VINCENT Training

[**DATASET**]
tabular_dataset_path=..\..\dataset\malmem\ : path of the tabular dataset
tabular_trainfile=train_split_macro_minmaxdeleted.csv : tabular training file
tabular_testfile=test_split_macro_minmaxdeleted.csv : tabular testing file
classification=Family_int : Classification label for the tabular dataset

trainName=train_8x8_MI.pickle : path of the pickle train images
ytrainName=Ytrain_multi.pickle : path of the pickle train label
testName=test_8x8_MI.pickle  : path of the pickle test images
ytestName=Ytest_multi.pickle : path of the pickle test label

toBinaryMap={"0": 0, "1": 1, "2": 1, "3": 1} : used by Magneto to encode the dataset using binary labels
OutputDirMagneto = MAGNETO_out\malmem\ : Output files for Magneto
OutputDir= .\res\malmem\  :  Output files for VINCENT
VIT_Teacher_Path=./res/malmem/2023-04-06-14-04-51.h5  :  Teacher model
VINCENTPath=./res/malmem/models/PESI.tf :  VINCENT model
Baseline=./res/malmem/cnn2023-05-15-12-54-54/20.tf   :  Baseline (CNN) model


```








