# kaggle contest: nfl_player_contact_detection

> The solution of the 87th, bronze medal, top 10%.

project：https://github.com/dingyn-Reno/kaggle_nfl_player_contact_detection

Data：https://www.kaggle.com/datasets/dyn201885095/testdata/download?datasetVersionNumber=24

Submit code：https://www.kaggle.com/code/yefawu/nfl-3in1?scriptVersionId=120272348

Chinese page：https://github.com/dingyn-Reno/kaggle_nfl_player_contact_detection/blob/main/README_CN.md

## Run the project
```bash
python main.py --config config/config_for_train.yaml --device 0
python main.py --config config/config.yaml --device 0
```
## Start tensorboard
```bash
cd NFL
tensorboard --logdir=work_dir/nfl5/runs --bind_all
```
## Train and Test
test: 
```bash
python main.py --config config/config.yaml --phase test --save-score True --device 0 --weights ?
```
train: 
```bash
python main.py --config ? --device 
```

## Solution

### Step 1: Feature Engineering

- Expand the test label to the dimension of the training label according to the list of submitted samples.
- According to the data of the helmet sensor, intercept keyframes.
- Processing table data
- Capture the image according to the positioning coordinate box of the image.
- Feature enhancement of the image.

### Step 2: Model selection

We used the model fusion scheme of 0.5resnet50+0.5mobilenet_v3 (0.701) and the model fusion scheme of 0.4resnet50+0.4mobilenet_v3+0.2*resnext (0.701).

We tested the following models but did not choose them in the end: swin transformer, vit, regnety, efficientnet_b4, hrnet

The following models are used by other teams, but we don't use them: resnet3D，xgboost，efficientnet_v2

### Step 3: TTA

We used two TTA strategies: median blur and horizontal flip.

## Summary

As our first Kaggle competition, it is very encouraging to win the bronze medal, but this competition is not familiar with many skills and methods, such as the use of some tricks, the use of pre-training parameters, and some tuning strategies.

Thanks to team members Jinrong Zhang and Sifan Zhang for their efforts.



