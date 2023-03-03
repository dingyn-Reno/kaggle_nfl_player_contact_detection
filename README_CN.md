# kaggle contest: nfl_player_contact_detection

> The solution of the 87th, bronze medal, top 10%.

代码链接：https://github.com/dingyn-Reno/kaggle_nfl_player_contact_detection
数据链接：https://www.kaggle.com/datasets/dyn201885095/testdata/download?datasetVersionNumber=24
提交代码：https://www.kaggle.com/code/yefawu/nfl-3in1?scriptVersionId=120272348
英文页面：

## 运行项目
```bash
python main.py --config config/config_for_train.yaml --device 0
python main.py --config config/config.yaml --device 0
```
## 启动tensorboard
```bash
cd NFL
tensorboard --logdir=work_dir/nfl5/runs --bind_all
```
## 训练和测试
test: 
```bash
python main.py --config config/config.yaml --phase test --save-score True --device 0 --weights ?
```
train: 
```bash
python main.py --config ? --device 
```

## 解决方案

### 第一步：特征工程

- 将测试的label根据提交样例的列表扩充到训练label的维度。
- 根据头盔传感器的数据，截取部分帧
- 处理表格数据
- 根据图像的定位坐标框截取图像
- 对图像进行特征增强处理

### 第二步：模型选择

我们使用了0.5*resnet50+0.5*mobilenet_v3的模型融合方案(0.701)以及0.4*resnet50+0.4*mobilenet_v3+0.2*resnext的模型融合方案(0.701)。

以下模型我们进行了测试但最终没有选用：swin transformer,vit,regnety,efficientnet_b4,hrnet

以下模型被其他队伍使用但我们没有用：resnet3D，xgboost，efficientnet_v2

### 第三步：TTA

我们使用了中值模糊和水平翻转两种TTA策略。

## 总结

作为我们的第一次kaggle比赛，拿到铜牌很值得鼓励，但是这次比赛对很多技巧和方法还不熟悉，比如一些trick的使用上，预训练参数的使用上，以及一些调优策略等。





