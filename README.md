# atmaCup11 6th solution


### 必要ライブラリ
* python
* torch
* torchvision
* numpy
* pandas
* scikit-learn
* lightGBM
* seaborn
* pytorch-lightning
* timm
* albumentations

## 準備

./data/raw　にコンペのデータを置いてください．

```
.
├── data
│     ├── raw
│     │    ├── photos
│     │    ├── atmaCup#11_sample_submission.csv
│     │    ├── materials.csv
│     │    ├── techniques.csv
│     │    ├── test.csv
│     │    └── train.csv
│     │

```


### step.1 
* 前処理ファイルの作成

```bash
$ python preprocess.py -f
```
./data/procにdf_proc_train_nn.pklとdf_proc_test_nn.pklが生成されます．


### step.2
* EfficientNet_b1

```bash
$ python train_multitask.py --data-dir [dataset dir path] --arch resnet34 --init-weight-path [step.1で保存した学習済モデルpath]
$ python train_multitask.py --data-dir [dataset dir path] --arch efficientnet_b0 --init-weight-path [step.1で保存した学習済モデルpath]
```

### step.3
* ResNet34, EfficientNet_b0 補助タスク付きモデルのフュージョンモデルを学習

```bash
$ python train__multitask.py --data-dir [dataset dir path] --arch fusion ## engine/multi_task_trainer.py 150, 151行目にstep.2で学習したモデルパスを指定
```

### 推論

```bash
$ python test_multitask.py --data-dir [dataset dir path] --arch fusion --res-dir [step.3で保存した学習済モデル]
```