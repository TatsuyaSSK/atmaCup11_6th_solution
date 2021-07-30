# atmaCup11 6th solution

atmaCup11の6位解法のコードです．
解法の詳細は[discussion](https://www.guruguru.science/competitions/17/discussions/462f4199-66aa-4b40-8585-09d2b296d5e8/)に記載しています．
シングルモデルで一番スコアの良かったEfficientNet-B1とstackingの際に利用したResNet18の学習・推論を行うコードになっています．


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

`data/raw/`にコンペのデータを置いてください．

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

## 実行手順
以下の手順はすべて`easy_gold/`ディレクトリで行ってください．


### step.1 
* 前処理ファイルの作成

```bash
$ python preprocess.py -f
```
`data/proc/`に`df_proc_train_nn.pkl`と`df_proc_test_nn.pkl`が生成されます．


### step.2
* EfficientNet-B1による学習と推論

```bash
$ python train.py -model efficientnet_b1 -lr 0.001 -tta 5 -img_size 512  -ep 1000 -es 200 -batch 32
```
`data/submission/`に`[実行時年月日-時分秒]_multiLabelNet--[score]--_submission.csv`と`[実行時年月日-時分秒]_multiLabelNet--[score]--_oof.csv`が生成されます．


### step.3
* ResNet18による学習と推論

```bash
$ python train.py -model resnet18 -lr 0.001 -tta 5 -img_size 512  -ep 1000 -es 200 -batch 32
```
`data/submission/`に`[実行時年月日-時分秒]_ResNet_Wrapper--[score]--_submission.csv`と`[実行時年月日-時分秒]_ResNet_Wrapper--[score]--_oof.csv`が生成されます．


### step.4
* lightGBMによるstacking

```bash
$ mkdir ../data/submission/stack_dir
$ cp ../data/submission/*.csv ../data/submission/stack_dir
```
`data/submission/`の中に`stack_dir`という名前のディレクトリを作成し，step2, 3で生成されたcsvをすべてコピーします．


```bash
$ python train.py -m stack -stack_dir stack_dir -lr 0.001 -ep 1000 -es 200 -f 15
```

`data/submission/`に提出ファイルである`[実行時年月日-時分秒]_SimpleStackingWrapper--[score]--_submission.csv`が生成されます．