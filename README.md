# Paddle 分布式训练及CTR预估模型应用

原始models：[PaddleRec-Ctr](https://github.com/PaddlePaddle/models/tree/develop/PaddleRec/ctr)

## 数据准备

### 数据说明

- 数据来源：Kaggle公司举办的[展示广告竞赛](https://www.kaggle.com/c/criteo-display-ad-challenge/)中所使用的Criteo数据集。该数据包含数百万展示广告的特征值和点击反馈，目的是对点击率（CTR）的预测做基准预测。

- 数据背景：Criteo是在线效果类数字营销厂商，于2005年在法国巴黎成立，目前的核心业务是重定向广告（retargeting）。Criteo在全球范围内共有31间办事处，有6间位于欧洲，有5间位于北美，有1间在巴西，在亚太地区总共有5间办事处。Criteo是在线效果类展示广告厂商于2014年5月13日宣布启动在中国的业务和运营，并将北京设为中国区总部所在地。Criteo的核心产品主要包括访客广告、流失客户广告、移动应用内效果型广告和AD-X 移动广告跟踪分析产品等。Criteo拥有世界领先的自主学习式推荐引擎和预测引擎，能够通过其对于市场的洞察提供可评估的结果，因而能够在正确的时间通过推送广告，将对的产品推荐给对的用户。并且，随着每一条广告的交付，Criteo的引擎在预测和推荐方面的精确性也不断提高。

- 数据格式：

      - 格式：<label> <integer feature 1>  <integer feature 13> <categorical feature 1> ... <categorical feature 26>  。共计39个特征，13个数值特征（int），26个类别特征。若value为空值，则为空白

  - 训练数据：train.txt：Criteo 公司在七天内的部分流量。每行对应的是Critio的展示广告，第一列代表该广告是否被点击。我们对正样本（已点击）的和负样本（未点击）均做了子采样来减少数据量。类别特征的值已经过哈希处理为64位来进行脱敏。特征的语义没有公开，并且有些特征有缺失值。行按照时间排序。

    - 示例：

  | label | f1   | f2   | f3   | f4   | f5    | f6   | f7   | f8   | f9   | f10  | f11  | f12  | f13  | f14      | f15      | f16      | f17      | f18      | f19      | f20      | f21      | f22      | f23      | f24      | f25      | f26      | f27      | f28      | f29      | f30      | f31      | f32      | f33      | f34      | f35      | f36      | f37      | f38      | f39      |
  | ----- | ---- | ---- | ---- | ---- | ----- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- |
  | 0     | 1    | 1    | 5    | 0    | 1382  | 4    | 15   | 2    | 181  | 1    | 2    |      | 2    | 68fd1e64 | 80e26c9b | fb936136 | 7b4723c4 | 25c83c98 | 7e0ccccf | de7995b8 | 1f89b562 | a73ee510 | a8cd5504 | b2cb9c98 | 37c9c164 | 2824a5f6 | 1adce6ef | 8ba8b39a | 891b62e7 | e5ba7672 | f54016b9 | 21ddcdc9 | b1252a9d | 07b5194c |          | 3a171ecb | c5c50484 | e8b83407 | 9727dd16 |
  | 0     | 2    | 0    | 44   | 1    | 102   | 8    | 2    | 2    | 4    | 1    | 1    |      | 4    | 68fd1e64 | f0cf0024 | 6f67f7e5 | 41274cd7 | 25c83c98 | fe6b92e5 | 922afcc0 | 0b153874 | a73ee510 | 2b53e5fb | 4f1b46f3 | 6.23E+11 | d7020589 | b28479f6 | e6c5b5cd | c92f3b61 | 07c540c4 | b04e4670 | 21ddcdc9 | 5840adea | 60f6221e |          | 3a171ecb | 43f13e8b | e8b83407 | 731c3655 |
  | 1     | 1    | 4    | 2    | 0    | 0     | 0    | 1    | 0    | 0    | 1    | 1    |      | 0    | 68fd1e64 | 2c16a946 | 503b9dbc | e4dbea90 | f3474129 | 13718bbd | 38eb9cf4 | 1f89b562 | a73ee510 | 547c0ffe | bc8c9f21 | 60ab2f07 | 46f42a63 | 07d13a8f | 18231224 | e6b6bdc7 | e5ba7672 | 74ef3502 |          |          | 5316a17f |          | 32c7478e | 9117a34a |          |          |



  - 测试数据：test.txt：测试集于训练集的计算方式相同，但对应的是训练集时间段的后一天的事件。并且第一列（label）已被移除。

    - 示例：

    | label | f1   | f2   | f3   | f4   | f5   | f6   | f7   | f8   | f9   | f10  | f11  | f12  | f13  | f14      | f15      | f16      | f17      | f18      | f19      | f20      | f21      | f22      | f23      | f24      | f25      | f26      | f27      | f28      | f29      | f30      | f31      | f32      | f33      | f34      | f35      | f36      | f37      | f38      | f39      |
    | ----- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- |
    |       |      | 29   | 50   | 5    | 7260 | 437  | 1    | 4    | 14   |      | 1    | 0    | 6    | 5a9ed9b0 | a0e12995 | a1e14474 | 08a40877 | 25c83c98 |          | 964d1fdd | 5b392875 | a73ee510 | de89c3d2 | 59cd5ae7 | 8d98db20 | 8b216f7b | 1adce6ef | 78c64a1d | 3ecdadf7 | 3486227d | 1616f155 | 21ddcdc9 | 5840adea | 2c277e62 |          | 423fab69 | 54c91918 | 9b3e8820 | e75c9ae9 |
    |       | 27   | 17   | 45   | 28   | 2    | 28   | 27   | 29   | 28   | 1    | 1    |      | 23   | 68fd1e64 | 960c983b | 9fbfbfd5 | 38c11726 | 25c83c98 | 7e0ccccf | fe06fd10 | 062b5529 | a73ee510 | ca53fc84 | 67360210 | 895d8bbb | 4f8e2224 | f862f261 | b4cc2435 | 4c0041e5 | e5ba7672 | b4abdd09 | 21ddcdc9 | 5840adea | 36a7ab86 |          | 32c7478e | 85e4d73f | 010f6491 | ee63dd9b |
    |       | 1    | 1    | 19   | 7    | 1    | 3    | 1    | 7    | 7    | 1    | 1    |      | 2    | 09ca0b81 | 8947f767 | a87e61f7 | c4ba2a67 | 25c83c98 | 7e0ccccf | ce6020cc | 062b5529 | a73ee510 | b04d3cfe | 70dcd184 | 899eb56b | aca22cf9 | b28479f6 | a473257f | 88f592e4 | d4bb7bd8 | bd17c3da | 1d04f4a4 | a458ea53 | 82bdc0bb |          | 32c7478e | 5bdcd9c4 | 010f6491 | cca57dcc |

    

### 数据处理

- 下载数据

  ```bash
  cd data && ./download.sh && cd ..
  ```
  
- 数据读取

  - code：reader.py
  - 原始数据中前13个feature为int型，通过reader.py将其做了数据归一化处理为float型，避免过大和过小的数据在模型训练中的影响。
  
  ```
   .── CriteoDataset
  │
  ├── train
  │
  ├── test
  │
  ├── infer
  ```

  

## 模型训练
### 网络结构

- code: network_conf.py (只用到ctr_dnn_model) 详细讲解

### 训练方式
#### 单机训练

```bash
python train.py \
        --train_data_path data/raw/train.txt \
        2>&1 | tee train.log
```

#### 分布式训练

```bash
sh cluster_train.sh
```

注：batch_size由默认的1000修改为64，可提高auc

### 训练结果

- 单机训练

      - 速度太慢，迭代到第1轮batch= 4919时就停住了

- 分布式训练

      - 设置：2pserver、2trainer

      - 训练日志：alldata/log/trainer0.log 、alldata/log/trainer1.log 

      - 训练结果：

    ```bash
    2019-05-11 08:34:19,678-INFO: TRAIN --> pass: 9 batch: 2577 loss: 0.467225006104 auc: 0.787909292672, batch_auc: 0.797377570934
    pass_id: 0, pass_time_cost: 3150.447569
    pass_id: 1, pass_time_cost: 3177.322331
    pass_id: 2, pass_time_cost: 3174.676812
    pass_id: 3, pass_time_cost: 3209.558880
    pass_id: 4, pass_time_cost: 3134.910369
    pass_id: 5, pass_time_cost: 3202.956675
    pass_id: 6, pass_time_cost: 3169.575809
    pass_id: 7, pass_time_cost: 3210.294044
    pass_id: 8, pass_time_cost: 3039.102302
    pass_id: 9, pass_time_cost: 3036.933163
    ```





## 模型预测
### 预测方式
```bash
python infer.py \
        --model_path models/pass-0/ \
        --data_path data/raw/valid.txt
```

### 预测结果：

- log：alldata/log/infer.txt

```bash
2019-05-13 09:35:49,177-INFO: TEST --> batch: 4500 loss: [0.46127334] auc: [0.78797872]
```


## 实验对比

### 原始数据情况

|        | label | 数量     | 比例       |
| ------ | ----- | -------- | ----------  |
| 负样本 | 0     | 34095179 | 0.74377662 |
| 正样本 | 1     | 11745438 | 0.25622338 |

###实验数据：

- mini-demo：1%的全量数据
- demo：10%的全量数据
- raw：全量数据

### BaseLine实验    
- 数据说明： mini-data(1%全量数据)

- 网络配置：一层网络，batch_size =1000

- 修改方式：

    - 在network_conf.py  第151行修改,如下如所示，将input=fc3 修改为input=fc1

    - 在cluster_train.sh 里第35行和第47行将batch_size=64  修改为batch_size=1000

- 运行方式：
    - 修改完以上两个文件后，执行：sh cluster_train.sh
- 输出日志：
    - pserver：pserver0.log、pserver1.log	参数服务器的输出日志
    - trainer：trainer0.log、trainer1.log        每个trainer的输出日志 可以通过 tail -f trainer0.log -n 999 查看输出结果

- 实验效果
    - 训练时间：33s（一轮迭代）
    - auc：0.50234167

### 优化实验（一）：优化网络层
- 数据说明： mini-data(1%全量数据)

- 网络配置：三层网络，batch_size =1000

- 修改方式：

    - 在network_conf.py  第151行修改,如下如所示，将input=fc1 修改为input=fc3

    - 在cluster_train.sh 里第35行和第47行将batch_size=64  修改为batch_size=1000

- 运行方式：
    - 修改完以上两个文件后，执行：sh cluster_train.sh
- 输出日志：
    - pserver：pserver0.log、pserver1.log	参数服务器的输出日志
    - trainer：trainer0.log、trainer1.log        每个trainer的输出日志 可以通过 tail -f trainer0.log -n 999 查看输出结果

- 实验效果
    - 训练时间：35s（一轮迭代）
    - auc：0.54893279

### 优化实验（二）：调整batch_size
- 数据说明： mini-data(1%全量数据)

- 网络配置：三层网络，batch_size =64

- 修改方式：

    - 在network_conf.py  第151行修改,如下如所示，将input=fc1 修改为input=fc3

    - 在cluster_train.sh 里第35行和第47行将batch_size=1000  修改为batch_size=64

- 运行方式：
    - 修改完以上两个文件后，执行：sh cluster_train.sh
- 输出日志：
    - pserver：pserver0.log、pserver1.log	参数服务器的输出日志
    - trainer：trainer0.log、trainer1.log        每个trainer的输出日志 可以通过 tail -f trainer0.log -n 999 查看输出结果

- 实验效果
    - 训练时间：103s（一轮迭代）
    - auc：0.74322927

### 优化实验（三）：增加数据集
- 数据说明： 全量数据

- 网络配置：三层网络，batch_size =64，数据量由10%数据（demo_data）扩充到全量数据

- 修改方式：

    - 在network_conf.py  第151行修改,如下如所示，将input=fc1 修改为input=fc3

    - 在cluster_train.sh 里第35行和第47行将batch_size=1000  修改为batch_size=64
    - 在cluster_train.sh 里将连个pserver和两个trainer的train_data_path地址修改为raw_data的地址，如下图所示，注意：一共需要修改四个地址

- 运行方式：
    - 修改完以上两个文件后，执行：sh cluster_train.sh
- 输出日志：
    - pserver：pserver0.log、pserver1.log	参数服务器的输出日志
    - trainer：trainer0.log、trainer1.log        每个trainer的输出日志 可以通过 tail -f trainer0.log -n 999 查看输出结果

- 实验效果
    - 训练时间：3150s（一轮迭代）
    - auc：0.81093872


## 优化实验对比结果
- 表格1:对mini_demo(1%全量数据)做训练，采取不同的优化方式，得到最后的优化方案的实验结果

| 评估      | batch_size | batch_1000 | batch_1000 | batch_64    | batch_64   |
| --------- | ---------- | ---------- | ---------- | ----------- | ---------- |
| 优化方式  | 评估       | 一层网络   | 三层网络   | 一层网络    | 三层网络   |
| mini_demo | time       | 33s        | 35s        | 97s         | 103s       |
|           | auc        | 0.50234167 | 0.54893279 | 0.721332392 | 0.74322927 |

- 表格2: 增加数据集对模型精度的提升

|      | batch_size | time  | auc        |
| ---- | ---------- | ----- | ---------- |
| demo | 64         | 1133s | 0.73777626 |
| 全量 | 64         | 3150s | 0.81093872 |

## 优化方案总结
由以上两个表格可知：

- batch 大小的改变：在数据集和其他参数不变的情况下，batch_size由1000变为64.可以提升20%的模型精度
- 网络结构的改变：在数据集和batch_size等参数不变的情况，由一层网络变为三层网络结构，大约可提升2～4%的模型精度
- 数据集的改变：由demo数据（10%全量数据）扩充到全量数据，采用同样的batch_size，同样的迭代次数和其他超参数，大约可提升7%的精度



















