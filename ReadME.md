# 基于多网络融合的药物-靶标相互作用预测算法
设计了一种名为MNDT（Multi-Network Drug-Target Interaction Prediction）的新算法，旨在充分利用来自多种来源的药物和靶标信息，包括但不限于药物的化学结构网络、靶标的序列信息、药物的具体结构信息以及它们之间的互动信息等多元维度，以期综合表征药物和靶标的多重特性。MNDT的核心设计理念在于构建一个多层、多视角的信息融合框架，将不同角度得到的数据信息深度融合，并借助深度学习技术高效挖掘和抽象表达这些信息。
# 依赖包
+ Python 3.8
+ torch 2.1
+ numpy >=1.24
+ scikit_learn
+ rdkit
+ transformers
# 结构
+ README.md：此文件。
+ data：里存放的是论文中使用的三个数据集
+ data_pre.py：数据处理。
+ gat.py：图神经网络。
+ interformer.py：interformer网络。
+ sw_tf.py：swin-transformer网络
+ transformer.py：transformer网络
+ hyperparameter.py：超参数
+ model.py：MNDT 模型结构。
# 运行
~~~
Python main.py
~~~