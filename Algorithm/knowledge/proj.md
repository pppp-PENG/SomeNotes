# 项目

### 腾讯广告算法大赛

##### 数据集

1. 用户行为序列 jsonl 文件
    - 每一行为一个用户 token 序列（兴趣 emb + 交互记录）
    - 每一个记录包括 user_id, item_id, user_feature, item_feature, action_type, timestamp
2. item feat json 文件
    - 记录所有 item 的 feature，便于负采样
3. 体量
    - 100 万 user、400 万 item


##### 数据预处理

1. 对 timestamp 进行分桶，计算与上一次点击的时间差，按秒分时日周月年等分成不同的桶（起始值单独一个桶）；

    额外采取特征——当前点击距离 2024-10-01（起始时间）的时间，分为日周月三个时间差特征

2. 采用left padding，缺失值用 0 填充

4. 负采样全局采取（比只采样曝光未点击的效果好）

5. 将所有特征都在 collate_fn 整合成 tensor 再输出（充分利用多进程提高 gpu 利用率）

##### 模型

1. Embedding 层：将 item 特征 embedding，concat + Linear + ReLU + Norm（也是推理使用的 item 空间）
2. 特征融合层：将 user 和时间戳的相关特征 embedding 后，和 item 的 embedding 一起 concat + Linear ReLU + Norm
3. 模型层：通过多层 HSTU 输出预测曝光 embedding，再通过一个 MLP 输出预测点击 embedding，HSTU 的改进如下：
    - 投影部分使用 SiLU 激活
    - 将 HSTU 注意力分数的 Norm 部分改为除以 $(L * \sqrt{d})$ 再接一个 ReLU，效果更好
    - 最后，将门控前的 attn 结果与门控之后的结果 y 与 U 三者一起 concat + Linear + ReLU + Norm 才输出结果

##### 训练

1. 训练验证集比为 99:1
2. 采用 infoNCE + triplet loss
    - 其中 infoNCE loss 的正负样本比为 1:B\*L（预测的 logits 形状为 [B\*L, D]）
    - infoNCE loss 的 temperature 为 0.1，triplet loss 的 margin 为 0.1











# 八股

### 召回

##### UserCF

##### ItemCF

##### 向量召回

##### 矩阵分解

##### 双塔

##### FM 召回

### 排序

##### FM/DeepFM

##### Wide & Deep

##### DCN

##### AutoInt

##### DIN

##### 双塔

##### 启发式重排
