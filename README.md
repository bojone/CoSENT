# CoSENT
比Sentence-BERT更有效的句向量方案

## 介绍

- 博客：https://kexue.fm/archives/8847
- 数据：https://github.com/bojone/BERT-whitening/tree/main/chn

## 效果

train训练、test测试：
| | ATEC | BQ | LCQMC | PAWSX | STS-B | Avg |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| BERT+CoSENT | **49.74** | **72.38** | 78.69 | **60.00** | **80.14** | **68.19** |
| Sentence-BERT | 46.36 | 70.36 | **78.72** | 46.86 | 66.41 | 61.74 |
| RoBERTa+CoSENT | **50.81** | **71.45** | **79.31** | **61.56** | **81.13** | **68.85** |
| Sentence-RoBERTa | 48.29 | 69.99 | 79.22 | 44.10 | 72.42 | 62.80 |

NLI训练、test测试：
| | ATEC | BQ | LCQMC | PAWSX | STS-B | Avg |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| BERT+CoSENT | **28.93** | 41.84 | **66.07** | **20.49** | 73.91 | **46.25** |
| Sentence-BERT | 28.19 | **42.73** | 64.98 | 15.38 | **74.88 | 45.23 |
| RoBERTa+CoSENT | 31.84 | **46.65** | **68.43** | **20.89** | **74.37** | **48.43** |
| Sentence-RoBERTa | **31.87** | 45.60 | 67.89 | 15.64 | 73.93 | 46.99 |


## 环境

需要`bert4keras >= 0.10.8`。个人实验环境是tensorflow 1.15 + keras 2.3.1 + bert4keras 0.10.8。

## 其他

- PyTorch版本（非官方）：https://github.com/shawroad/CoSENT_Pytorch
- PyTorch版本（非官方）：https://github.com/xiangking/PyTorch_CoSENT

## 交流
QQ交流群：808623966，微信群请加机器人微信号spaces_ac_cn

