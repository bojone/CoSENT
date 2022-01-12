#! -*- coding:utf-8 -*-

import json
import numpy as np
import scipy.stats
from scipy.optimize import minimize
from bert4keras.backend import keras, K
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.snippets import open
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.optimizers import Adam
from tqdm import tqdm
import sys

task_name = sys.argv[1]
assert task_name in ['ATEC', 'BQ', 'LCQMC', 'PAWSX']


def load_data(filename):
    """加载数据（带标签）
    单条格式：(文本1, 文本2, 标签)
    """
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in f:
            l = l.strip().split('\t')
            if len(l) == 3:
                D.append((l[0], l[1], int(l[2])))
    return D


# 基本参数
maxlen = 128 if task_name != 'PAWSX' else 256
batch_size = 32
epochs = 5

# 模型路径
config_path = '/root/kg/bert/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '/root/kg/bert/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '/root/kg/bert/chinese_L-12_H-768_A-12/vocab.txt'

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

# 加载数据集
data_path = '/root/senteval_cn/'
datasets = [
    load_data('%s%s/%s.%s.data' % (data_path, task_name, task_name, f))
    for f in ['train', 'valid', 'test']
]
train_data, valid_data, test_data = datasets


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text1, text2, label) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(
                text1, text2, maxlen=maxlen
            )
            batch_token_ids.append(token_ids)
            batch_segment_ids.append([0] * len(segment_ids))
            batch_labels.append([label])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


# 转换数据集
train_generator = data_generator(train_data, batch_size)
valid_generator = data_generator(valid_data, batch_size)
test_generator = data_generator(test_data, batch_size)

# 构建模型
base = build_transformer_model(config_path, checkpoint_path)
output = keras.layers.Lambda(lambda x: x[:, 0])(base.output)
# output = keras.layers.GlobalAveragePooling1D()(base.output)
output = keras.layers.Dense(units=2, activation='softmax')(output)
model = keras.models.Model(base.inputs, output)

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=Adam(2e-5),
    metrics=['accuracy']
)


def compute_corrcoef(x, y):
    """Spearman相关系数
    """
    return scipy.stats.spearmanr(x, y).correlation


def optimal_threshold(y_true, y_pred):
    """最优阈值的自动搜索
    """
    loss = lambda t: -np.mean((y_true > 0.5) == (y_pred > np.tanh(t)))
    result = minimize(loss, 1, method='Powell')
    return np.tanh(result.x), -result.fun


class Evaluator(keras.callbacks.Callback):
    """保存验证集分数最好的模型
    """
    def __init__(self):
        self.best_accuracy = 0.
        self.best_threshold = 0.

    def on_epoch_end(self, epoch, logs=None):
        spearman, accuracy, threshold = self.evaluate(valid_generator)
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.best_threshold = threshold
            model.save_weights('%s.interact.weights' % task_name)
        print(
            u'spearman: %.5f, accuracy: %.5f, threshold: %.5f, best_accuracy: %.5f\n'
            % (spearman, accuracy, threshold, self.best_accuracy)
        )

    def evaluate(self, data, threshold=None):
        Y_true, Y_pred = [], []
        for x_true, y_true in data:
            Y_true.extend(y_true[:, 0])
            y_pred = model.predict(x_true)[:, 1]
            Y_pred.extend(y_pred)
        Y_true, Y_pred = np.array(Y_true), np.array(Y_pred)
        spearman = compute_corrcoef(Y_true, Y_pred)
        if threshold is None:
            threshold, accuracy = optimal_threshold(Y_true, Y_pred)
        else:
            accuracy = np.mean((Y_true > 0.5) == (Y_pred > threshold))
        return spearman, accuracy, threshold


if __name__ == '__main__':

    evaluator = Evaluator()

    model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        callbacks=[evaluator]
    )
    model.load_weights('%s.interact.weights' % task_name)
    metrics = evaluator.evaluate(test_generator, evaluator.best_threshold)
    metrics = tuple(metrics[:2])
    print(u'test spearman: %.5f, test accuracy: %.5f' % metrics)

else:

    model.load_weights('%s.interact.weights' % task_name)
