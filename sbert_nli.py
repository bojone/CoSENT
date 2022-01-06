#! -*- coding:utf-8 -*-

import json, glob
import numpy as np
import scipy.stats
from bert4keras.backend import keras, K
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.snippets import open
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.optimizers import Adam
from tqdm import tqdm


def load_sim_data(filename):
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


def load_nli_data(filename):
    """加载数据（带标签）
    单条格式：(文本1, 文本2, 标签)
    """
    D = []
    with open(filename, encoding='utf-8') as f:
        labels = ['contradiction', 'neutral', 'entailment']
        for l in f:
            l = json.loads(l)
            if l['gold_label'] not in labels:
                continue
            text1, text2 = l['sentence1'], l['sentence2']
            label = labels.index(l['gold_label'])
            D.append((text1, text2, label))
    return D


# 基本参数
maxlen = 64
batch_size = 64
epochs = 10

# 模型路径
config_path = '/root/kg/bert/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '/root/kg/bert/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '/root/kg/bert/chinese_L-12_H-768_A-12/vocab.txt'

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

# 加载数据集
train_data = []
for f in glob.glob('/root/cnsd/cnsd-*/*.jsonl'):
    train_data += load_nli_data(f)

data_path = '/root/senteval_cn/'
valid_datas = [
    load_sim_data('%s%s/%s.valid.data' % (data_path, task_name, task_name))
    for task_name in ['ATEC', 'BQ', 'LCQMC', 'PAWSX', 'STS-B']
]
test_datas = [
    load_sim_data('%s%s/%s.test.data' % (data_path, task_name, task_name))
    for task_name in ['ATEC', 'BQ', 'LCQMC', 'PAWSX', 'STS-B']
]


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text1, text2, label) in self.sample(random):
            for text in [text1, text2]:
                token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
                batch_token_ids.append(token_ids)
                batch_segment_ids.append(segment_ids)
                batch_labels.append([label])
            if len(batch_token_ids) == self.batch_size * 2 or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


# 转换数据集
train_generator = data_generator(train_data, batch_size)
valid_generators = [data_generator(D, batch_size) for D in valid_datas]
test_generators = [data_generator(D, batch_size) for D in test_datas]


def merge(inputs):
    """向量合并：a、b、|a-b|拼接
    """
    a, b = inputs[::2], inputs[1::2]
    o = K.concatenate([a, b, K.abs(a - b)], axis=1)
    return K.repeat_elements(o, 2, 0)


# 构建模型
base = build_transformer_model(config_path, checkpoint_path)
output = keras.layers.Lambda(lambda x: x[:, 0])(base.output)
# output = keras.layers.GlobalAveragePooling1D()(base.output)
encoder = keras.models.Model(base.inputs, output)

output = keras.layers.Lambda(merge)(output)
output = keras.layers.Dense(units=3, activation='softmax')(output)
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


def l2_normalize(vecs):
    """l2标准化
    """
    norms = (vecs**2).sum(axis=1, keepdims=True)**0.5
    return vecs / np.clip(norms, 1e-8, np.inf)


class Evaluator(keras.callbacks.Callback):
    """保存验证集分数最好的模型
    """
    def __init__(self):
        self.best_val_score = 0.

    def on_epoch_end(self, epoch, logs=None):
        val_scores = [self.evaluate(D) for D in valid_generators]
        val_score = np.mean(val_scores)
        if val_score > self.best_val_score:
            self.best_val_score = val_score
            model.save_weights('nli.sbert.weights')
        template = 'ATEC: %.5f, BQ: %.5f, LCQMC: %.5f, PAWSX: %.5f, STS-B: %.5f'
        print(
            u'%s, val_score: %.5f, best_val_score: %.5f\n' %
            (template % tuple(val_scores), val_score, self.best_val_score)
        )

    def evaluate(self, data):
        Y_true, Y_pred = [], []
        for x_true, y_true in data:
            Y_true.extend(y_true[::2, 0])
            x_vecs = encoder.predict(x_true)
            x_vecs = l2_normalize(x_vecs)
            y_pred = (x_vecs[::2] * x_vecs[1::2]).sum(1)
            Y_pred.extend(y_pred)
        return compute_corrcoef(Y_true, Y_pred)


if __name__ == '__main__':

    evaluator = Evaluator()

    model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=5000,
        epochs=epochs,
        callbacks=[evaluator]
    )
    model.load_weights('nli.sbert.weights')
    test_scores = [evaluator.evaluate(D) for D in test_generators]
    template = 'ATEC: %.5f, BQ: %.5f, LCQMC: %.5f, PAWSX: %.5f, STS-B: %.5f'
    print(template % tuple(test_scores))

else:

    model.load_weights('nli.sbert.weights')
