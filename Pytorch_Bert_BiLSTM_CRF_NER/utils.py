from config import *
import torch
from torch.utils import data
import pandas as pd
from seqeval.metrics import classification_report
from transformers import BertTokenizer
from transformers import logging

logging.set_verbosity_warning()


# 加载词表
def get_vocab():
    df = pd.read_csv(VOCAB_PATH, names=['word', 'id'])
    return list(df['word']), dict(df.values)

# 加载标签表
def get_label():
    df = pd.read_csv(LABEL_PATH, names=['label', 'id'])
    return list(df['label']), dict(df.values)

# 定义数据集
class Dataset(data.Dataset):
    # 这个类训练和测试共用，所以定义一个参数来区分加载哪个类
    def __init__(self, type='train', base_len=50):
        super().__init__() # “run the parent class’s setup before adding my own setup.”
        self.base_len = base_len # 定义句子的参考长度，特殊情况再作处理
        sample_path = TRAIN_SAMPLE_PATH if type == 'train' else TEST_SAMPLE_PATH
        self.df = pd.read_csv(sample_path, names=['word', 'label'])
        _, self.word2id = get_vocab()
        _, self.label2id = get_label()
        self.get_points()
        # 初始化Bert
        self.tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)

    ## 切分文本
    # 计算分割点
    def get_points(self):
        self.points = [0]  # 初始化分割点列表，起始点为0
        i = 0              # 当前索引位置

        while True:
            # 检查是否已接近数据末尾
            if i + self.base_len >= len(self.df):
                self.points.append(len(self.df))  # 添加最后一个分割点（数据末尾）
                break

            # 检查目标位置的标签
            if self.df.loc[i + self.base_len, 'label'] == 'O':
                # 标签为'O'，可以在此处安全分割
                i += self.base_len        # 移动到下一个目标位置
                self.points.append(i)     # 添加分割点
            else:
                # 标签不是'O'，向后移动一位继续寻找合适的分割点
                i += 1

    # 文本数字化
    def __len__(self):
        return len(self.points) - 1  # 字段数=分割点数-1

    def __getitem__(self, index):
        # 根据索引提取对应的数据片段
        df = self.df[self.points[index]:self.points[index + 1]]

        # 获取未知词和默认标签的ID
        word_unk_id = self.word2id[WORD_UNK]    # 未知词的ID
        label_o_id = self.label2id['O']         # 'O'标签的ID（作为未知标签的默认值）
        # input = [self.word2id.get(w, word_unk_id) for w in df['word']]
        # 注意：先自己将句子做分词，再转id，避免bert自动分词导致句子长度变化
        input = self.tokenizer.encode(list(df['word']), add_special_tokens=False)
        # 将标签序列转换为ID序列
        target = [self.label2id.get(l, label_o_id) for l in df['label']]

        # return input, target
        # bert要求句子长度不能超过512
        return input[:MAX_POSITION_EMBEDDINGS], target[:MAX_POSITION_EMBEDDINGS]

# 数据校对整理
def collate_fn(batch):
    # 按序列长度降序排列，有助于RNN的packed sequence优化
    batch.sort(key=lambda x: len(x[0]), reverse=True)
    # 获取批次中最长序列的长度
    max_len = len(batch[0][0])
    # 初始化结果列表
    input = []   # 存储填充后的输入序列
    target = []  # 存储填充后的标签序列
    mask = []    # 存储注意力掩码
    # 遍历批次中的每个样本
    for item in batch:
        # 计算当前样本需要填充的长度
        pad_len = max_len - len(item[0])
        # 对输入序列进行填充（使用WORD_PAD_ID）
        input.append(item[0] + [WORD_PAD_ID] * pad_len)
        # 对标签序列进行填充（使用LABEL_O_ID，避免影响损失计算）
        target.append(item[1] + [LABEL_O_ID] * pad_len)
        # 生成注意力掩码（1=真实词汇，0=填充位置）
        mask.append([1] * len(item[0]) + [0] * pad_len)
    # 转换为PyTorch张量并返回 (moved to device in training loop for efficiency)
    return torch.tensor(input), torch.tensor(target), torch.tensor(mask).bool()

def extract(label, text):
    i = 0           # 当前处理位置的索引
    res = []        # 存储提取的实体结果

    # 遍历整个标签序列
    while i < len(label):
        # 检查当前位置是否为实体标签（非'O'）
        if label[i] != 'O':
            # 解析实体标签：'B-PER' -> prefix='B', name='PER'
            prefix, name = label[i].split('-')

            # 初始化实体的起始和结束位置
            start = end = i
            i += 1

            # 向后扫描，寻找同类型的'I-'标签，确定实体的完整边界
            while i < len(label) and label[i] == 'I-' + name:
                end = i     # 更新实体结束位置
                i += 1

            # 提取完整实体：[实体类型, 实体对应的文本片段]
            res.append([name, text[start:end + 1]])
        else:
            # 当前位置为'O'标签，跳过
            i += 1

    return res


def report(y_true, y_pred):
    return classification_report(y_true, y_pred)

if __name__ == '__main__':
    dataset = Dataset()
    loader = data.DataLoader(dataset, batch_size=100, collate_fn=collate_fn)
    print(next(iter(loader)))
