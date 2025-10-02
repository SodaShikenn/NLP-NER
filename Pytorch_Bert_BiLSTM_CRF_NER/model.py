from config import *
import torch.nn as nn
from torchcrf import CRF
import torch
from transformers import BertModel

class Model(nn.Module):

    def __init__(self):
        super().__init__()
        # # 词嵌入层：将词汇ID转换为稠密向量
        # self.embed = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM, WORD_PAD_ID)
        self.bert = BertModel.from_pretrained(BERT_MODEL)

        # 双向LSTM层：捕获序列的前向和后向依赖关系
        self.lstm = nn.LSTM(
            EMBEDDING_DIM,
            HIDDEN_SIZE,
            batch_first=True,
            bidirectional=True,
        )
        self.linear = nn.Linear(2 * HIDDEN_SIZE, TARGET_SIZE)
        self.crf = CRF(TARGET_SIZE, batch_first=True)

    def _get_lstm_feature(self, input, mask):
        # out = self.embed(input)
        out = self.bert(input, mask)[0]
        out, _ = self.lstm(out)
        return self.linear(out)

    def forward(self, input, mask):
        out = self._get_lstm_feature(input, mask)
        return self.crf.decode(out, mask)

    def loss_fn(self, input, target, mask):
        y_pred = self._get_lstm_feature(input, mask)
        return -self.crf.forward(y_pred, target, mask, reduction='mean')

if __name__ == '__main__':
    model = Model()
    input = torch.randint(0, 3000, (100, 50))
    print(len(model(input, None)))