import torch.nn as nn
from config import *
from torchcrf import CRF
import torch

class Model(nn.Module):
    
    def __init__(self):
        super().__init__()
        # 词嵌入层：将词汇ID转换为稠密向量
        self.embed = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM, WORD_PAD_ID)
        # 双向LSTM层：捕获序列的前向和后向依赖关系
        self.lstm = nn.LSTM(
            EMBEDDING_DIM,
            HIDDEN_SIZE,
            batch_first=True,
            bidirectional=True,
        )
        self.linear = nn.Linear(2 * HIDDEN_SIZE, TARGET_SIZE)
        self.crf = CRF(TARGET_SIZE, batch_first=True)

    def _get_lstm_feature(self, input):
        # Step 1: 词嵌入 [batch_size, seq_len] -> [batch_size, seq_len, embed_dim]
        out = self.embed(input)
        # Step 2: BiLSTM编码 [batch_size, seq_len, embed_dim] -> [batch_size, seq_len, 2*hidden_size]
        out, _ = self.lstm(out)
        # Step 3: 线性变换 [batch_size, seq_len, 2*hidden_size] -> [batch_size, seq_len, num_tags]
        return self.linear(out)

    def forward(self, input, mask):
        out = self._get_lstm_feature(input)
        return self.crf.decode(out, mask)

    def loss_fn(self, input, target, mask):
        y_pred = self._get_lstm_feature(input)
        return -self.crf.forward(y_pred, target, mask, reduction='mean')

if __name__ == '__main__':
    model = Model()
    input = torch.randint(0, 3000, (100, 50))
    print(model(input, None).shape)