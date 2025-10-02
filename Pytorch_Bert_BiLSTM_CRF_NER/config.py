# BERT改造
BERT_MODEL = './hf_demo/bert-base-chinese'
EMBEDDING_DIM = 768
MAX_POSITION_EMBEDDINGS = 512

ORIGIN_DIR = './Part1_NER/code/Pytorch_Bert_BiLSTM_CRF_NER/input/origin/'
ANNOTATION_DIR = './Part1_NER/code/Pytorch_Bert_BiLSTM_CRF_NER/output/annotation/'

TRAIN_SAMPLE_PATH = './Part1_NER/code/Pytorch_Bert_BiLSTM_CRF_NER/output/train_sample.txt'
TEST_SAMPLE_PATH = './Part1_NER/code/Pytorch_Bert_BiLSTM_CRF_NER/output/test_sample.txt'

VOCAB_PATH = './Part1_NER/code/Pytorch_Bert_BiLSTM_CRF_NER/output/vocab.txt'
LABEL_PATH = './Part1_NER/code/Pytorch_Bert_BiLSTM_CRF_NER/output/label.txt'

WORD_PAD = '<PAD>'
WORD_UNK = '<UNK>'

WORD_PAD_ID = 0
WORD_UNK_ID = 1
LABEL_O_ID = 0

VOCAB_SIZE = 3000
# EMBEDDING_DIM = 100
HIDDEN_SIZE = 256
TARGET_SIZE = 31
LR = 1e-3
EPOCH = 5

MODEL_DIR = './Part1_NER/code/Pytorch_Bert_BiLSTM_CRF_NER/output/model/'

import torch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# DEVICE = 'cpu'
