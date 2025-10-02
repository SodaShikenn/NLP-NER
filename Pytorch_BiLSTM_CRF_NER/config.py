ORIGIN_DIR = '__MACOSX/code/Pytorch_BiLSTM_CRF_NER/input/origin/'
ANNOTATION_DIR = '__MACOSX/code/Pytorch_BiLSTM_CRF_NER/output/annotation/'

TRAIN_SAMPLE_PATH = '__MACOSX/code/Pytorch_BiLSTM_CRF_NER/output/train_sample.txt'
TEST_SAMPLE_PATH = '__MACOSX/code/Pytorch_BiLSTM_CRF_NER/output/test_sample.txt'

VOCAB_PATH = '__MACOSX/code/Pytorch_BiLSTM_CRF_NER/output/vocab.txt'
LABEL_PATH = '__MACOSX/code/Pytorch_BiLSTM_CRF_NER/output/label.txt'

WORD_PAD = '<PAD>'
WORD_UNK = '<UNK>'

WORD_PAD_ID = 0
WORD_UNK_ID = 1
LABEL_O_ID = 0

VOCAB_SIZE = 3000
EMBEDDING_DIM = 100
HIDDEN_SIZE = 256
TARGET_SIZE = 31
LR = 1e-3
EPOCH = 5

MODEL_DIR = '__MACOSX/code/Pytorch_BiLSTM_CRF_NER/output/model/'

DEVICE = 'cpu'

# import torch
# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'