import numpy as np

# numpyの型を明示的に設定
_np_qint8 = np.int8
_np_quint8 = np.uint8
_np_qint16 = np.int16
_np_quint16 = np.uint16
_np_qint32 = np.int
_np_quint32 = np.uint32
_np_float32 = np.float32
_np_float64 = np.float64
_np_float16 = np.float16

# FutureWarningsを無視する
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


#General 
from os import listdir
from pickle import dump, load
from numpy import array, argmax
from tqdm import tqdm

#Keras
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model, load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Input, Dense, LSTM, Embedding, Dropout
from keras.layers import add

#日本語処理
from googletrans import Translator
import MeCab

#Flickrデータセット
DATASET_DIR = "Flickr8k_Dataset"
TOKEN_FILE = "Flickr8k_text/Flickr8k.token.txt"

#ファイル名
IMAGE_FEATURES = "image_features_dict.pkl"
IMAGE_TEXTS = "image_texts_dict.txt"
TOKENIZER = "tokenizer.pkl"
TRAINED_MODEL = "model_19.h5"
TEST_IMAGE = "test_doginwater.jpg
