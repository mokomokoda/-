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

class ImagePreprocessor():
    """
    画像を訓練用に前処理する。
    """

    dataset_dit = ""

    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.feature_extractor = self.GetFeatureExtractor()
        self.image_features_dict = {}
        return

    def GetFeatureExtractor(self):
        """
        ・モデルをロードする。
        ・ソフトマックスの最終層を取り除く。
        ・モデルを再定義する。
        """
        model = VGG16()
        model.layers.pop()
        model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
        return model

    def GetImageFeature(self, filename, model):
        """
        ・特徴量抽出器をロードする。
        ・画像ファイルをロードする。
        ・PixcelをNumpy形式に変換する。
        ・学習用にモデルを変形する。
        ・VGGモデル用の前処理を行う。
        ・画像の特徴量を取得する。
        """
        image = load_img(filename, target_size=(224, 224))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)
        image_feature = model.predict(image, verbose=0)
        return image_feature

    def MakeFeaturesDict(self):
        """
        ・ファイルから画像特徴量を取得する。
        ・ファイルから画像IDを取得する。
        ・画像IDと画像特徴量を辞書に保存する。
        """
        for name in tqdm(listdir(self.dataset_dir)):
            filename = self.dataset_dir + '/' + name
            image_feature = self.GetImageFeature(filename, self.feature_extractor)
            image_id = name.split('.')[0]
            self.image_features_dict[image_id] = image_feature#1×4096
        return self.image_features_dict

    def SaveDict(self, filename):
        dump(self.image_features_dict, open(filename, 'wb'))
        return



