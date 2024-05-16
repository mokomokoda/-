import numpy as np


_np_qint8 = np.int8
_np_quint8 = np.uint8
_np_qint16 = np.int16
_np_quint16 = np.uint16
_np_qint32 = np.int32
_np_quint32 = np.uint32
_np_float32 = np.float32
_np_float64 = np.float64
_np_float16 = np.float16


import warnings
warnings.filterwarnings("ignore", category=FutureWarning)



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


from googletrans import Translator
import MeCab


DATASET_DIR = "Flickr8k_Dataset"
TOKEN_FILE = "Flickr8k_text/Flickr8k.token.txt"


IMAGE_FEATURES = "image_features_dict.pkl"
IMAGE_TEXTS = "image_texts_dict.txt"
TOKENIZER = "tokenizer.pkl"
TRAINED_MODEL = "model_19.h5"
TEST_IMAGE = "test_doginwater.jpg"

class ImagePreprocessor()
    dataset_dit = ""

    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.feature_extractor = self.GetFeatureExtractor()
        self.image_features_dict = {}
        return

    def GetFeatureExtractor(self):
       
        model = VGG16()
        model.layers.pop()
        model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
        return model

    def GetImageFeature(self, filename, model):
   = load_img(filename, target_size=(224, 224))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)
        image_feature = model.predict(image, verbose=0)
        return image_feature

    def MakeFeaturesDict(self):
     
        for name in tqdm(listdir(self.dataset_dir)):
            filename = self.dataset_dir + '/' + name
            image_feature = self.GetImageFeature(filename, self.feature_extractor)
            image_id = name.split('.')[0]
            self.image_features_dict[image_id] = image_feature#1×4096
        return self.image_features_dict

    def SaveDict(self, filename):
        dump(self.image_features_dict, open(filename, 'wb'))
        return
class TextPreprocessor():
    

    def __init__(self, token_file):
        self.tokens = self.LoadTexts(token_file)
        self.image_texts_dict = {}
        self.m = MeCab.Tagger ("mecabrc")
        self.translator = Translator()
        return

    def LoadTexts(self, filename):
       
        with open(filename, "r") as file:
            text = file.read()
        return text

    def __translate(self, text):
       
        text_ja = self.translator.translate(text, dest='ja').text
        return text_ja

    def __mecab(self, text_ja):
        
        text_mecab = self.m.parse(text_ja).split('\n')
        return text_mecab

    def Preprocess(self, text):
    
        word_list = []
        text_ja = self.__translate(text)
        text_mecab = self.__mecab(text_ja)
        for word in text_mecab:
            word_list.append(word.split('\t')[0])
        text_preprocess = ' '.join(word_list).replace(" 、", "").replace(" 。", "").replace(" EOS ", "")
        text_final = 'startseq ' + text_preprocess + ' endseq'
        return text_final

    def GetIDAndText(self, token):
      
        token = token.split()
        image_id, image_text = token[0], token[1:]
        image_id = image_id.split('.')[0]
        image_text = ' '.join(image_text)
        return image_id, image_text

    def MakeTextsDict(self):
      
        for token in tqdm(self.tokens.split('\n')):
            image_id, image_text = self.GetIDAndText(token)
            image_text = self.Preprocess(image_text)
            if image_id not in self.image_texts_dict:
                self.image_texts_dict[image_id] = []
            self.image_texts_dict[image_id].append(image_text)
        return self.image_texts_dict

    def SaveDict(self, filename):
        dump(self.image_texts_dict, open(filename, 'wb'))
        return
        
class Trainer(TextPreprocessor):
   

    def __init__(self, features_dict, texts_dict, epochs):
        self.train_texts_dict = texts_dict
        self.train_features_dict = features_dict
        self.tokenizer = Tokenizer()
        self.epochs = epochs
        return

    def __dictToList(self, texts_dict):
       
        texts_list = []
        for key in texts_dict.keys():
            [texts_list.append(d) for d in texts_dict[key]]
        return texts_list

    def MakeTokenizer(self):
      
        train_texts_list = self.__dictToList(self.train_texts_dict)
        self.tokenizer.fit_on_texts(train_texts_list)
        dump(self.tokenizer, open('tokenizer.pkl', 'wb'))
        return None

    def GetVocabSize(self):
       
        self.vocab_size = len(self.tokenizer.word_index) + 1
        print("Vocabulary Size of Texts:　", self.vocab_size)
        return

    def GetMaxLength(self):
     
        lists = self.__dictToList(self.train_texts_dict)
        self.max_length = max(len(d.split()) for d in lists)
        print("Max Length of Texts: ", self.max_length)
        return self.max_length

    def MakeCaptioningModel(self):
       
       
        inputs1 = Input(shape=(4096,))
        ie1 = Dropout(0.5)(inputs1)
        ie2 = Dense(256, activation='relu')(ie1)
        
        inputs2 = Input(shape=(self.max_length,))
        se1 = Embedding(self.vocab_size, 256, mask_zero=True)(inputs2)
        se2 = Dropout(0.5)(se1)
        se3 = LSTM(256)(se2)

        decoder1 = add([ie2, se3])
        decoder2 = Dense(256, activation='relu')(decoder1)
        outputs = Dense(self.vocab_size, activation='softmax')(decoder2)
       
        model = Model(inputs=[inputs1, inputs2], outputs=outputs)
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        #model.summary()
        self.model = model
        return

    def MakeInputOutput(self, image_texts, image_feature):
     
        X1, X2, y = [], [], []
        for image_text in image_texts:
            seq = self.tokenizer.texts_to_sequences([image_text])[0]
            for i in range(1, len(seq)):
                in_seq, out_seq = seq[:i], seq[i]
                in_seq = pad_sequences([in_seq], maxlen=self.max_length)[0]
                out_seq = to_categorical([out_seq], num_classes=self.vocab_size)[0]
                X1.append(image_feature)
                X2.append(in_seq)
                y.append(out_seq)
        return array(X1), array(X2), array(y)

    def DataGenerator(self):
    
        while 1:
            for key, image_texts in self.train_texts_dict.items():
                image_feature = self.train_features_dict[key][0]
                in_img, in_seq, out_word = self.MakeInputOutput(image_texts, image_feature)
                yield [[in_img, in_seq], out_word]

    def TrainModel(self):
      
        self.MakeTokenizer()
        self.GetVocabSize()
        self.GetMaxLength()
        self.MakeCaptioningModel()

        steps=len(self.train_texts_dict)
        for i in range(self.epochs):
            generator = self.DataGenerator()
            self.model.fit_generator(generator, epochs=self.epochs, steps_per_epoch=steps, verbose=1)
            self.model.save('model_' + str(i) + '.h5')
        return None
        class Predictor(ImagePreprocessor):
 

    def __init__(self, model_file, token_file, max_length):
        self.model = load_model(model_file)
        self.tokenizer = load(open(token_file, "rb"))
        self.max_length= max_length
        self.feature_extractor = self.GetFeatureExtractor()
        return

    def IDToWord(self, integer):
       
        for word, index in self.tokenizer.word_index.items():
            if index == integer:
                return word
        return None

    def GetFeatureExtractor(self):

        return super().GetFeatureExtractor()

    def GetImageFeature(self, filename, model):
      
        return super().GetImageFeature(filename, model) 

    def Inference(self, test_image):
     
        image_feature = self.GetImageFeature(test_image, self.feature_extractor)
        text = "startseq"
        for i in range(self.max_length):
            seq = self.tokenizer.texts_to_sequences([text])[0]
            seq = pad_sequences([seq], maxlen=self.max_length)
            yhat = self.model.predict([image_feature,seq], verbose=0)
            yhat = argmax(yhat)
            word = self.IDToWord(yhat)
            if word is None:
                break
            text += " " + word
            if word == "endseq":
                break
        return text
        
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

from keras.applications import VGG16

vgg16_weights_path = VGG16(weights='imagenet', include_top=False).weights

import shutil

source_file = "vgg16_weights_tf_dim_ordering_tf_kernels.h5"

destination_directory = "‪Users\\phantom\\AppData\\Roaming\\.anaconda\\navigator\\scripts\py36\\app.bat"

shutil.move(source_file, destination_directory)
import urllib.request

url = "https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5"
filename = "vgg16_weights_tf_dim_ordering_tf_kernels.h5"

urllib.request.urlretrieve(url, filename)


ip = ImagePreprocessor(dataset_dir=DATASET_DIR)
image_features_dict = ip.MakeFeaturesDict()
ip.SaveDict(filename=IMAGE_FEATURES)


tp = TextPreprocessor(token_file=TOKEN_FILE)
image_texts_dict = tp.MakeTextsDict()
tp.SaveDict(filename=IMAGE_TEXTS)

image_features_dict = load(open(IMAGE_FEATURES, 'rb'))
image_texts_dict = load(open(IMAGE_TEXTS, 'rb'))


tr = Trainer(features_dict=image_features_dict, texts_dict=image_texts_dict, epochs=20)
tr.TrainModel()


pr = Predictor(model_file=TRAINED_MODEL, token_file=TOKENIZER, max_length=tr.GetMaxLength())
pr.Inference(TEST_IMAGE)

        
