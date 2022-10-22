
!pip install num2words
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model
from num2words import num2words
import keras
from keras.layers import Input
from keras.models import Model
import nltk
from nltk.data import load
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from keras.models import Sequential
from keras.layers.recurrent import LSTM, GRU
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
import re
from textblob import TextBlob
from keras.layers import Bidirectional, SpatialDropout1D,Concatenate
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('tagsets')
nltk.download('omw-1.4')
from google.colab import files
from collections import Counter
import tensorflow as tf
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.callbacks import ReduceLROnPlateau
import io
import string
from sklearn.metrics import mean_squared_error
import math
import sklearn
import gc


print(tf.__version__)

"""Uploading train data"""

uploaded = files.upload()

df = pd.read_csv(io.BytesIO(uploaded['train.csv']))

"""Removal unnecessary columns"""

df.drop(['url_legal' , 'license'] , axis=1 , inplace = True)
df.head()

"""Text proccessing"""

df['proccessed_text']=df['excerpt']

def convert_lower_case(data):
    return data.lower()

def number_removing(data):
    return re.sub(r'\d+', '', data)




def punctation_removing(data):
    return data.translate(str.maketrans('','', string.punctuation))

def remove_stop_words(data):
    stop_words = stopwords.words('english')
    words = word_tokenize(str(data))
    new_text = ""
    for w in words:
        if w not in stop_words and len(w) > 1:
            new_text = new_text + " " + w
    return new_text

def text_lemmization(data):
    lemmatizer=WordNetLemmatizer()
    tokens = word_tokenize(str(data))
    new_text = ""
    for w in tokens:
        new_text = new_text + " " + lemmatizer.lemmatize(w)
    return new_text

def text_stemming(data):
    stemmer= PorterStemmer()    
    tokens = word_tokenize(str(data))
    new_text = ""
    for w in tokens:
        new_text = new_text + " " + stemmer.stem(w)
    return new_text

def proccess_text(data):
    data=convert_lower_case(data)

    data=number_removing(data)
    
    data=punctation_removing(data)
    data=remove_stop_words(data)
    data=text_lemmization(data)
    return data

df['proccessed_text']=df['proccessed_text'].apply(proccess_text)

"""Feature extraction"""

def part_of_speech_tagging(data):
    res = data.lower()
    tokens = nltk.word_tokenize(res)
    tags = nltk.pos_tag(tokens)
    return Counter( tag for word,  tag in tags)

def polarity_txt(text):
    return TextBlob(text).sentiment[0]
def subj_txt(text):
    return  TextBlob(text).sentiment[1]

def part_of_speech_feature(data):
    tagdict = load('help/tagsets/upenn_tagset.pickle')
    d = dict.fromkeys(tagdict.keys(), 0)
    for key, value in data.items():
        d[key]=value
    d=list(d.values())    
    return np.array(d,dtype=int)

df['subjectivity'] = df['excerpt'].apply(subj_txt)
df['polarity'] = df['excerpt'].apply(polarity_txt)
#newcol=df[['subjectivity','polarity']].values
partOfSpeech=df['excerpt'].apply(part_of_speech_tagging)
df['pos_tag']=partOfSpeech.apply(part_of_speech_feature)

df.head()

xtrain, xtest, ytrain, ytest = train_test_split(df[['proccessed_text','subjectivity','polarity','pos_tag']], df['target'], test_size=0.2,random_state=50)
sub_pol=xtrain[['subjectivity','polarity']].values
sub_pol_test=xtest[['subjectivity','polarity']].values

"""


Tokenizer for keras model"""

EMBEDDING_DIMENSION = 100
VOCABULARY_SIZE = 5000
MAX_LENGTH = int((df["proccessed_text"].apply(lambda x: len(x.split()))).mean())
OOV_TOK = '<OOV>'

tokenizer = Tokenizer(num_words=VOCABULARY_SIZE, oov_token=OOV_TOK)
tokenizer.fit_on_texts(list(xtrain.proccessed_text) + list(xtest.proccessed_text))
xtrain_sequences = tokenizer.texts_to_sequences(xtrain.proccessed_text)
xtest_sequences = tokenizer.texts_to_sequences(xtest.proccessed_text)
word_index = tokenizer.word_index
print('Vocabulary size:', len(word_index))

xtrain_pad = sequence.pad_sequences(xtrain_sequences, maxlen=MAX_LENGTH, padding='post', truncating='post')
xtest_pad = sequence.pad_sequences(xtest_sequences, maxlen=MAX_LENGTH, padding='post', truncating='post')

"""BERT Approach"""

# import os
# model_name = 'bert_version13'

# filenames = next(os.walk("/content"), (None, None, []))[2]
# for i in filenames:
#   print(i)

# trn = pd.read_csv(io.BytesIO(uploaded['train.csv']),index_col='id')

# tst = pd.read_csv(io.BytesIO(uploaded['test.csv']),index_col='id')

# sample_file = pd.read_csv(io.BytesIO(uploaded['sample_submission.csv']))

# y = trn['target'].values
# trn.head()

# trn['excerpt'].str.split(' ').apply(len).describe()

# trn['excerpt'].str.split(' ').apply(len).hist(bins=50)

# !pip install transformers==4.5.1

# os.getcwd()

# def bert_encode(texts, tokenizer, max_len=205):
#     input_ids = []
#     token_type_ids = []
#     attention_mask = []
    
#     for text in texts:
#         token = tokenizer(text, max_length=max_len, truncation=True, padding='max_length',
#                          add_special_tokens=True)
#         input_ids.append(token['input_ids'])
#         token_type_ids.append(token['token_type_ids'])
#         attention_mask.append(token['attention_mask'])
    
#     return np.array(input_ids), np.array(token_type_ids), np.array(attention_mask)

# # tokenizer, bert_config = load_tokenizer()
# import json
# tokenizer = transformers.BertTokenizerFast.from_pretrained("bert-large-uncased")
# tokenizer.save_pretrained("/content")

# # filenames = next(os.walk("/content/config"), (None, None, []))[2]
# # for i in filenames:
# #   print(i)
# #   if i=="tokenizer.json":
# #     continue
# #   try:
# #     f = open("/content/config/"+i)
# #     data = json.load(f)
# #     d=json.dumps(data,indent=4)
# #     print(d)
# #   except:
# #     continue

# bert_config = transformers.BertConfig.from_pretrained("/content")
# bert_config.output_hidden_states = True

# X = bert_encode(trn['excerpt'].values, tokenizer, max_len=205)
# X_tst = bert_encode(tst['excerpt'].values, tokenizer, max_len=205)
# y = trn['target'].values
# print(X[0].shape, X_tst[0].shape, y.shape)

# def build_model(bert_model, max_len=205):    
#     input_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_ids")
#     token_type_ids = Input(shape=(max_len,), dtype=tf.int32, name="token_type_ids")
#     attention_mask = Input(shape=(max_len,), dtype=tf.int32, name="attention_mask")

#     sequence_output = bert_model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)[0]
#     clf_output = sequence_output[:, 0, :]
#     clf_output = Dropout(.1)(clf_output)
#     out = Dense(1, activation='linear')(clf_output)
    
#     model = Model(inputs=[input_ids, token_type_ids, attention_mask], outputs=out)
#     model.compile(tf.keras.optimizers.Adam(lr=1e-5), loss="mean_squared_error", metrics=[RootMeanSquaredError()])
    
#     return model

# def scheduler(epoch, lr, warmup=5, decay_start=10):
#     if epoch <= warmup:
#         return lr / (warmup - epoch + 1)
#     elif warmup < epoch <= decay_start:
#         return lr
#     else:
#         return lr * tf.math.exp(-.1)

# ls = tf.keras.callbacks.LearningRateScheduler(scheduler)
# es = tf.keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)

# cv = sklearn.model_selection.KFold(n_splits=5, shuffle=True, random_state=42)

# p = np.zeros_like(y, dtype=float)
# p_tst = np.zeros((X_tst[0].shape[0], ), dtype=float)

# for i, (i_trn, i_val) in enumerate(cv.split(X[0]), 1):
#     print(f'training CV #{i}:')
#     tf.random.set_seed(42 + i)

#     bert_model = transformers.TFBertModel.from_pretrained("bert-large-uncased", config=bert_config)
#     clf = build_model(bert_model, max_len=205)
#     if i == 1:
#         print(clf.summary())
#     history = clf.fit([x[i_trn] for x in X], y[i_trn],
#                       validation_data=([x[i_val] for x in X], y[i_val]),
#                       epochs=9,
#                       batch_size=8,
#                       callbacks=[ls])
#     clf.save_weights(f'{model_name}_cv{i}.h5')

#     p[i_val] = clf.predict([x[i_val] for x in X]).flatten()
#     p_tst += clf.predict(X_tst).flatten() / 5
    
#     tf.keras.backend.clear_session()
#     del clf, bert_model
#     gc.collect()

"""Model sequential
The model includes 7 layers these are: Input layer for embedding,Embedding, Bidrectional LSTM, Dropout, Bidrectional LSTM, Dropout, Dense. After adding layers, we compile the model with adam optimizer and mean square error loss function. Additonally, model includes stop mechanism and return to best weights.



"""

model = Sequential()
model.add(Embedding(VOCABULARY_SIZE, EMBEDDING_DIMENSION, input_length=MAX_LENGTH))
model.add(Bidirectional(LSTM(EMBEDDING_DIMENSION,return_sequences=True)))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(EMBEDDING_DIMENSION)))
model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(loss='mse',optimizer="adam", metrics=[RootMeanSquaredError()])
model.summary()

num_epochs = 30
history = model.fit(xtrain_pad, np.array(ytrain), epochs=num_epochs, 
                    validation_split=0.1,
                   callbacks=[EarlyStopping(
    min_delta=0.001, # minimium amount of change to count as an improvement
    patience=5, # how many epochs to wait before stopping
    restore_best_weights=True,
),ReduceLROnPlateau(monitor='val_root_mean_squared_error', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)])

"""Predict"""
predicted_test_values = model.predict(xtest_pad)
predicted_train_values = model.predict(xtrain_pad)
# calculate root mean squared error
print("Train rmse score ",math.sqrt(mean_squared_error(ytrain, predicted_train_values)))
print("Test rmse score ",math.sqrt(mean_squared_error(ytest, predicted_test_values)))

def graph_plots(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.show()
  
graph_plots(history, "root_mean_squared_error")
graph_plots(history, "loss")

plot_model(model, to_file='multilayer_bilstm_graph.png')

"""Model sequential
The model includes 4 layers these are: Input layer for embedding, Embedding, Bidrectional LSTM, Dense. After adding layers, we compile the model with adam optimizer and mean square error loss function. Additonally, model includes stop mechanism and return to best weights.



"""

model = Sequential()
model.add(Embedding(VOCABULARY_SIZE, EMBEDDING_DIMENSION, input_length=MAX_LENGTH))
model.add(Bidirectional(LSTM(EMBEDDING_DIMENSION)))
model.add(Dense(1))

model.compile(loss='mse',optimizer="adam", metrics=[RootMeanSquaredError()])
model.summary()

num_epochs = 30
history = model.fit(xtrain_pad, np.array(ytrain), epochs=num_epochs, 
                    validation_split=0.1,
                   callbacks=[EarlyStopping(
    min_delta=0.001, # minimium amount of change to count as an improvement
    patience=5, # how many epochs to wait before stopping
    restore_best_weights=True,
),ReduceLROnPlateau(monitor='val_root_mean_squared_error', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)])

"""Predict"""
predicted_test_values = model.predict(xtest_pad)
predicted_train_values = model.predict(xtrain_pad)
# calculate root mean squared error
print("Train rmse score ",math.sqrt(mean_squared_error(ytrain, predicted_train_values)))
print("Test rmse score ",math.sqrt(mean_squared_error(ytest, predicted_test_values)))

def graph_plots(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.show()
  
graph_plots(history, "root_mean_squared_error")
graph_plots(history, "loss")

plot_model(model, to_file='bilstm_graph.png')

"""Model sequential The model includes 4 layers these are: Input layer for embedding, Embedding, Bidrectional GRU, Dense. After adding layers, we compile the model with adam optimizer and mean square error loss function. Additonally, model includes stop mechanism and return to best weights."""

model = Sequential()
model.add(Embedding(VOCABULARY_SIZE, EMBEDDING_DIMENSION, input_length=MAX_LENGTH))
model.add(Bidirectional(GRU(EMBEDDING_DIMENSION)))
model.add(Dense(1))

model.compile(loss='mse',optimizer="adam", metrics=[RootMeanSquaredError()])
model.summary()

num_epochs = 30
history = model.fit(xtrain_pad, np.array(ytrain), epochs=num_epochs, 
                    validation_split=0.1,
                   callbacks=[EarlyStopping(
    min_delta=0.001, # minimium amount of change to count as an improvement
    patience=5, # how many epochs to wait before stopping
    restore_best_weights=True,
),ReduceLROnPlateau(monitor='val_root_mean_squared_error', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)])

"""Predict"""
predicted_test_values = model.predict(xtest_pad)
predicted_train_values = model.predict(xtrain_pad)
# calculate root mean squared error
print("Train rmse score ",math.sqrt(mean_squared_error(ytrain, predicted_train_values)))
print("Test rmse score ",math.sqrt(mean_squared_error(ytest, predicted_test_values)))

def graph_plots(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.show()
  
graph_plots(history, "root_mean_squared_error")
graph_plots(history, "loss")

plot_model(model, to_file='bigru_graph.png')

"""## FunctionalAPI model

The model uses a Bidirectional LSTM recurrent neural network and dense layers to calculate the final result. It takes three different and separated inputs, which are concatenated in further layers of the neural network. The first input of the neural network is preprocessed text data(without punctations, stopwords, numbers). The two remaining inputs are metadata extracted from the text. The second input is a dictionary with a number of each part of speech in the text. The third input is a list with two elements polarity and subjectivity of the text. Before merging the inputs neural networks operate separately on two first inputs. The first input is processed by two layers of bidirectional LSTM. The second input is processed by two dense layers. The result of this processing is merged with the third input. The concatenated inputs are processed by a dense layer with a linear activation function that provides the final result.
"""

input1 = Input(shape=(MAX_LENGTH,))

input2=Input(shape=(45,))

input3=Input(shape=(2,))

embedding_layer=Embedding(VOCABULARY_SIZE, EMBEDDING_DIMENSION, input_length=MAX_LENGTH)(input1)
dropOut=SpatialDropout1D(0.3)(embedding_layer)
biLSTM1=Bidirectional(LSTM(128, return_sequences=True))(dropOut)

biLSTM2=Bidirectional(LSTM(128))(biLSTM1)


dense2_layer_1 = Dense(90)(input2)

dropOut2=Dropout(0.1)(dense2_layer_1)

dense2_layer_2=Dense(20)(dropOut2)

dropOut3=Dropout(0.1)(input3)
concat_layer = Concatenate()([biLSTM2, dense2_layer_2,dropOut3])#,dense3_layer_1])


final_dropout=Dropout(0.1)(concat_layer)

#dense_last=Dense(100)(final_dropout)

output = Dense(1, activation='linear')(final_dropout)

model = Model(inputs=[input1,input2,input3], outputs=output)
# summarize layer
#opt = tf.keras.optimizers.Adam(learning_rate=0.01)
opt = tf.keras.optimizers.SGD(learning_rate=0.001)
model.compile(optimizer='rmsprop',loss='mse',  metrics=[RootMeanSquaredError()])
print(model.summary())

#history = model.fit(x=[xtrain_pad,np.stack(xtrain.pos_tag,axis=0),sub_pol], y=np.array(ytrain), epochs=10,batch_size=256, validation_data=([xtest_pad,np.stack(xtest.pos_tag,axis=0),sub_pol_test],ytest))
history = model.fit(x=[xtrain_pad,np.stack(xtrain.pos_tag,axis=0),sub_pol], y=np.array(ytrain), epochs=30, 
                    validation_split=0.3,
                   callbacks=[EarlyStopping(
    min_delta=0.001, # minimium amount of change to count as an improvement
    patience=5, # how many epochs to wait before stopping
    restore_best_weights=True,
),ReduceLROnPlateau(monitor='val_root_mean_squared_error', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)])

predicted_test_values = model.predict([xtest_pad,np.stack(xtest.pos_tag,axis=0),sub_pol_test])
predicted_train_values = model.predict([xtrain_pad,np.stack(xtrain.pos_tag,axis=0),sub_pol])
# calculate root mean squared error
print("Train rmse score ",math.sqrt(mean_squared_error(ytrain, predicted_train_values)))
print("Test rmse score ",math.sqrt(mean_squared_error(ytest, predicted_test_values)))

def graph_plots(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.show()
  
graph_plots(history, "root_mean_squared_error")
graph_plots(history, "loss")

plot_model(model, to_file='multilayer_perceptron_graph.png')
