import numpy as np
import pandas as pd

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, Dropout
from keras.models import load_model
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.metrics import AUC
from keras.optimizers import Adam

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt


data = pd.read_csv("Islenmis_yorum.csv", sep="\t").drop("Unnamed: 0", axis=1)
df = data.copy()[0:400000]  # train
df2 = data.copy()[400000:]  # test

tokenizer = Tokenizer(split=' ',num_words=2000)

tokenizer.fit_on_texts(df["Comments"].values.astype(str)) 

document_count = tokenizer.document_count  
vocab_size = len(tokenizer.word_index) 

# train data sequence
train_sequences = tokenizer.texts_to_sequences(df["Comments"].values.astype(str))
max_length = max([len(x) for x in train_sequences])
word_index = tokenizer.word_index
train_padded = pad_sequences(train_sequences, maxlen=max_length)  


trainLabels = df["Score"].values

# for test values:
test_sequences = tokenizer.texts_to_sequences(df2["Comments"].values.astype(str))
test_padded = pad_sequences(test_sequences, maxlen=max_length)
testLabels = df2["Score"].values

# dengesizlik için
class_weights_dict ={0:5.79176586, 1:0.54724321} 


# Model

input_dim = vocab_size
output_dim = train_padded.shape[1]
epochs = 10
batch_size_ = 500
opt = Adam(learning_rate=0.001)

# define the model
model = Sequential()
model.add(Embedding(input_dim, output_dim, input_length=max_length, name= 'embeded'))
model.add(SpatialDropout1D(0.25))
model.add(LSTM(128, activation="relu", return_sequences=True))
model.add(LSTM(64, activation="relu"))
model.add(Dropout(0.25))
model.add(Dense(1, activation='sigmoid'))

# compile the model
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=["accuracy"])

# summarize the model
print(model.summary())

# fit the model
model.fit(train_padded, trainLabels, epochs=epochs, verbose=1, 
          batch_size=batch_size_, class_weight=class_weights_dict)

# evaluate the model
loss, accuracy = model.evaluate(test_padded, testLabels, verbose=0)
print('Accuracy: %f' % (accuracy*100))

pred = model.predict_classes(test_padded)
pred = pred.reshape(-1,)
print(f"test auc score : {roc_auc_score(pred, testLabels)}")

model.save("son_model.h5")

