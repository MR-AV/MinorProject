from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score
from utils import load_data

import os
import pickle
import keras
import numpy as np
## Keras
from keras import regularizers
from keras import optimizers
from keras.models import Model, Sequential
from keras.layers import Dense, Embedding, LSTM
from keras.layers import Input, Flatten, Dropout, Activation, BatchNormalization
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
from keras.utils import to_categorical
from keras.optimizers import SGD


# load RAVDESS dataset
X_train, X_test, y_train, y_test = load_data(test_size=0.25)

"""from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_y_1 = LabelEncoder()
y_train = labelencoder_y_1.fit_transform(y_train)"""
# print some details
# number of samples in training data
print("[+] Number of training samples:", X_train.shape[0])
# number of samples in testing data
print("[+] Number of testing samples:", X_test.shape[0])
# number of features used
# this is a vector of features extracted 
# using utils.extract_features() method
print("[+] Number of features:", X_train.shape[1])
# best model, determined by a grid search
model_params = {
    'alpha': 0.01,
    'batch_size': 256,
    'epsilon': 1e-08, 
    'hidden_layer_sizes': (300,), 
    'learning_rate': 'adaptive', 
    'max_iter': 500, 
}
# initialize Multi Layer Perceptron classifier
# with best parameters ( so far )
#model = MLPClassifier(**model_params)

# train the model
print("[*] Training the model...")
#model.fit(X_train, y_train)
from sklearn.preprocessing import LabelEncoder
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)
lb = LabelEncoder()
'''y_train = np_utils.to_categorical(lb.fit_transform(y_train))
y_test = np_utils.to_categorical(lb.fit_transform(y_test))'''

X_train = np.expand_dims(X_train, axis=2)
X_test = np.expand_dims(X_test, axis=2)

"""from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_y_1 = LabelEncoder()
y_train = labelencoder_y_1.fit_transform(y_train)

labelencoder_y_2 = LabelEncoder()
y_test = labelencoder_y_1.fit_transform(y_test)
"""

model = Sequential()
model.add(Conv1D(64, 3, padding='same',input_shape=(X_train.shape[1], 1)))
model.add(Activation('relu'))
model.add(Conv1D(64, 3, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(MaxPooling1D())
model.add(Conv1D(128, 3, padding='same'))
model.add(Activation('relu'))
model.add(Conv1D(128, 3, padding='same'))
model.add(Activation('relu'))
model.add(Conv1D(128, 3, padding='same'))
model.add(Activation('relu'))
model.add(Conv1D(128, 3, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(MaxPooling1D())
model.add(Conv1D(256, 3, padding='same'))
model.add(Activation('relu'))
model.add(Conv1D(256, 3, padding='same'))
model.add(Activation('relu'))
model.add(LSTM(units=128, return_sequences = True)) 
model.add(Flatten())
#adding Lstm

# Edit according to target class no.
model.add(Dense(4))
model.add(Activation('softmax'))

#model.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])
#opt = keras.optimizers.SGD(lr=0.0001, momentum=0.0, decay=0.0, nesterov=False)
opt = 'adam'
model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['categorical_accuracy'])

history = model.fit(X_train, y_train, batch_size=16, epochs=5, validation_data=(X_test, y_test))
y_pred = model.predict_classes(X_test)

# predict 25% of data to measure how good we are
#y_pred = model.predict(X_test)

rounded_labels=np.argmax(y_test, axis=1)
rounded_labels[1]
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(rounded_labels, y_pred)

#plotting loss
"""import matplotlib.pyplot as plt
print("history keys = ", history.history.keys())
loss_train = history.history['loss']
loss_val = history.history['val_loss']
epochs = range(1,6)
plt.plot(epochs, loss_train, 'g', label='Training loss')
plt.plot(epochs, loss_val, 'b', label='validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()"""

#plotting accuracy
"""loss_train = history.history['categorical_accuracy']
loss_val = history.history['val_categorical_accuracy']
epochs = range(1,51)
plt.plot(epochs, loss_train, 'g', label='Training accuracy')
plt.plot(epochs, loss_val, 'b', label='validation accuracy')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
"""

# now we save the model
# make result directory if doesn't exist yet
if not os.path.isdir("result"):
    os.mkdir("result")

pickle.dump(model, open("result/mlp_classifier.model", "wb"))

