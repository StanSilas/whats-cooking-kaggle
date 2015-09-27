
import os
import json

import numpy as np
import pandas as pd

from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder



print('loading data ...')

input_dir = 'data/'


def load_data(filename):
	with open(os.path.join(input_dir, filename)) as train_f:
	    train_data = json.loads(train_f.read())

	X_train = [x['ingredients'] for x in train_data]
	X_train = [dict(zip(x,np.ones(len(x)))) for x in X_train]
	ids = [str(x['id']) for x in train_data]

	return X_train, ids


X_train, _ = load_data('train.json')
X_test, test_ids = load_data('test.json')

vec = DictVectorizer()
X_train = vec.fit_transform(X_train).toarray()
X_train = X_train.astype(np.float32)

X_test = vec.transform(X_test).astype(np.float32)

feature_names = np.array(vec.feature_names_)

lbl = LabelEncoder()

y_train = [y['cuisine'] for y in train_data]
y_train = lbl.fit_transform(y_train).astype(np.int32)

label_names = lbl.classes_ 

from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train,len(label_names))



from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD

model = Sequential()
model.add(Dense(len(feature_names), 64, init='uniform'))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(64, 64, init='uniform'))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(64, len(label_names), init='uniform'))
model.add(Activation('softmax'))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd)


model.fit(X_train, y_train, nb_epoch=20, batch_size=16)



def make_submission(y_pred, ids, encoder, fname):
    with open(fname, 'w') as f:
        f.write('id,cuisine')
        f.write('\n')
        for i, y_class in zip(test_ids,lbl.inverse_transform(pred)):
            f.write(','.join([i,y_class]))
            f.write('\n')
    print("Wrote submission to file {}.".format(fname))

pred = model.predict_classes(X_test.toarray())
make_submission(proba, test_ids, lbl, fname='data/keras-submit.csv')
