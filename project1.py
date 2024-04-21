import pandas as pd

data = pd.read_csv("visit.csv")
x = data["Time"]
y = data["Buy"]

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

model = Sequential()
model.add(Dense(1, input_shape=(1,), activation='sigmoid'))
model.compile(Adam(learning_rate=0.99), loss='binary_crossentropy', metrics=['accuracy'])

model.fit(x,y,epochs=40)

yp = model.predict(x)

ypc = yp > 0.5

from sklearn.metrics import accuracy_score, confusion_matrix
print(accuracy_score(y, ypc))
print(confusion_matrix(y, ypc))

from keras.wrappers.scikit_learn import KerasClassifier
def cls_fun():
    model = Sequential()
    model.add(Dense(1, input_shape=(1,), activation='sigmoid'))
    model.compile(Adam(learning_rate=0.99), loss='binary_crossentropy', metrics=['accuracy'])
    return model

wrapper_model = KerasClassifier(build_fn=cls_fun, epochs=40)

from sklearn.model_selection import cross_val_score, KFold
kf = KFold(4)
acc = cross_val_score(wrapper_model, x,y, cv=kf)
print(acc)
print(acc.mean())