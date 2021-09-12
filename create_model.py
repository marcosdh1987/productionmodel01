#import libraries

import numpy as np # linear algebra
import pandas as pd # data processing

ds = pd.read_csv(r'/Volumes/sharedfolder/ftpdata/ds/data_full_cleaned.csv')

print('Data Read ok')

X = ds[['Pressure[Bar]','DP[Bar]','Temperature[C]','Velocity[m/s]','LiqDen[kg/m3]','WaterCut[%]','choke','QgStd[m3/d]']].astype(float)
y = X.pop('QgStd[m3/d]')

from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=0)

#initial model to test ideas
from sklearn.ensemble import RandomForestRegressor

#for evaluate model
from sklearn.metrics import mean_squared_error


#try with deep learning model

from keras.layers import Input, Dense, Concatenate
from keras.models import Sequential
from keras.models import Model

model = Sequential()
model.add(Dense(64, input_dim=len(X.columns), activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_absolute_error')
model.summary()


from keras.callbacks import Callback

# Display training progress by printing a single dot for each completed epoch
class PrintDot(Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')
    
#Visualize the model's training progress using the stats stored in the history object.
from keras.callbacks import History 
history = History()
    
    
history = model.fit(X_train,y_train, epochs=2000, batch_size=12, validation_split = 0.2, verbose=0,
  callbacks=[PrintDot()])


print("model trained")
preds_valid = model.predict(X_valid)
print("the model error is ",f": {mean_squared_error(y_valid, preds_valid, squared=False)/1000:.4}","%")

print("Saving model....")
# Save your model
import joblib
joblib.dump(model, 'gas_model.pkl')
print("Model dumped!")