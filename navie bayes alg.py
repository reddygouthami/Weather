import pandas as pd
import numpy as np

dt=pd.read_csv('weather.csv')
print(dt.head())



#converting text into numerical
dt['play'],_=pd.factorize(dt['play'])
dt['outlook'],_=pd.factorize(dt['outlook'])
dt['temperature'],_=pd.factorize(dt['temperature'])
dt['humidity'],_=pd.factorize(dt['humidity'])
dt['windy'],_=pd.factorize(dt['windy'])

print(dt.head())
x=dt.iloc[:,:4]
y=dt.iloc[:,-1]


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=0)


from sklearn.naive_bayes import GaussianNB
model=GaussianNB().fit(x_train,y_train)

y_pred=model.predict(x_test)

from sklearn.metrics import accuracy_score,confusion_matrix
conf_mat=confusion_matrix(y_test,y_pred)
print('accuracy',accuracy_score)

#to predict output

new=pd.DataFrame()


outlook=input('enter a outlook(sunny/overcast/rainy)')
if(outlook=='sunny'):
    outlook=0
if(outlook=='overcast'):
    outlook=1
if(outlook=='rainy'):
    outlook=2
temperature=input('enter a temp(hot/mild/cool)')
if(temperature=='hot'):
    temperature=0
if(temperature=='mild'):
    tempearature=1
if(temperature=='cool'):
    temperature=2
humidity=input('enter a humidity(high/normal)')
if(humidity=='high'):
    humidity=0
if(humidity=='normal'):
    humidity=1
windy=input('enter a windy(True/False)')
if(windy=='False'):
    windy=0
if(windy=='True'):
    windy=1
new['outlook']=[outlook]
new['temperature']=[temperature]
new['humidity']=[humidity]
new['windy']=[windy]

print(new)
predicted_y=model.predict(new)
print(predicted_y)








              


