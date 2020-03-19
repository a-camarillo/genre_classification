import pandas as pd 
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv('EDA/features2.csv')

X = data[['popularity','key','mode','energy','loudness','danceability','valence']]
y = data[['genre']]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)

dt = DecisionTreeClassifier(min_samples_split=500, random_state=1)
rf = RandomForestClassifier(min_samples_split=500, random_state=1,n_estimators=250)
ada = AdaBoostClassifier(base_estimator=rf,random_state=1)

model1 = dt.fit(X_train,y_train)
model2 = rf.fit(X_train,y_train.values.ravel())
model3 = ada.fit(X_train,y_train.values.ravel())

dt_test = model1.predict(X_test)
rf_test = model2.predict(X_test)
ada_test = model3.predict(X_test)

dt_accuracy = accuracy_score(y_test,dt_test)
rf_accuracy = accuracy_score(y_test,rf_test)
ada_accuracy = accuracy_score(y_test,ada_test)
print('Decision Tree Classifier Accuracy is:', round(dt_accuracy*100,2),'%')
print('Random Forest Classifier Accuracy is:', round(rf_accuracy*100,2),'%')
print('AdaBoost Classifier Accuracy is:', round(ada_accuracy*100,2),'%')