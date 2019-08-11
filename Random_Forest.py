"""
Josué Alexis M.G.
09-08-19
Random Forest
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

"se carga la base de datos en una variable"
"it´s loaded the database in a variable"
df = pd.read_csv('C:\\automobile.csv')
print(df.head(3))


"variables para entrenamiento"
"variables to training"
x = df[['longitud','ancho','peso']]
y = df['Traccion']

"30% test      70% training"
"división de mi base de datos para entrenamiento y test"
"split of my database to test and training"
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

"cantidad de arboles creados en mi sistema"
"number of trees created in my system"
bosque = RandomForestClassifier(n_estimators=100)
bosque.fit(x_train,y_train)

y_pred = bosque.predict(x_test)

print(accuracy_score(y_test, y_pred))

"predicción"
"prediction"
print(bosque.predict([[160.3, 65, 53.5]]))

"presición de mi sistema"
"acuracy of my system"
puntajes = cross_val_score(bosque, x, y, cv=5) 
print(puntajes.mean())

print( "Accuracy: %0.2f (+/-%0.2f)" %(puntajes.mean(), puntajes.std() *2))