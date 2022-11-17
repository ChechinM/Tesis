#IMPORTANDO LIBRERIAS
import pickle
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split

#LEYENDA LA DATA
df = pd.read_csv('data/data.csv', delimiter=';')

#CAMBIANDO COLUMNAS CON TIPO OBJECT A VALORES CATEGORICOS
for feature in df.columns: 
    if df[feature].dtype == 'object': 
        df[feature] = pd.Categorical(df[feature]).codes

#MODELADO
y = df.TARGET.copy()
X = df.drop(['TARGET'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=123)

#ENTRENAMOS LOS DATOS SEGUN EL MODELO DE REGRESION LOGISTICA
modelRegression = LogisticRegression(solver='liblinear')
modelRegression.fit(X_train, y_train)

#PRECIDIENDO CON EL MODELO PREVIAMENTE ENTRENADO
y_predict = modelRegression.predict(X_test)

#CALCULANDO ALGUNAS METRICAS
results_df = pd.DataFrame(data=[["Regresion Logistica",  metrics.accuracy_score(y_test, y_predict), metrics.precision_score(y_test, y_predict),metrics.recall_score(y_test, y_predict) ]], 
                          columns=['Modelo', 'Accuracy', 'Precision', 'Recall'])

#ENTRENAMOS LOS DATOS SEGUN EL MODELO DE ANALISIS DISCRIMINANTE
modelDiscriminant = LinearDiscriminantAnalysis()
modelDiscriminant.fit(X_train, y_train)

#PRECIDIENDO CON EL MODELO PREVIAMENTE ENTRENADO
y_predict = modelDiscriminant.predict(X_test)

#CALCULANDO ALGUNAS METRICAS
results_df_2 = pd.DataFrame(data=[["Analisis Discriminante",  metrics.accuracy_score(y_test, y_predict), metrics.precision_score(y_test, y_predict),metrics.recall_score(y_test, y_predict) ]], 
                          columns=['Modelo', 'Accuracy', 'Precision', 'Recall'])

#CREANDO MODELOS
pickle.dump(modelRegression, open('data/modelRegression.pkl', 'wb'))
pickle.dump(modelDiscriminant, open('data/modelDiscriminant.pkl', 'wb'))

#COMPARACION
results_df = results_df.append(results_df_2, ignore_index=True)
print(results_df)