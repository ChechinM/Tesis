# IMPORTANDO LIBRERIAS
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split

# LEYENDA LA DATA
df = pd.read_csv('application_data.csv')

# ELIMINANDO LAS COLUMNAS QUE TIENEN VALORES NULOS MAYOR AL 50%
df = df.dropna(thresh=(len(df)*0.50), axis=1)

# CAMBIANDO COLUMNAS CON TIPO OBJECT A VALORES CATEGORICOS
for feature in df.columns:
    if df[feature].dtype == 'object':
        df[feature] = pd.Categorical(df[feature]).codes

# CAMBIANDO COLUMNAS CON VALORES NULOS A SU PROMEDIO
df.AMT_GOODS_PRICE.fillna(df.AMT_GOODS_PRICE.mean(), inplace=True)
df.AMT_ANNUITY.fillna(df.AMT_ANNUITY.mean(), inplace=True)
df.CNT_FAM_MEMBERS.fillna(df.CNT_FAM_MEMBERS.mean(), inplace=True)
df.EXT_SOURCE_2.fillna(df.EXT_SOURCE_2.mean(), inplace=True)
df.EXT_SOURCE_3.fillna(df.EXT_SOURCE_3.mean(), inplace=True)
df.YEARS_BEGINEXPLUATATION_AVG.fillna(df.YEARS_BEGINEXPLUATATION_AVG.mean(), inplace=True)
# df.ENTRANCES_AVG.fillna(df.ENTRANCES_AVG.mean(), inplace=True)
df.FLOORSMAX_AVG.fillna(df.FLOORSMAX_AVG.mean(), inplace=True)
# df.LIVINGAREA_AVG.fillna(df.LIVINGAREA_AVG.mean(), inplace=True)
df.YEARS_BEGINEXPLUATATION_MODE.fillna(df.YEARS_BEGINEXPLUATATION_MODE.mean(), inplace=True)
# df.ENTRANCES_MODE.fillna(df.ENTRANCES_MODE.mean(), inplace=True)
df.FLOORSMAX_MODE.fillna(df.FLOORSMAX_MODE.mean(), inplace=True)
# df.LIVINGAREA_MODE.fillna(df.LIVINGAREA_MODE.mean(), inplace=True)
df.YEARS_BEGINEXPLUATATION_MEDI.fillna(df.YEARS_BEGINEXPLUATATION_MEDI.mean(), inplace=True)
# df.ENTRANCES_MEDI.fillna(df.ENTRANCES_MEDI.mean(), inplace=True)
df.FLOORSMAX_MEDI.fillna(df.FLOORSMAX_MEDI.mean(), inplace=True)
# df.LIVINGAREA_MEDI.fillna(df.LIVINGAREA_MEDI.mean(), inplace=True)
df.TOTALAREA_MODE.fillna(df.TOTALAREA_MODE.mean(), inplace=True)
df.OBS_30_CNT_SOCIAL_CIRCLE.fillna(df.OBS_30_CNT_SOCIAL_CIRCLE.mean(), inplace=True)
df.DEF_30_CNT_SOCIAL_CIRCLE.fillna(df.DEF_30_CNT_SOCIAL_CIRCLE.mean(), inplace=True)
df.OBS_60_CNT_SOCIAL_CIRCLE.fillna(df.OBS_60_CNT_SOCIAL_CIRCLE.mean(), inplace=True)
df.DEF_60_CNT_SOCIAL_CIRCLE.fillna(df.DEF_60_CNT_SOCIAL_CIRCLE.mean(), inplace=True)
df.DAYS_LAST_PHONE_CHANGE.fillna(df.DAYS_LAST_PHONE_CHANGE.mean(), inplace=True)
df.FLAG_DOCUMENT_2.fillna(df.FLAG_DOCUMENT_2.mean(), inplace=True)
df.FLAG_DOCUMENT_3.fillna(df.FLAG_DOCUMENT_3.mean(), inplace=True)
df.FLAG_DOCUMENT_4.fillna(df.FLAG_DOCUMENT_4.mean(), inplace=True)
df.FLAG_DOCUMENT_5.fillna(df.FLAG_DOCUMENT_5.mean(), inplace=True)
df.FLAG_DOCUMENT_6.fillna(df.FLAG_DOCUMENT_6.mean(), inplace=True)
df.FLAG_DOCUMENT_7.fillna(df.FLAG_DOCUMENT_7.mean(), inplace=True)
df.FLAG_DOCUMENT_8.fillna(df.FLAG_DOCUMENT_8.mean(), inplace=True)
df.FLAG_DOCUMENT_9.fillna(df.FLAG_DOCUMENT_9.mean(), inplace=True)
df.FLAG_DOCUMENT_10.fillna(df.FLAG_DOCUMENT_10.mean(), inplace=True)
df.FLAG_DOCUMENT_11.fillna(df.FLAG_DOCUMENT_11.mean(), inplace=True)
df.FLAG_DOCUMENT_12.fillna(df.FLAG_DOCUMENT_12.mean(), inplace=True)
df.FLAG_DOCUMENT_13.fillna(df.FLAG_DOCUMENT_13.mean(), inplace=True)
df.FLAG_DOCUMENT_14.fillna(df.FLAG_DOCUMENT_14.mean(), inplace=True)
df.FLAG_DOCUMENT_15.fillna(df.FLAG_DOCUMENT_15.mean(), inplace=True)
df.FLAG_DOCUMENT_16.fillna(df.FLAG_DOCUMENT_16.mean(), inplace=True)
df.FLAG_DOCUMENT_17.fillna(df.FLAG_DOCUMENT_17.mean(), inplace=True)
df.FLAG_DOCUMENT_18.fillna(df.FLAG_DOCUMENT_18.mean(), inplace=True)
df.FLAG_DOCUMENT_19.fillna(df.FLAG_DOCUMENT_19.mean(), inplace=True)
df.FLAG_DOCUMENT_20.fillna(df.FLAG_DOCUMENT_20.mean(), inplace=True)
df.FLAG_DOCUMENT_21.fillna(df.FLAG_DOCUMENT_21.mean(), inplace=True)
df.AMT_REQ_CREDIT_BUREAU_HOUR.fillna(df.AMT_REQ_CREDIT_BUREAU_HOUR.mean(), inplace=True)
df.AMT_REQ_CREDIT_BUREAU_DAY.fillna(df.AMT_REQ_CREDIT_BUREAU_DAY.mean(), inplace=True)
df.AMT_REQ_CREDIT_BUREAU_WEEK.fillna(df.AMT_REQ_CREDIT_BUREAU_WEEK.mean(), inplace=True)
df.AMT_REQ_CREDIT_BUREAU_MON.fillna(df.AMT_REQ_CREDIT_BUREAU_MON.mean(), inplace=True)
df.AMT_REQ_CREDIT_BUREAU_QRT.fillna(df.AMT_REQ_CREDIT_BUREAU_QRT.mean(), inplace=True)
df.AMT_REQ_CREDIT_BUREAU_YEAR.fillna(df.AMT_REQ_CREDIT_BUREAU_YEAR.mean(), inplace=True)

# CAMBIANDO EL VALOR DE LAS COLUMNAS A ENTEROS
df['DAYS_REGISTRATION'] = df['DAYS_REGISTRATION'].astype(int, errors='ignore')
df['CNT_FAM_MEMBERS'] = df['CNT_FAM_MEMBERS'].astype(int, errors='ignore')
df['OBS_30_CNT_SOCIAL_CIRCLE'] = df['OBS_30_CNT_SOCIAL_CIRCLE'].astype(int, errors='ignore')
df['DEF_30_CNT_SOCIAL_CIRCLE'] = df['DEF_30_CNT_SOCIAL_CIRCLE'].astype(int, errors='ignore')
df['DAYS_LAST_PHONE_CHANGE'] = df['DAYS_LAST_PHONE_CHANGE'].astype(int, errors='ignore')
df['AMT_REQ_CREDIT_BUREAU_HOUR'] = df['AMT_REQ_CREDIT_BUREAU_HOUR'].astype(int, errors='ignore')
df['DAYS_BIRTH'] = (abs(df['DAYS_BIRTH'])//365.25).astype(int)


# MODELADO

y = df.TARGET.copy()
X = df.drop(['TARGET'], axis=1)

# DIVIDIMOS EL CONJUNTO DE ENTRENAMIENTO Y TEST
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=123)

# ENTRENAMOS LOS DATOS SEGUN EL MODELO DE REGRESION LOGISTICA
modelRegression = LogisticRegression(solver='liblinear')
modelRegression.fit(X_train.values, y_train.values)

# PRECIDIENDO CON EL MODELO PREVIAMENTE ENTRENADO
y_predict = modelRegression.predict(X_test)

# CALCULANDO ALGUNAS METRICAS
results_df = pd.DataFrame(data=[["Regresion Logistica",  metrics.accuracy_score(y_test, y_predict), metrics.precision_score(y_test, y_predict), metrics.recall_score(y_test, y_predict)]],
                          columns=['Modelo', 'Accuracy', 'Precision', 'Recall'])
print(results_df)


# ENTRENAMOS LOS DATOS SEGUN EL MODELO DE ANALISIS DISCRIMINANTE
modelDiscriminant = LinearDiscriminantAnalysis()
modelDiscriminant.fit(X_train.values, y_train.values)

# PRECIDIENDO CON EL MODELO PREVIAMENTE ENTRENADO
y_predict = modelDiscriminant.predict(X_test)

# CALCULANDO ALGUNAS METRICAS
results_df_2 = pd.DataFrame(data=[["Analisis Discriminante",  metrics.accuracy_score(y_test, y_predict), metrics.precision_score(y_test, y_predict), metrics.recall_score(y_test, y_predict)]],
                            columns=['Modelo', 'Accuracy', 'Precision', 'Recall'])
print(results_df_2)


# CREANDO MODELOS
pickle.dump(modelRegression, open('Flask-API/data/modelRegression.pkl', 'wb'))
pickle.dump(modelDiscriminant, open('Flask-API/data/modelDiscriminant.pkl', 'wb'))

# Cuadro Comparativo
results_df = results_df.append(results_df_2, ignore_index=True)
print(results_df)
