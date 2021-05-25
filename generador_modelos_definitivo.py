import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import functools 
import seaborn as sns
import pingouin as pg
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder #Integer encoding
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from scipy.stats import randint as sp_randint
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import tensorflow as tf
from statsmodels.tools.eval_measures import rmse
from statsmodels.tsa.holtwinters import ExponentialSmoothing as HWES
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.callbacks import EarlyStopping
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ParameterGrid
from sklearn import preprocessing
from tensorflow import keras
import xgboost as xgb
import math
import joblib
#Semilla
np.random.seed(7)

"""# Funciones"""

#Con esta función defino el target del dataset que se puede concatenar con la parte numérica o alfabética
#lista contiene el nombre de las columnas que son target
#lista_nombre contiene los nuevos nombres para esas columnas
def target(dataframe,lista,lista_nombre):
  df=dataframe.dropna();
  df_target=pd.DataFrame(columns=lista_nombre)
  for i in range(0,len(lista),1):
    n=df[lista[i]].copy();
    df_target.iloc[:,i]=n;
  return df_target

#se proporciona el dataframe. Además, si desea obtener la parte númerica debe colocar "0" en indicador, por el contrario, "1".
def separacion(dataframe,indicador):
  df=dataframe.dropna();
  count_n=0; count_a=0;
  for i in range(0,df.shape[1],1):
    n1=df.iloc[:,i].copy();
    if str(df.iloc[:,i].dtypes) == 'float64' or str(df.iloc[:,i].dtypes) == 'int64':
      if count_n==0: num=n1.copy(); count_n=1;
      elif count_n==1: num=pd.concat([num,n1],axis=1);
    elif str(df.iloc[:,i].dtypes) == 'object':
      if count_a==0: alf=n1.copy(); count_a=1;
      elif count_a==1: alf=pd.concat([alf,n1],axis=1);
  if indicador==0: return num; print(num);
  elif indicador==1: return alf; print(alf);

#Retrasos en los datos para obtener la mayor correlación. Se hace de 0 a 12 meses.
#se debe proporcionar el "dataframe"
#"variable_analisis" es la variable independiente (sobre la que se desea realizar el análisis). El conteo comienza desde 1, siendo la última variable del dataframe
#el "metodo" depende si se quiere hacer una evaluación de la correlación de pearson o spearman
def retraso(dataframe,variable_analisis,metodo): #n_re=meses de retraso, dataframe, a=sobre que variable se quiere obtener la correlación
  n_re=12; ##n_re=N° de meses de retrazo
  dataframe2=dataframe.copy() #para no modificar el dataframe original
  b=dataframe2.shape[1]-variable_analisis;
  spear=np.empty(shape=(n_re+1,b)) #las filas son el número de retrazos y las columnas son las variables
  for i in range(0,n_re+1,1):
    if i>0: 
      dataframe2.iloc[:,:b]=dataframe2.iloc[:,:b:].shift(1); #dataframe.shape[1]-a es porque se deben retrazar todas menos el target
      df=dataframe2.dropna()
      spear[i,:]=df.corr(metodo).iloc[b:b+1,:b] #spear es una matriz de correlaciones
    elif i==0: #guarde los primeros valores de correlación mostrados en el apartado anterior
      spear[i,:]=dataframe2.corr(metodo).iloc[b:b+1,:b] #spear es una matriz de correlaciones
  spear=abs(spear) #nos interesa la mayor correlación sin importar el signo (por el momento)
  delay=np.empty(shape=b,dtype=int);
  datos=np.empty(shape=b,dtype=object); #1
  #Obtener los retrasos respectivos
  for z in range(0,b,1):
    delay[z]=functools.reduce(lambda sub, ele: sub * 10 + ele, np.where(spear[:,z]==np.amax(spear[:,z])))  #debido a esto contiene todos los retrazos, hay algunos que no se usarán
    datos[z]=str(dataframe2.columns[z])
  print(dataframe2.columns[b])
  print(datos) #1
  lista=delay.tolist(); print(lista) #contiene los retrazos adecuados para cada variable    
  return lista

#se debe ingresar el dataframe y la lista que contiene a los retrazos para cada una de las variables. 
def delay(dataframe, lista): #permite obtener el dataframe con los retrasos respectivos
  count=0
  dataframe2=dataframe.copy()
  for i in lista:
    dataframe2.iloc[:,count]=dataframe2.iloc[:,count].shift(i);
    count=count+1;
  dataframe2=dataframe2.dropna()
  #dataframe2.head(13) para verificar que se hayan hecho los retrazos respectivos
  return dataframe2

#Esta función permite calcular el MAPE, para ello se debe ingresar el Y_actual e Y_Predicted
def MAPE(Y_actual,Y_Predicted):
    mape = np.mean(np.abs((Y_actual - Y_Predicted)/Y_actual))*100
    return mape

"""# Importación de la data"""

#Única entrada al notebook
df=pd.read_excel('BD_actualizado.xlsx')

lista=['RUNT MERCADO','RUNT YAMAHA']; nombre=['TARGET MERCADO','TARGET YAMAHA'];
df_target=target(df,lista,nombre) #se define el target

numericos=separacion(df,0)
numericos=pd.concat([numericos,df_target],axis=1)

numericos_clean4=numericos.drop(['RUNT MERCADO','RUNT YAMAHA','WHOLESALE','ICC','ISM','MES','PIB GROWTH'],axis=1)
numericos_clean4['DIAS HABILES']=numericos_clean4['DIAS HABILES']/(numericos_clean4['DIAS HABILES']+numericos_clean4['FESTIVOS'])
numericos_clean4=numericos_clean4.drop(['FESTIVOS'],axis=1)
numericos_clean4.rename(columns={'DIAS HABILES': 'RATIO_DH_F'},inplace=True) #esta variable tiene estacionalidad

dataset=numericos_clean4.copy()
dataset_mercado=dataset.drop(['TARGET YAMAHA'],axis=1)
dataset_yamaha=dataset.drop(['TARGET MERCADO'],axis=1)

"""# Entrenamiento valores en tiempo t

## Redes Neuronales (RN)

### Yamaha
"""

# Para una sola salida
def preprocesamientoRN(dataframe):
  df=dataframe.copy()
  df=df.values
  X=df[:,0:df.shape[1]-1]
  Y=df[:,df.shape[1]-1:]
  min_max_scaler=preprocessing.MinMaxScaler([-1,1])
  X_scale=min_max_scaler.fit_transform(X)
  Y_scale=min_max_scaler.fit_transform(Y)
  X_test=X_scale[X_scale.shape[0]-4:,:]
  Y_test=Y_scale[Y_scale.shape[0]-4:,:]
  X_scale=X_scale[:X_scale.shape[0]-4,:]
  Y_scale=Y_scale[:Y_scale.shape[0]-4,:]
  X_train, X_valid, Y_train, Y_valid=train_test_split(X_scale,Y_scale, test_size=0.3,random_state=1)
  return min_max_scaler, X_train, X_valid, X_test, Y_train, Y_valid, Y_test

def entrenamientoRN(X_train, Y_train, X_valid, Y_valid):
  model = Sequential([
  Dense(6, activation='relu', input_shape=(X_train.shape[1],)),
  Dense(4, activation='relu'),
  Dense(Y_train.shape[1], activation='tanh'),
  ])
  es = EarlyStopping(monitor='loss', mode='min', verbose=1,patience=10,restore_best_weights=True)
  model.compile(optimizer='adam',
                loss='mean_absolute_error',
                metrics=['mse'])
  hist=model.fit(X_train, Y_train, batch_size=2, epochs=300, validation_data=(X_valid, Y_valid), callbacks=[es])
  return hist, model

scaler, X_train, X_valid, X_test, Y_train, Y_valid, Y_test=preprocesamientoRN(dataset_yamaha)

hist_yamaha, modelRN_yamaha=entrenamientoRN(X_train,Y_train,X_valid,Y_valid)

# Estimaciones
Y_hat_train=modelRN_yamaha.predict(X_train); Y_hat_train=scaler.inverse_transform(Y_hat_train);
Y_hat_valid=modelRN_yamaha.predict(X_valid); Y_hat_valid=scaler.inverse_transform(Y_hat_valid);
Y_hat_test=modelRN_yamaha.predict(X_test); Y_hat_test=scaler.inverse_transform(Y_hat_test);
# Reales
Y_train_normal=scaler.inverse_transform(Y_train)
Y_valid_normal=scaler.inverse_transform(Y_valid)
Y_test_normal=scaler.inverse_transform(Y_test)

#errores=[mean_absolute_error(Y_train_normal,Y_hat_train), mean_absolute_percentage_error(Y_train_normal,Y_hat_train)]
errores=[mean_absolute_error(Y_valid_normal,Y_hat_valid), mean_absolute_percentage_error(Y_valid_normal,Y_hat_valid)]
np.save('error_RNN_actual_Yamaha.npy', errores)
#modelo = np.load('error_RNN_actual_Yamaha.npy')

#Guardar modelo
modelRN_yamaha.save('modelRN_yamaha.h5')

"""### Mercado"""

scaler, X_train, X_valid, X_test, Y_train, Y_valid, Y_test=preprocesamientoRN(dataset_mercado)

hist_mercado, modelRN_mercado=entrenamientoRN(X_train,Y_train,X_valid,Y_valid)

# Estimaciones
Y_hat_train=modelRN_mercado.predict(X_train); Y_hat_train=scaler.inverse_transform(Y_hat_train);
Y_hat_valid=modelRN_mercado.predict(X_valid); Y_hat_valid=scaler.inverse_transform(Y_hat_valid);
Y_hat_test=modelRN_mercado.predict(X_test); Y_hat_test=scaler.inverse_transform(Y_hat_test);
# Reales
Y_train_normal=scaler.inverse_transform(Y_train)
Y_valid_normal=scaler.inverse_transform(Y_valid)
Y_test_normal=scaler.inverse_transform(Y_test)

#errores=[mean_absolute_error(Y_train_normal,Y_hat_train), mean_absolute_percentage_error(Y_train_normal,Y_hat_train)]
errores=[mean_absolute_error(Y_valid_normal,Y_hat_valid), mean_absolute_percentage_error(Y_valid_normal,Y_hat_valid)]
np.save('error_RNN_actual_Mercado.npy', errores)
#modelo = np.load('error_RNN_actual_Mercado.npy')

#Guardar modelo
modelRN_mercado.save('modelRN_mercado.h5')

"""## Random Forest (RF)

### Yamaha
"""

# Para una sola salida
def preprocesamientoRF_XG(dataframe):
  df=dataframe.copy()
  df=df.values
  X=df[:,0:df.shape[1]-1]
  Y=df[:,df.shape[1]-1:]
  X_test=X[X.shape[0]-4:,:]
  Y_test=Y[Y.shape[0]-4:,:].ravel()
  X=X[:X.shape[0]-4,:]
  Y=Y[:Y.shape[0]-4,:].ravel()
  X_train, X_valid, Y_train, Y_valid=train_test_split(X,Y, test_size=0.3,random_state=1)
  return X_train, X_valid, X_test, Y_train, Y_valid, Y_test

X_train, X_valid, X_test, Y_train, Y_valid, Y_test=preprocesamientoRF_XG(dataset_yamaha)

# Grid de hiperparámetros evaluados
# ==============================================================================
param_grid = ParameterGrid( #los mejores parámetros
                {'n_estimators': range(20,41,1),
                 'max_features': ['auto','sqrt'],
                 'max_depth'   : range(20,41,1),
                }
             )

# Loop para ajustar un modelo con cada combinación de hiperparámetros
# ==============================================================================
resultados = {'params': [], 'rmse_train': [], 'rmse_valid': [], 'mae_train':[], 'mae_valid':[], 'mape_train':[], 'mape_valid':[]}

for params in param_grid:
    
    modeloRF_yamaha = RandomForestRegressor(
                n_jobs       = -1, #usa todos los procesos
                random_state = 1,
                criterion='mse',
                bootstrap=True,
                ** params
             )
    
    modeloRF_yamaha.fit(X_train, Y_train)
    y_hat_train=modeloRF_yamaha.predict(X_train)
    y_hat_valid=modeloRF_yamaha.predict(X_valid)
    mse_train=mean_squared_error(Y_train, y_hat_train)
    mse_valid=mean_squared_error(Y_valid, y_hat_valid)
    mae_train=mean_absolute_error(Y_train, y_hat_train)
    mae_valid=mean_absolute_error(Y_valid, y_hat_valid)
    mape_train=MAPE(Y_train, y_hat_train)
    mape_valid=MAPE(Y_valid, y_hat_valid)

    
    resultados['params'].append(params)
    resultados['rmse_train'].append(math.sqrt(mse_train))
    resultados['rmse_valid'].append(math.sqrt(mse_valid))
    resultados['mae_train'].append(mae_train)
    resultados['mae_valid'].append(mae_valid)
    resultados['mape_train'].append(mape_train)
    resultados['mape_valid'].append(mape_valid)


    print(f"Modelo: {params} \u2713")

# Resultados
# ==============================================================================
resultados = pd.DataFrame(resultados)
resultados = pd.concat([resultados, resultados['params'].apply(pd.Series)], axis=1)
resultados = resultados.drop(columns = 'params')
resultados = resultados.sort_values('mape_valid', ascending=True)
best=resultados.head(1)

"""Mejor modelo"""

# Grid de hiperparámetros evaluados
# ==============================================================================
param_grid = ParameterGrid( #los mejores parámetros
                {'n_estimators': best['n_estimators'].values,
                 'max_features': best['max_features'].values,
                 'max_depth'   : best['max_depth'].values,
                }
             )

# Loop para ajustar un modelo con cada combinación de hiperparámetros
# ==============================================================================
resultados = {'params': [], 'rmse_train': [], 'rmse_valid': [], 'mae_train':[], 'mae_valid':[], 'mape_train':[], 'mape_valid':[]}

for params in param_grid:
    
    modeloRF_yamaha = RandomForestRegressor(
                n_jobs       = -1, #usa todos los procesos
                random_state = 1,
                criterion='mse',
                bootstrap=True,
                ** params
             )
    
    modeloRF_yamaha.fit(X_train, Y_train)
    y_hat_train=modeloRF_yamaha.predict(X_train)
    y_hat_valid=modeloRF_yamaha.predict(X_valid)
    mse_train=mean_squared_error(Y_train, y_hat_train)
    mse_valid=mean_squared_error(Y_valid, y_hat_valid)
    mae_train=mean_absolute_error(Y_train, y_hat_train)
    mae_valid=mean_absolute_error(Y_valid, y_hat_valid)
    mape_train=MAPE(Y_train, y_hat_train)
    mape_valid=MAPE(Y_valid, y_hat_valid)

    
    resultados['params'].append(params)
    resultados['rmse_train'].append(math.sqrt(mse_train))
    resultados['rmse_valid'].append(math.sqrt(mse_valid))
    resultados['mae_train'].append(mae_train)
    resultados['mae_valid'].append(mae_valid)
    resultados['mape_train'].append(mape_train)
    resultados['mape_valid'].append(mape_valid)


    print(f"Modelo: {params} \u2713")

# Resultados
# ==============================================================================
resultados = pd.DataFrame(resultados)
resultados = pd.concat([resultados, resultados['params'].apply(pd.Series)], axis=1)
resultados = resultados.drop(columns = 'params')
resultados = resultados.sort_values('mape_valid', ascending=True)
resultados.head(4)

Y_train=Y_train.reshape([-1,1])
y_hat_train=y_hat_train.reshape([-1,1])
Y_valid=Y_valid.reshape([-1,1])
y_hat_valid=y_hat_valid.reshape([-1,1])
Y_test=Y_test.reshape([-1,1])
y_hat_test=modeloRF_yamaha.predict(X_test)
y_hat_test=y_hat_test.reshape([-1,1])

#errores=[mean_absolute_error(Y_train,y_hat_train), mean_absolute_percentage_error(Y_train,y_hat_train)]
errores=[mean_absolute_error(Y_valid,y_hat_valid), mean_absolute_percentage_error(Y_valid,y_hat_valid)]
np.save('error_RF_actual_Yamaha.npy', errores)
#modelo = np.load('error_RF_actual_Yamaha.npy')

# save the model to disk
joblib.dump(modeloRF_yamaha, 'modeloRF_yamaha.pkl')
 
"""### Mercado"""

X_train, X_valid, X_test, Y_train, Y_valid, Y_test=preprocesamientoRF_XG(dataset_mercado)

# Grid de hiperparámetros evaluados
# ==============================================================================
param_grid = ParameterGrid( #los mejores parámetros
                {'n_estimators': range(20,41,2),
                 'max_features': ['auto','sqrt'],
                 'max_depth'   : range(20,41,2),
                }
             )

# Loop para ajustar un modelo con cada combinación de hiperparámetros
# ==============================================================================
resultados = {'params': [], 'rmse_train': [], 'rmse_valid': [], 'mae_train':[], 'mae_valid':[], 'mape_train':[], 'mape_valid':[]}

for params in param_grid:
    
    modeloRF_mercado = RandomForestRegressor(
                n_jobs       = -1, #usa todos los procesos
                random_state = 1,
                criterion='mse',
                bootstrap=True,
                ** params
             )
    
    modeloRF_mercado.fit(X_train, Y_train)
    y_hat_train=modeloRF_mercado.predict(X_train)
    y_hat_valid=modeloRF_mercado.predict(X_valid)
    mse_train=mean_squared_error(Y_train, y_hat_train)
    mse_valid=mean_squared_error(Y_valid, y_hat_valid)
    mae_train=mean_absolute_error(Y_train, y_hat_train)
    mae_valid=mean_absolute_error(Y_valid, y_hat_valid)
    mape_train=MAPE(Y_train, y_hat_train)
    mape_valid=MAPE(Y_valid, y_hat_valid)

    
    resultados['params'].append(params)
    resultados['rmse_train'].append(math.sqrt(mse_train))
    resultados['rmse_valid'].append(math.sqrt(mse_valid))
    resultados['mae_train'].append(mae_train)
    resultados['mae_valid'].append(mae_valid)
    resultados['mape_train'].append(mape_train)
    resultados['mape_valid'].append(mape_valid)


    print(f"Modelo: {params} \u2713")

# Resultados
# ==============================================================================
resultados = pd.DataFrame(resultados)
resultados = pd.concat([resultados, resultados['params'].apply(pd.Series)], axis=1)
resultados = resultados.drop(columns = 'params')
resultados = resultados.sort_values('mape_valid', ascending=True)
best=resultados.head(1)

# Grid de hiperparámetros evaluados
# ==============================================================================
param_grid = ParameterGrid( #los mejores parámetros
                {'n_estimators': best['n_estimators'].values,
                 'max_features': best['max_features'].values,
                 'max_depth'   : best['max_depth'].values,
                }
             )

# Loop para ajustar un modelo con cada combinación de hiperparámetros
# ==============================================================================
resultados = {'params': [], 'rmse_train': [], 'rmse_valid': [], 'mae_train':[], 'mae_valid':[], 'mape_train':[], 'mape_valid':[]}

for params in param_grid:
    
    modeloRF_mercado = RandomForestRegressor(
                n_jobs       = -1, #usa todos los procesos
                random_state = 1,
                criterion='mse',
                bootstrap=True,
                ** params
             )
    
    modeloRF_mercado.fit(X_train, Y_train)
    y_hat_train=modeloRF_mercado.predict(X_train)
    y_hat_valid=modeloRF_mercado.predict(X_valid)
    mse_train=mean_squared_error(Y_train, y_hat_train)
    mse_valid=mean_squared_error(Y_valid, y_hat_valid)
    mae_train=mean_absolute_error(Y_train, y_hat_train)
    mae_valid=mean_absolute_error(Y_valid, y_hat_valid)
    mape_train=MAPE(Y_train, y_hat_train)
    mape_valid=MAPE(Y_valid, y_hat_valid)

    
    resultados['params'].append(params)
    resultados['rmse_train'].append(math.sqrt(mse_train))
    resultados['rmse_valid'].append(math.sqrt(mse_valid))
    resultados['mae_train'].append(mae_train)
    resultados['mae_valid'].append(mae_valid)
    resultados['mape_train'].append(mape_train)
    resultados['mape_valid'].append(mape_valid)


    print(f"Modelo: {params} \u2713")

# Resultados
# ==============================================================================
resultados = pd.DataFrame(resultados)
resultados = pd.concat([resultados, resultados['params'].apply(pd.Series)], axis=1)
resultados = resultados.drop(columns = 'params')
resultados = resultados.sort_values('mape_valid', ascending=True)
resultados.head(4)

Y_train=Y_train.reshape([-1,1])
y_hat_train=y_hat_train.reshape([-1,1])
Y_valid=Y_valid.reshape([-1,1])
y_hat_valid=y_hat_valid.reshape([-1,1])
Y_test=Y_test.reshape([-1,1])
y_hat_test=modeloRF_mercado.predict(X_test)
y_hat_test=y_hat_test.reshape([-1,1])

#errores=[mean_absolute_error(Y_train,y_hat_train), mean_absolute_percentage_error(Y_train,y_hat_train)]
errores=[mean_absolute_error(Y_valid,y_hat_valid), mean_absolute_percentage_error(Y_valid,y_hat_valid)]
np.save('error_RF_actual_Mercado.npy', errores)
#modelo = np.load('error_RF_actual_Mercado.npy')

# save the model to disk
joblib.dump(modeloRF_mercado, 'modeloRF_mercado.pkl')

"""## XGBoost Tree (XG)

### Yamaha
"""

X_train, X_valid, X_test, Y_train, Y_valid, Y_test=preprocesamientoRF_XG(dataset_yamaha)

#Grid de hiperparámetros evaluados
# ==============================================================================
param_grid = ParameterGrid( #los mejores parámetros
                {'colsample_bytree':[0.4,0.5,0.6,0.7,0.8],
                 'subsample':[0.4,0.5,0.6,0.7,0.8],
                 'gamma': [5,10],
                 'max_depth'   : range(5,25,5),
                 'n_estimators': range(10,60,10),
                 'reg_lambda':[0,1]
                }
             )

# Loop para ajustar un modelo con cada combinación de hiperparámetros
# ==============================================================================
resultados = {'params': [], 'rmse_train': [], 'rmse_valid': [], 'mae_train':[], 'mae_valid':[], 'mape_train':[], 'mape_valid':[]}

for params in param_grid:
    
    modeloXG_yamaha=xgb.XGBRegressor(objectcive ='reg:squarederror',
                                              random_state=1, 
                                              ** params)
    
    modeloXG_yamaha.fit(X_train, Y_train)
    y_hat_train=modeloXG_yamaha.predict(X_train)
    y_hat_valid=modeloXG_yamaha.predict(X_valid)
    mse_train=mean_squared_error(Y_train, y_hat_train)
    mse_valid=mean_squared_error(Y_valid, y_hat_valid)
    mae_train=mean_absolute_error(Y_train, y_hat_train)
    mae_valid=mean_absolute_error(Y_valid, y_hat_valid)
    y_hat_train=np.matrix(y_hat_train)
    #y_hat_train=y_hat_train.T
    y_hat_valid=np.matrix(y_hat_valid)
    #y_hat_valid=y_hat_valid.T
    mape_train=MAPE(Y_train, y_hat_train)
    mape_valid=MAPE(Y_valid, y_hat_valid)

    
    resultados['params'].append(params)
    resultados['rmse_train'].append(math.sqrt(mse_train))
    resultados['rmse_valid'].append(math.sqrt(mse_valid))
    resultados['mae_train'].append(mae_train)
    resultados['mae_valid'].append(mae_valid)
    resultados['mape_train'].append(mape_train)
    resultados['mape_valid'].append(mape_valid)
    


    print(f"Modelo: {params} \u2713")

# Resultados
# ==============================================================================
resultados = pd.DataFrame(resultados)
resultados = pd.concat([resultados, resultados['params'].apply(pd.Series)], axis=1)
resultados = resultados.drop(columns = 'params')
resultados = resultados.sort_values('rmse_valid', ascending=True)
best=resultados.head(1)

"""Mejor modelo"""

#Grid de hiperparámetros evaluados
# ==============================================================================
param_grid = ParameterGrid( #los mejores parámetros
                {'colsample_bytree':best['colsample_bytree'].values,
                 'subsample':best['subsample'].values,
                 'gamma': best['gamma'].values,
                 'max_depth'   : [int(best['max_depth'].values)],
                 'n_estimators': [int(best['n_estimators'].values)],
                 'reg_lambda':best['reg_lambda'].values
                }
             )

# Loop para ajustar un modelo con cada combinación de hiperparámetros
# ==============================================================================
resultados = {'params': [], 'rmse_train': [], 'rmse_valid': [], 'mae_train':[], 'mae_valid':[], 'mape_train':[], 'mape_valid':[]}

for params in param_grid:
    
    modeloXG_yamaha=xgb.XGBRegressor(objectcive ='reg:squarederror',
                                              random_state=1, 
                                              ** params)
    
    modeloXG_yamaha.fit(X_train, Y_train)
    y_hat_train=modeloXG_yamaha.predict(X_train)
    y_hat_valid=modeloXG_yamaha.predict(X_valid)
    mse_train=mean_squared_error(Y_train, y_hat_train)
    mse_valid=mean_squared_error(Y_valid, y_hat_valid)
    mae_train=mean_absolute_error(Y_train, y_hat_train)
    mae_valid=mean_absolute_error(Y_valid, y_hat_valid)
    y_hat_train=np.matrix(y_hat_train)
    #y_hat_train=y_hat_train.T
    y_hat_valid=np.matrix(y_hat_valid)
    #y_hat_valid=y_hat_valid.T
    mape_train=MAPE(Y_train, y_hat_train)
    mape_valid=MAPE(Y_valid, y_hat_valid)

    
    resultados['params'].append(params)
    resultados['rmse_train'].append(math.sqrt(mse_train))
    resultados['rmse_valid'].append(math.sqrt(mse_valid))
    resultados['mae_train'].append(mae_train)
    resultados['mae_valid'].append(mae_valid)
    resultados['mape_train'].append(mape_train)
    resultados['mape_valid'].append(mape_valid)
    


    print(f"Modelo: {params} \u2713")

# Resultados
# ==============================================================================
resultados = pd.DataFrame(resultados)
resultados = pd.concat([resultados, resultados['params'].apply(pd.Series)], axis=1)
resultados = resultados.drop(columns = 'params')
resultados = resultados.sort_values('rmse_valid', ascending=True)
resultados.head(4)

Y_train=Y_train.reshape([-1,1])
y_hat_train=y_hat_train.reshape([-1,1])
Y_valid=Y_valid.reshape([-1,1])
y_hat_valid=y_hat_valid.reshape([-1,1])
Y_test=Y_test.reshape([-1,1])
y_hat_test=modeloXG_yamaha.predict(X_test)
y_hat_test=y_hat_test.reshape([-1,1])

#errores=[mean_absolute_error(Y_train,y_hat_train), mean_absolute_percentage_error(Y_train,y_hat_train)]
errores=[mean_absolute_error(Y_valid,y_hat_valid), mean_absolute_percentage_error(Y_valid,y_hat_valid)]
np.save('error_XG_actual_Yamaha.npy', errores)
#modelo = np.load('error_XG_actual_Yamaha.npy')

# save the model to disk
joblib.dump(modeloXG_yamaha, 'modeloXG_yamaha.pkl')

"""### Mercado"""

X_train, X_valid, X_test, Y_train, Y_valid, Y_test=preprocesamientoRF_XG(dataset_mercado)

#Grid de hiperparámetros evaluados
# ==============================================================================
param_grid = ParameterGrid( #los mejores parámetros
                {'colsample_bytree':[0.4,0.5,0.6,0.7,0.8],
                 'subsample':[0.4,0.5,0.6,0.7,0.8],
                 'gamma': [5,10],
                 'max_depth'   : range(5,25,5),
                 'n_estimators': range(10,60,10),
                 'reg_lambda':[0,1]
                }
             )

# Loop para ajustar un modelo con cada combinación de hiperparámetros
# ==============================================================================
resultados = {'params': [], 'rmse_train': [], 'rmse_valid': [], 'mae_train':[], 'mae_valid':[], 'mape_train':[], 'mape_valid':[]}

for params in param_grid:
    
    modeloXG_mercado=xgb.XGBRegressor(objectcive ='reg:squarederror',
                                              random_state=1, 
                                              ** params)
    
    modeloXG_mercado.fit(X_train, Y_train)
    y_hat_train=modeloXG_mercado.predict(X_train)
    y_hat_valid=modeloXG_mercado.predict(X_valid)
    mse_train=mean_squared_error(Y_train, y_hat_train)
    mse_valid=mean_squared_error(Y_valid, y_hat_valid)
    mae_train=mean_absolute_error(Y_train, y_hat_train)
    mae_valid=mean_absolute_error(Y_valid, y_hat_valid)
    y_hat_train=np.matrix(y_hat_train)
    #y_hat_train=y_hat_train.T
    y_hat_valid=np.matrix(y_hat_valid)
    #y_hat_valid=y_hat_valid.T
    mape_train=MAPE(Y_train, y_hat_train)
    mape_valid=MAPE(Y_valid, y_hat_valid)

    
    resultados['params'].append(params)
    resultados['rmse_train'].append(math.sqrt(mse_train))
    resultados['rmse_valid'].append(math.sqrt(mse_valid))
    resultados['mae_train'].append(mae_train)
    resultados['mae_valid'].append(mae_valid)
    resultados['mape_train'].append(mape_train)
    resultados['mape_valid'].append(mape_valid)
    


    print(f"Modelo: {params} \u2713")

# Resultados
# ==============================================================================
resultados = pd.DataFrame(resultados)
resultados = pd.concat([resultados, resultados['params'].apply(pd.Series)], axis=1)
resultados = resultados.drop(columns = 'params')
resultados = resultados.sort_values('rmse_valid', ascending=True)
best=resultados.head(1)

"""Mejor modelo"""

#Grid de hiperparámetros evaluados
# ==============================================================================
param_grid = ParameterGrid( #los mejores parámetros
                {'colsample_bytree':best['colsample_bytree'].values,
                 'subsample':best['subsample'].values,
                 'gamma': best['gamma'].values,
                 'max_depth'   : [int(best['max_depth'].values)],
                 'n_estimators': [int(best['n_estimators'].values)],
                 'reg_lambda':best['reg_lambda'].values
                }
             )

# Loop para ajustar un modelo con cada combinación de hiperparámetros
# ==============================================================================
resultados = {'params': [], 'rmse_train': [], 'rmse_valid': [], 'mae_train':[], 'mae_valid':[], 'mape_train':[], 'mape_valid':[]}

for params in param_grid:
    
    modeloXG_mercado=xgb.XGBRegressor(objectcive ='reg:squarederror',
                                              random_state=1, 
                                              ** params)
    
    modeloXG_mercado.fit(X_train, Y_train)
    y_hat_train=modeloXG_mercado.predict(X_train)
    y_hat_valid=modeloXG_mercado.predict(X_valid)
    mse_train=mean_squared_error(Y_train, y_hat_train)
    mse_valid=mean_squared_error(Y_valid, y_hat_valid)
    mae_train=mean_absolute_error(Y_train, y_hat_train)
    mae_valid=mean_absolute_error(Y_valid, y_hat_valid)
    y_hat_train=np.matrix(y_hat_train)
    #y_hat_train=y_hat_train.T
    y_hat_valid=np.matrix(y_hat_valid)
    #y_hat_valid=y_hat_valid.T
    mape_train=MAPE(Y_train, y_hat_train)
    mape_valid=MAPE(Y_valid, y_hat_valid)

    
    resultados['params'].append(params)
    resultados['rmse_train'].append(math.sqrt(mse_train))
    resultados['rmse_valid'].append(math.sqrt(mse_valid))
    resultados['mae_train'].append(mae_train)
    resultados['mae_valid'].append(mae_valid)
    resultados['mape_train'].append(mape_train)
    resultados['mape_valid'].append(mape_valid)
    


    print(f"Modelo: {params} \u2713")

# Resultados
# ==============================================================================
resultados = pd.DataFrame(resultados)
resultados = pd.concat([resultados, resultados['params'].apply(pd.Series)], axis=1)
resultados = resultados.drop(columns = 'params')
resultados = resultados.sort_values('rmse_valid', ascending=True)
resultados.head(4)

Y_train=Y_train.reshape([-1,1])
y_hat_train=y_hat_train.reshape([-1,1])
Y_valid=Y_valid.reshape([-1,1])
y_hat_valid=y_hat_valid.reshape([-1,1])
Y_test=Y_test.reshape([-1,1])
y_hat_test=modeloXG_mercado.predict(X_test)
y_hat_test=y_hat_test.reshape([-1,1])

#errores=[mean_absolute_error(Y_train,y_hat_train), mean_absolute_percentage_error(Y_train,y_hat_train)]
errores=[mean_absolute_error(Y_valid,y_hat_valid), mean_absolute_percentage_error(Y_valid,y_hat_valid)]
np.save('error_XG_actual_Mercado.npy', errores)
#modelo = np.load('error_XG_actual_Mercado.npy')

# save the model to disk
joblib.dump(modeloXG_mercado, 'modeloXG_mercado.pkl')


"""# Entrenamiento con variables rezagadas (t-12)

## RN

### Yamaha
"""

def dataset_rezagado_yamaha3(df):
  df2=df.copy()
  df2=df2.drop(['IEE','ICE'],axis=1)
  target_sin_rezago=df2['TARGET YAMAHA']
  lista=[12,12,12,12,12,0,12,12]
  df3=delay(df2,lista)
  count=0
  for i in lista:
    df3.rename(columns={df3.columns[count]:(df3.columns[count]+'_'+str(i))},inplace=True) #agregar indicador de retardo
    count+= 1
  df3=pd.concat([df3,target_sin_rezago],axis=1)
  df3=df3.dropna()
  df3=df3.reset_index(drop=True)
  return df3

dataset_r_yamaha=dataset_rezagado_yamaha3(dataset)

scaler, X_train, X_valid, X_test, Y_train, Y_valid, Y_test=preprocesamientoRN(dataset_r_yamaha)

hist_r_yamaha, modelRN_r_yamaha=entrenamientoRN(X_train,Y_train,X_valid,Y_valid)

# Estimaciones
Y_hat_train=modelRN_r_yamaha.predict(X_train); Y_hat_train=scaler.inverse_transform(Y_hat_train);
Y_hat_valid=modelRN_r_yamaha.predict(X_valid); Y_hat_valid=scaler.inverse_transform(Y_hat_valid);
Y_hat_test=modelRN_r_yamaha.predict(X_test); Y_hat_test=scaler.inverse_transform(Y_hat_test);
# Reales
Y_train_normal=scaler.inverse_transform(Y_train)
Y_valid_normal=scaler.inverse_transform(Y_valid)
Y_test_normal=scaler.inverse_transform(Y_test)

#errores=[mean_absolute_error(Y_train_normal,Y_hat_train), mean_absolute_percentage_error(Y_train_normal,Y_hat_train)]
errores=[mean_absolute_error(Y_valid_normal,Y_hat_valid), mean_absolute_percentage_error(Y_valid_normal,Y_hat_valid)]
np.save('error_RN_rez_Yamaha.npy', errores)
#modelo = np.load('error_RN_rez_Yamaha.npy')

#Guardar modelo
modelRN_r_yamaha.save('modelRN_r_yamaha.h5')

"""### Mercado"""

def dataset_rezagado_mercado(df):
  df2=df.copy()
  df2=df2.drop(['IEE','ICE','TARGET YAMAHA'],axis=1)
  target_sin_rezago=df2['TARGET MERCADO']
  lista=[12,12,12,12,12,0,12]
  df3=delay(df2,lista)
  count=0
  for i in lista:
    df3.rename(columns={df3.columns[count]:(df3.columns[count]+'_'+str(i))},inplace=True) #agregar indicador de retardo
    count+= 1
  df3=pd.concat([df3,target_sin_rezago],axis=1)
  df3=df3.dropna()
  df3=df3.reset_index(drop=True)
  return df3

dataset_r_mercado=dataset_rezagado_mercado(dataset)

scaler, X_train, X_valid, X_test, Y_train, Y_valid, Y_test=preprocesamientoRN(dataset_r_mercado)

hist_r_mercado, modelRN_r_mercado=entrenamientoRN(X_train,Y_train,X_valid,Y_valid)

plot_loss(hist_r_mercado)

# Estimaciones
Y_hat_train=modelRN_r_mercado.predict(X_train); Y_hat_train=scaler.inverse_transform(Y_hat_train);
Y_hat_valid=modelRN_r_mercado.predict(X_valid); Y_hat_valid=scaler.inverse_transform(Y_hat_valid);
Y_hat_test=modelRN_r_mercado.predict(X_test); Y_hat_test=scaler.inverse_transform(Y_hat_test);
# Reales
Y_train_normal=scaler.inverse_transform(Y_train)
Y_valid_normal=scaler.inverse_transform(Y_valid)
Y_test_normal=scaler.inverse_transform(Y_test)

#errores=[mean_absolute_error(Y_train_normal,Y_hat_train), mean_absolute_percentage_error(Y_train_normal,Y_hat_train)]
errores=[mean_absolute_error(Y_valid_normal,Y_hat_valid), mean_absolute_percentage_error(Y_valid_normal,Y_hat_valid)]
np.save('error_RN_rez_Mercado.npy', errores)
#modelo = np.load('error_RN_rez_Mercado.npy')
#print(modelo[0])
#print(modelo[1])

#Guardar modelo
modelRN_r_mercado.save('modelRN_r_mercado.h5')

"""## RF

### Yamaha
"""

X_train, X_valid, X_test, Y_train, Y_valid, Y_test=preprocesamientoRF_XG(dataset_r_yamaha)

# Grid de hiperparámetros evaluados
# ==============================================================================
param_grid = ParameterGrid( #los mejores parámetros
                {'n_estimators': range(20,41,1),
                 'max_features': ['auto','sqrt'],
                 'max_depth'   : [20],
                }
             )

# Loop para ajustar un modelo con cada combinación de hiperparámetros
# ==============================================================================
resultados = {'params': [], 'rmse_train': [], 'rmse_valid': [], 'mae_train':[], 'mae_valid':[], 'mape_train':[], 'mape_valid':[]}

for params in param_grid:
    
    modeloRF_r_yamaha = RandomForestRegressor(
                n_jobs       = -1, #usa todos los procesos
                random_state = 1,
                criterion='mse',
                bootstrap=True,
                ** params
             )
    
    modeloRF_r_yamaha.fit(X_train, Y_train)
    y_hat_train=modeloRF_r_yamaha.predict(X_train)
    y_hat_valid=modeloRF_r_yamaha.predict(X_valid)
    mse_train=mean_squared_error(Y_train, y_hat_train)
    mse_valid=mean_squared_error(Y_valid, y_hat_valid)
    mae_train=mean_absolute_error(Y_train, y_hat_train)
    mae_valid=mean_absolute_error(Y_valid, y_hat_valid)
    mape_train=MAPE(Y_train, y_hat_train)
    mape_valid=MAPE(Y_valid, y_hat_valid)

    
    resultados['params'].append(params)
    resultados['rmse_train'].append(math.sqrt(mse_train))
    resultados['rmse_valid'].append(math.sqrt(mse_valid))
    resultados['mae_train'].append(mae_train)
    resultados['mae_valid'].append(mae_valid)
    resultados['mape_train'].append(mape_train)
    resultados['mape_valid'].append(mape_valid)


    print(f"Modelo: {params} \u2713")

# Resultados
# ==============================================================================
resultados = pd.DataFrame(resultados)
resultados = pd.concat([resultados, resultados['params'].apply(pd.Series)], axis=1)
resultados = resultados.drop(columns = 'params')
resultados = resultados.sort_values('mape_valid', ascending=True)
best=resultados.head(1)

"""Mejor modelo"""

# Grid de hiperparámetros evaluados
# ==============================================================================
param_grid = ParameterGrid( #los mejores parámetros
                {'n_estimators': best['n_estimators'].values,
                 'max_features': best['max_features'].values,
                 'max_depth'   : best['max_depth'].values,
                }
             )

# Loop para ajustar un modelo con cada combinación de hiperparámetros
# ==============================================================================
resultados = {'params': [], 'rmse_train': [], 'rmse_valid': [], 'mae_train':[], 'mae_valid':[], 'mape_train':[], 'mape_valid':[]}

for params in param_grid:
    
    modeloRF_r_yamaha = RandomForestRegressor(
                n_jobs       = -1, #usa todos los procesos
                random_state = 1,
                criterion='mse',
                bootstrap=True,
                ** params
             )
    
    modeloRF_r_yamaha.fit(X_train, Y_train)
    y_hat_train=modeloRF_r_yamaha.predict(X_train)
    y_hat_valid=modeloRF_r_yamaha.predict(X_valid)
    mse_train=mean_squared_error(Y_train, y_hat_train)
    mse_valid=mean_squared_error(Y_valid, y_hat_valid)
    mae_train=mean_absolute_error(Y_train, y_hat_train)
    mae_valid=mean_absolute_error(Y_valid, y_hat_valid)
    mape_train=MAPE(Y_train, y_hat_train)
    mape_valid=MAPE(Y_valid, y_hat_valid)

    
    resultados['params'].append(params)
    resultados['rmse_train'].append(math.sqrt(mse_train))
    resultados['rmse_valid'].append(math.sqrt(mse_valid))
    resultados['mae_train'].append(mae_train)
    resultados['mae_valid'].append(mae_valid)
    resultados['mape_train'].append(mape_train)
    resultados['mape_valid'].append(mape_valid)


    print(f"Modelo: {params} \u2713")

# Resultados
# ==============================================================================
resultados = pd.DataFrame(resultados)
resultados = pd.concat([resultados, resultados['params'].apply(pd.Series)], axis=1)
resultados = resultados.drop(columns = 'params')
resultados = resultados.sort_values('mape_valid', ascending=True)
resultados.head(4)

Y_train=Y_train.reshape([-1,1])
y_hat_train=y_hat_train.reshape([-1,1])
Y_valid=Y_valid.reshape([-1,1])
y_hat_valid=y_hat_valid.reshape([-1,1])
Y_test=Y_test.reshape([-1,1])
y_hat_test=modeloRF_r_yamaha.predict(X_test)
y_hat_test=y_hat_test.reshape([-1,1])

#errores=[mean_absolute_error(Y_train,y_hat_train), mean_absolute_percentage_error(Y_train,y_hat_train)]
errores=[mean_absolute_error(Y_valid,y_hat_valid), mean_absolute_percentage_error(Y_valid,y_hat_valid)]
np.save('error_RF_rez_Yamaha.npy', errores)
#modelo = np.load('error_RF_rez_Yamaha.npy')
#print(modelo[0])
#print(modelo[1])

# save the model to disk
joblib.dump(modeloRF_r_yamaha, 'modeloRF_r_yamaha.pkl')

"""### Mercado"""

dataset_r_mercado

X_train, X_valid, X_test, Y_train, Y_valid, Y_test=preprocesamientoRF_XG(dataset_r_mercado)

# Grid de hiperparámetros evaluados
# ==============================================================================
param_grid = ParameterGrid( #los mejores parámetros
                {'n_estimators': range(20,41,2),
                 'max_features': ['auto','sqrt'],
                 'max_depth'   : range(20,41,2),
                }
             )

# Loop para ajustar un modelo con cada combinación de hiperparámetros
# ==============================================================================
resultados = {'params': [], 'rmse_train': [], 'rmse_valid': [], 'mae_train':[], 'mae_valid':[], 'mape_train':[], 'mape_valid':[]}

for params in param_grid:
    
    modeloRF_r_mercado = RandomForestRegressor(
                n_jobs       = -1, #usa todos los procesos
                random_state = 1,
                criterion='mse',
                bootstrap=True,
                ** params
             )
    
    modeloRF_r_mercado.fit(X_train, Y_train)
    y_hat_train=modeloRF_r_mercado.predict(X_train)
    y_hat_valid=modeloRF_r_mercado.predict(X_valid)
    mse_train=mean_squared_error(Y_train, y_hat_train)
    mse_valid=mean_squared_error(Y_valid, y_hat_valid)
    mae_train=mean_absolute_error(Y_train, y_hat_train)
    mae_valid=mean_absolute_error(Y_valid, y_hat_valid)
    mape_train=MAPE(Y_train, y_hat_train)
    mape_valid=MAPE(Y_valid, y_hat_valid)

    
    resultados['params'].append(params)
    resultados['rmse_train'].append(math.sqrt(mse_train))
    resultados['rmse_valid'].append(math.sqrt(mse_valid))
    resultados['mae_train'].append(mae_train)
    resultados['mae_valid'].append(mae_valid)
    resultados['mape_train'].append(mape_train)
    resultados['mape_valid'].append(mape_valid)


    print(f"Modelo: {params} \u2713")

# Resultados
# ==============================================================================
resultados = pd.DataFrame(resultados)
resultados = pd.concat([resultados, resultados['params'].apply(pd.Series)], axis=1)
resultados = resultados.drop(columns = 'params')
resultados = resultados.sort_values('mape_valid', ascending=True)
best=resultados.head(1)

"""Mejor modelo"""

# Grid de hiperparámetros evaluados
# ==============================================================================
param_grid = ParameterGrid( #los mejores parámetros
                {'n_estimators': best['n_estimators'].values,
                 'max_features': best['max_features'].values,
                 'max_depth'   : best['max_depth'].values,
                }
             )

# Loop para ajustar un modelo con cada combinación de hiperparámetros
# ==============================================================================
resultados = {'params': [], 'rmse_train': [], 'rmse_valid': [], 'mae_train':[], 'mae_valid':[], 'mape_train':[], 'mape_valid':[]}

for params in param_grid:
    
    modeloRF_r_mercado = RandomForestRegressor(
                n_jobs       = -1, #usa todos los procesos
                random_state = 1,
                criterion='mse',
                bootstrap=True,
                ** params
             )
    
    modeloRF_r_mercado.fit(X_train, Y_train)
    y_hat_train=modeloRF_r_mercado.predict(X_train)
    y_hat_valid=modeloRF_r_mercado.predict(X_valid)
    mse_train=mean_squared_error(Y_train, y_hat_train)
    mse_valid=mean_squared_error(Y_valid, y_hat_valid)
    mae_train=mean_absolute_error(Y_train, y_hat_train)
    mae_valid=mean_absolute_error(Y_valid, y_hat_valid)
    mape_train=MAPE(Y_train, y_hat_train)
    mape_valid=MAPE(Y_valid, y_hat_valid)

    
    resultados['params'].append(params)
    resultados['rmse_train'].append(math.sqrt(mse_train))
    resultados['rmse_valid'].append(math.sqrt(mse_valid))
    resultados['mae_train'].append(mae_train)
    resultados['mae_valid'].append(mae_valid)
    resultados['mape_train'].append(mape_train)
    resultados['mape_valid'].append(mape_valid)


    print(f"Modelo: {params} \u2713")

# Resultados
# ==============================================================================
resultados = pd.DataFrame(resultados)
resultados = pd.concat([resultados, resultados['params'].apply(pd.Series)], axis=1)
resultados = resultados.drop(columns = 'params')
resultados = resultados.sort_values('mape_valid', ascending=True)
resultados

Y_train=Y_train.reshape([-1,1])
y_hat_train=y_hat_train.reshape([-1,1])
Y_valid=Y_valid.reshape([-1,1])
y_hat_valid=y_hat_valid.reshape([-1,1])
Y_test=Y_test.reshape([-1,1])
y_hat_test=modeloRF_r_mercado.predict(X_test)
y_hat_test=y_hat_test.reshape([-1,1])

#errores=[mean_absolute_error(Y_train,y_hat_train), mean_absolute_percentage_error(Y_train,y_hat_train)]
errores=[mean_absolute_error(Y_valid,y_hat_valid), mean_absolute_percentage_error(Y_valid,y_hat_valid)]
np.save('error_RF_rez_Mercado.npy', errores)
#modelo = np.load('error_RF_rez_Mercado.npy')
#print(modelo[0])
#print(modelo[1])

# save the model to disk
joblib.dump(modeloRF_r_mercado, 'modeloRF_r_mercado.pkl')


"""## XG

### Yamaha
"""

X_train, X_valid, X_test, Y_train, Y_valid, Y_test=preprocesamientoRF_XG(dataset_r_yamaha)

#Grid de hiperparámetros evaluados
# ==============================================================================
param_grid = ParameterGrid( #los mejores parámetros
                {'colsample_bytree':[0.4,0.5,0.6,0.7,0.8],
                 'subsample':[0.4,0.5,0.6,0.7,0.8],
                 'gamma': [5,10],
                 'max_depth'   : range(5,25,5),
                 'n_estimators': range(10,60,10),
                 'reg_lambda':[0,1]
                }
             )

# Loop para ajustar un modelo con cada combinación de hiperparámetros
# ==============================================================================
resultados = {'params': [], 'rmse_train': [], 'rmse_valid': [], 'mae_train':[], 'mae_valid':[], 'mape_train':[], 'mape_valid':[], 'mape_test':[]}

for params in param_grid:
    
    modeloXG_r_yamaha=xgb.XGBRegressor(objectcive ='reg:squarederror',
                                              random_state=1, 
                                              ** params)
    
    modeloXG_r_yamaha.fit(X_train, Y_train)
    y_hat_train=modeloXG_r_yamaha.predict(X_train)
    y_hat_valid=modeloXG_r_yamaha.predict(X_valid)
    mse_train=mean_squared_error(Y_train, y_hat_train)
    mse_valid=mean_squared_error(Y_valid, y_hat_valid)
    mae_train=mean_absolute_error(Y_train, y_hat_train)
    mae_valid=mean_absolute_error(Y_valid, y_hat_valid)
    y_hat_train=np.matrix(y_hat_train)
    #y_hat_train=y_hat_train.T
    y_hat_valid=np.matrix(y_hat_valid)
    #y_hat_valid=y_hat_valid.T
    mape_train=MAPE(Y_train, y_hat_train)
    mape_valid=MAPE(Y_valid, y_hat_valid)
    y_hat_test=modeloXG_r_yamaha.predict(X_test)
    mape_test=MAPE(Y_test,y_hat_test)

    
    resultados['params'].append(params)
    resultados['rmse_train'].append(math.sqrt(mse_train))
    resultados['rmse_valid'].append(math.sqrt(mse_valid))
    resultados['mae_train'].append(mae_train)
    resultados['mae_valid'].append(mae_valid)
    resultados['mape_train'].append(mape_train)
    resultados['mape_valid'].append(mape_valid)
    resultados['mape_test'].append(mape_test)
    


    print(f"Modelo: {params} \u2713")

# Resultados
# ==============================================================================
resultados = pd.DataFrame(resultados)
resultados = pd.concat([resultados, resultados['params'].apply(pd.Series)], axis=1)
resultados = resultados.drop(columns = 'params')
resultados = resultados.sort_values('rmse_valid', ascending=True)
best=resultados.head(1)

"""Mejor modelo"""

#Grid de hiperparámetros evaluados
# ==============================================================================
param_grid = ParameterGrid( #los mejores parámetros
                {'colsample_bytree':best['colsample_bytree'].values,
                 'subsample':best['subsample'].values,
                 'gamma': best['gamma'].values,
                 'max_depth'   : [int(best['max_depth'].values)],
                 'n_estimators': [int(best['n_estimators'].values)],
                 'reg_lambda':best['reg_lambda'].values
                }
             )

# Loop para ajustar un modelo con cada combinación de hiperparámetros
# ==============================================================================
resultados = {'params': [], 'rmse_train': [], 'rmse_valid': [], 'mae_train':[], 'mae_valid':[], 'mape_train':[], 'mape_valid':[], 'mape_test':[]}

for params in param_grid:
    
    modeloXG_r_yamaha=xgb.XGBRegressor(objectcive ='reg:squarederror',
                                              random_state=1, 
                                              ** params)
    
    modeloXG_r_yamaha.fit(X_train, Y_train)
    y_hat_train=modeloXG_r_yamaha.predict(X_train)
    y_hat_valid=modeloXG_r_yamaha.predict(X_valid)
    mse_train=mean_squared_error(Y_train, y_hat_train)
    mse_valid=mean_squared_error(Y_valid, y_hat_valid)
    mae_train=mean_absolute_error(Y_train, y_hat_train)
    mae_valid=mean_absolute_error(Y_valid, y_hat_valid)
    y_hat_train=np.matrix(y_hat_train)
    #y_hat_train=y_hat_train.T
    y_hat_valid=np.matrix(y_hat_valid)
    #y_hat_valid=y_hat_valid.T
    mape_train=MAPE(Y_train, y_hat_train)
    mape_valid=MAPE(Y_valid, y_hat_valid)
    y_hat_test=modeloXG_r_yamaha.predict(X_test)
    mape_test=MAPE(Y_test,y_hat_test)

    
    resultados['params'].append(params)
    resultados['rmse_train'].append(math.sqrt(mse_train))
    resultados['rmse_valid'].append(math.sqrt(mse_valid))
    resultados['mae_train'].append(mae_train)
    resultados['mae_valid'].append(mae_valid)
    resultados['mape_train'].append(mape_train)
    resultados['mape_valid'].append(mape_valid)
    resultados['mape_test'].append(mape_test)
    


    print(f"Modelo: {params} \u2713")

# Resultados
# ==============================================================================
resultados = pd.DataFrame(resultados)
resultados = pd.concat([resultados, resultados['params'].apply(pd.Series)], axis=1)
resultados = resultados.drop(columns = 'params')
resultados = resultados.sort_values('rmse_valid', ascending=True)
resultados.head(4)

Y_train=Y_train.reshape([-1,1])
y_hat_train=y_hat_train.reshape([-1,1])
Y_valid=Y_valid.reshape([-1,1])
y_hat_valid=y_hat_valid.reshape([-1,1])
Y_test=Y_test.reshape([-1,1])
y_hat_test=modeloXG_r_yamaha.predict(X_test)
y_hat_test=y_hat_test.reshape([-1,1])

#errores=[mean_absolute_error(Y_train,y_hat_train), mean_absolute_percentage_error(Y_train,y_hat_train)]
errores=[mean_absolute_error(Y_valid,y_hat_valid), mean_absolute_percentage_error(Y_valid,y_hat_valid)]
np.save('error_XG_rez_Yamaha.npy', errores)
#modelo = np.load('error_XG_rez_Yamaha.npy')
#print(modelo[0])
#print(modelo[1])

# save the model to disk
joblib.dump(modeloXG_r_yamaha, 'modeloXG_r_yamaha.pkl')



"""### Mercado"""

X_train, X_valid, X_test, Y_train, Y_valid, Y_test=preprocesamientoRF_XG(dataset_r_mercado)

#Grid de hiperparámetros evaluados
# ==============================================================================
param_grid = ParameterGrid( #los mejores parámetros
                {'colsample_bytree':[0.4,0.5,0.6,0.7,0.8],
                 'subsample':[0.4,0.5,0.6,0.7,0.8],
                 'gamma': [5,10],
                 'max_depth'   : range(5,25,5),
                 'n_estimators': range(10,60,10),
                 'reg_lambda':[0,1]
                }
             )

# Loop para ajustar un modelo con cada combinación de hiperparámetros
# ==============================================================================
resultados = {'params': [], 'rmse_train': [], 'rmse_valid': [], 'mae_train':[], 'mae_valid':[], 'mape_train':[], 'mape_valid':[], 'mape_test':[]}

for params in param_grid:
    
    modeloXG_r_mercado=xgb.XGBRegressor(objectcive ='reg:squarederror',
                                              random_state=1, 
                                              ** params)
    
    modeloXG_r_mercado.fit(X_train, Y_train)
    y_hat_train=modeloXG_r_mercado.predict(X_train)
    y_hat_valid=modeloXG_r_mercado.predict(X_valid)
    mse_train=mean_squared_error(Y_train, y_hat_train)
    mse_valid=mean_squared_error(Y_valid, y_hat_valid)
    mae_train=mean_absolute_error(Y_train, y_hat_train)
    mae_valid=mean_absolute_error(Y_valid, y_hat_valid)
    y_hat_train=np.matrix(y_hat_train)
    #y_hat_train=y_hat_train.T
    y_hat_valid=np.matrix(y_hat_valid)
    #y_hat_valid=y_hat_valid.T
    mape_train=MAPE(Y_train, y_hat_train)
    mape_valid=MAPE(Y_valid, y_hat_valid)
    y_hat_test=modeloXG_r_mercado.predict(X_test)
    mape_test=MAPE(Y_test,y_hat_test)

    
    resultados['params'].append(params)
    resultados['rmse_train'].append(math.sqrt(mse_train))
    resultados['rmse_valid'].append(math.sqrt(mse_valid))
    resultados['mae_train'].append(mae_train)
    resultados['mae_valid'].append(mae_valid)
    resultados['mape_train'].append(mape_train)
    resultados['mape_valid'].append(mape_valid)
    resultados['mape_test'].append(mape_test)
    


    print(f"Modelo: {params} \u2713")

# Resultados
# ==============================================================================
resultados = pd.DataFrame(resultados)
resultados = pd.concat([resultados, resultados['params'].apply(pd.Series)], axis=1)
resultados = resultados.drop(columns = 'params')
resultados = resultados.sort_values('rmse_valid', ascending=True)
best=resultados.head(1)

"""Mejor modelo"""

#Grid de hiperparámetros evaluados
# ==============================================================================
param_grid = ParameterGrid( #los mejores parámetros
                {'colsample_bytree':best['colsample_bytree'].values,
                 'subsample':best['subsample'].values,
                 'gamma': best['gamma'].values,
                 'max_depth'   : [int(best['max_depth'].values)],
                 'n_estimators': [int(best['n_estimators'].values)],
                 'reg_lambda':best['reg_lambda'].values
                }
             )

# Loop para ajustar un modelo con cada combinación de hiperparámetros
# ==============================================================================
resultados = {'params': [], 'rmse_train': [], 'rmse_valid': [], 'mae_train':[], 'mae_valid':[], 'mape_train':[], 'mape_valid':[], 'mape_test':[]}

for params in param_grid:
    
    modeloXG_r_mercado=xgb.XGBRegressor(objectcive ='reg:squarederror',
                                              random_state=1, 
                                              ** params)
    
    modeloXG_r_mercado.fit(X_train, Y_train)
    y_hat_train=modeloXG_r_mercado.predict(X_train)
    y_hat_valid=modeloXG_r_mercado.predict(X_valid)
    mse_train=mean_squared_error(Y_train, y_hat_train)
    mse_valid=mean_squared_error(Y_valid, y_hat_valid)
    mae_train=mean_absolute_error(Y_train, y_hat_train)
    mae_valid=mean_absolute_error(Y_valid, y_hat_valid)
    y_hat_train=np.matrix(y_hat_train)
    #y_hat_train=y_hat_train.T
    y_hat_valid=np.matrix(y_hat_valid)
    #y_hat_valid=y_hat_valid.T
    mape_train=MAPE(Y_train, y_hat_train)
    mape_valid=MAPE(Y_valid, y_hat_valid)
    y_hat_test=modeloXG_r_mercado.predict(X_test)
    mape_test=MAPE(Y_test,y_hat_test)

    
    resultados['params'].append(params)
    resultados['rmse_train'].append(math.sqrt(mse_train))
    resultados['rmse_valid'].append(math.sqrt(mse_valid))
    resultados['mae_train'].append(mae_train)
    resultados['mae_valid'].append(mae_valid)
    resultados['mape_train'].append(mape_train)
    resultados['mape_valid'].append(mape_valid)
    resultados['mape_test'].append(mape_test)
    


    print(f"Modelo: {params} \u2713")

# Resultados
# ==============================================================================
resultados = pd.DataFrame(resultados)
resultados = pd.concat([resultados, resultados['params'].apply(pd.Series)], axis=1)
resultados = resultados.drop(columns = 'params')
resultados = resultados.sort_values('rmse_valid', ascending=True)
resultados.head(4)

Y_train=Y_train.reshape([-1,1])
y_hat_train=y_hat_train.reshape([-1,1])
Y_valid=Y_valid.reshape([-1,1])
y_hat_valid=y_hat_valid.reshape([-1,1])
Y_test=Y_test.reshape([-1,1])
y_hat_test=modeloXG_r_mercado.predict(X_test)
y_hat_test=y_hat_test.reshape([-1,1])

#errores=[mean_absolute_error(Y_train,y_hat_train), mean_absolute_percentage_error(Y_train,y_hat_train)]
errores=[mean_absolute_error(Y_valid,y_hat_valid), mean_absolute_percentage_error(Y_valid,y_hat_valid)]
np.save('error_XG_rez_Mercado.npy', errores)
#modelo = np.load('error_XG_rez_Mercado.npy')
#print(modelo[0])
#print(modelo[1])

# save the model to disk
joblib.dump(modeloXG_r_mercado, 'modeloXG_r_mercado.pkl')


