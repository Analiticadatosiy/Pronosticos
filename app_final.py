import streamlit as st
import joblib
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from statsmodels.tsa.holtwinters import ExponentialSmoothing as HWES
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
import base64
import plotly.express as px
from dateutil.relativedelta import relativedelta
import datetime
import time

st.set_page_config(
  layout="wide",  # Can be "centered" or "wide". In the future also "dashboard", etc.
  initial_sidebar_state="auto",  # Can be "auto", "expanded", "collapsed"
  page_title='Pronósticos',  # String or None. Strings get appended with "• Streamlit". 
  page_icon=None,  # String, anything supported by st.image, or None.
)

#Importar el dataframe (reemplazar por la base de datos)
df=pd.read_excel('BD_actualizado.xlsx')
df=df.dropna()
df=df.reset_index(drop=True)
df.FECHA=df.FECHA.apply(lambda x: x.strftime('%d/%m/%Y'))

def get_table_download_link(df):
      """Generates a link allowing the data in a given panda dataframe to be downloaded
      in:  dataframe
      out: href string
      """
      csv = df.to_csv(index=False)
      b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
      href = f'<a href="data:file/csv;base64,{b64}">Download csv file</a>'

#Limpieza del dataframe
def limpieza_actual(df,variable): #(variable_df)
  df=df.dropna()
  df=df.reset_index(drop=True)
  dataset=df.copy()
  dataset=dataset.drop(['PIB GROWTH','ISM','ICC','MES','TEMPORADA','VACACIONES','CLIMA','INGRESOS','PRECIOS','IMPUESTOS','WHOLESALE', variable],axis=1)
  dataset2=dataset.copy() # tiene como index las fechas
  dataset2.set_index('FECHA',inplace=True)
  dataset['DIAS HABILES']=dataset['DIAS HABILES']/(dataset['DIAS HABILES']+dataset['FESTIVOS'])
  dataset=dataset.drop(['FESTIVOS'],axis=1)
  dataset.rename(columns={'DIAS HABILES': 'RATIO_DH_F'},inplace=True) #esta variable tiene estacionalidad
  dataset1=dataset.copy() # tiene como index número enteros
  dataset1=dataset1.drop(['FECHA'],axis=1)
  numpy1=dataset1.values
  return dataset1, numpy1, dataset2

def preprocesamientoRN(df): 
  X=df[:,0:df.shape[1]-1]
  Y=df[:,df.shape[1]-1:]
  min_max_scaler=preprocessing.MinMaxScaler([-1,1])
  X_scale=min_max_scaler.fit_transform(X)
  Y_scale=min_max_scaler.fit_transform(Y)
  return min_max_scaler, X_scale, Y_scale

def actual_individual(variable):
  if variable == 'Yamaha':
    variable_modelo='yamaha'
    variable_df='RUNT MERCADO'
    visualizar='RUNT YAMAHA'
  elif variable == 'Mercado':
    variable_modelo='mercado'
    variable_df='RUNT YAMAHA'
    visualizar='RUNT MERCADO'
  
  st.subheader('Estimar demanda un sólo mes - ' + variable)
  st.write('Por favor, ingrese para cada variable el valor que supone tendría en el tiempo futuro en el que desea estimar la demanda.\n Tenga en cuenta que el dato de festivos y días hábiles corresponde valor real del mes en cuestión que quiere proyectar.' )
  st.write('Puede guiarse de los últimos 6 datos de la tabla para que le sirvan de ejemplo y guía de cómo debe ingresar los datos supuestos.')
  dataset1,numpy1,dataset2=limpieza_actual(df,variable_df)
  dataset2=dataset2.rename(columns={"DESEMPLEO": "Desempleo", "INFLATION": "Inflación", "IEE":'IEC','OIL PRICE':'Precio petróleo', 'DIAS HABILES':'Días hábiles', 'FESTIVOS':'Días festivos'})
  st.write(dataset2.tail(6))  

  col1, col2=st.beta_columns(2)
  ini=2020
  year = col1.selectbox('Fecha a estimar (Año)', range(ini, ini+11))
  month = col2.selectbox('Fecha a estimar (Mes)', range(1, 13))
  DESEMPLEO = st.number_input("Desempleo", format="%.3f")
  INFLATION = st.number_input("Inflación", format="%.3f")
  TRM = st.number_input("Tasa de cambio representativa del mercado (TRM)", format="%.2f")
  SMMLV_TTE = st.number_input("Salario mínimo (SMMLV&TTE)", format="%.0f")
  IEE = st.number_input("Índice de Expectativas de los Consumidores (IEC)", format="%.2f")
  ICE = st.number_input("Índice de Condiciones Económicas (ICE)", format="%.2f")
  OIL_PRICE = st.number_input("Precio del crudo (en dólares)", format="%.3f")
  DIAS_HABILES = st.number_input("Dias hábiles", format="%.0f")
  FESTIVOS = st.number_input("Festivos", format="%.0f")

  if (st.button('Pronosticar')):
    modeloRN=keras.models.load_model('modelRN_'+ variable_modelo +'.h5') #(variable_modelo)
    modeloRF=joblib.load('modeloRF_' + variable_modelo + '.pkl')
    modeloXG=joblib.load('modeloXG_' + variable_modelo + '.pkl')

    X=np.array([[DESEMPLEO, INFLATION, TRM, SMMLV_TTE, IEE, ICE, OIL_PRICE, (DIAS_HABILES)/(DIAS_HABILES+FESTIVOS)]])
    X_RN=np.concatenate([numpy1,np.reshape(np.append(X, [6000]), (1,-1))])

    #Redes Neuronales
    scaler, X_scale, Y_scale=preprocesamientoRN(X_RN)
    y_hat_scale=modeloRN.predict(np.reshape(X_scale[-1],(1,-1)))
    y_hat_RN=scaler.inverse_transform(y_hat_scale).ravel()

    #Random Forest
    y_hat_RF=modeloRF.predict(X)

    #XGBoost
    y_hat_XG=modeloXG.predict(X)

    #Promedio
    y_hat_prom=(y_hat_RN+y_hat_RF+y_hat_XG)/3

    #index=[MES.strftime('%d/%m/%Y')]
    index=['1/'+str(month)+'/'+str(year)]
    #index=['t futuro']
    resultados=pd.DataFrame({'Redes Neuronales': np.around(y_hat_RN), 'Random Forest':np.around(y_hat_RF), 'XGBoost':np.around(y_hat_XG), 'Promedio':np.around(y_hat_prom)}, index=index)
    st.write(resultados)

    errores_RN = np.load('error_RNN_actual_'+variable+'.npy')
    errores_RF = np.load('error_RF_actual_'+variable+'.npy')
    errores_XG = np.load('error_XG_actual_'+variable+'.npy')

    errores=pd.DataFrame()
    errores['Errores']=['MAE','MAPE']
    errores['Redes Neuronales']=[int(errores_RN[0]), str(round(errores_RN[1]*100,2))+'%']
    errores['Random Forest']=[int(errores_RF[0]), str(round(errores_RF[1]*100,2))+'%']
    errores['XGBoost']=[int(errores_XG[0]), str(round(errores_XG[1]*100,2))+'%']
    errores.set_index('Errores',inplace=True)

    st.write(errores)


def actual_lote(variable):
  if variable == 'Yamaha':
    variable_modelo='yamaha'
    variable_df='RUNT MERCADO'
    visualizar='RUNT YAMAHA'
  elif variable == 'Mercado':
    variable_modelo='mercado'
    variable_df='RUNT YAMAHA'
    visualizar='RUNT MERCADO'
  
  st.subheader('Estimar demanda para varios meses - ' + variable)
  st.write('Por favor suba un archivo con los valores que supone que tendrán las variables en el horizonte futuro, tenga en cuenta que debe tener las mismas variables y en el mismo orden de la tabla.')
  st.write('Para facilitar el cargue de los datos, utilice y descargue la **plantilla** que aparece a continuación y una vez diligenciada vuelva a cargarla.') 
    
  dataset1,numpy1,dataset2=limpieza_actual(df,variable_df)
  dataset2=dataset2.rename(columns={"DESEMPLEO": "Desempleo", "INFLATION": "Inflación", "IEE":'IEC','OIL PRICE':'Precio petróleo', 'DIAS HABILES':'Días hábiles', 'FESTIVOS':'Días festivos'})
  st.write(dataset2.tail(6))

  # Cargar plantilla para que usuario descargue
  plantilla_lote_actual=pd.read_excel('plantilla_lote.xlsx')
  csv = plantilla_lote_actual.to_csv(index=False)
  b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
  href = f'<a href="data:file/csv;base64,{b64}">Descargue plantilla</a> una vez diligenciada vuelva a subir a la aplicación (nombre como quiera y agregue la extensión del archivo .csv)'
  st.markdown(href, unsafe_allow_html=True)


  st.markdown('**Subir plantilla**')
  st.write(":warning: El archivo que suba debe tener extensión 'xlsx'")
  data_file=st.file_uploader('Archivo',type=['xlsx'])
  if data_file is not None:
    df_p=pd.read_excel(data_file)
    index_pron=df_p['Fecha']
    df_p=df_p.drop(['Fecha'],axis=1)
    columns=df_p.shape[1]
    df_p.iloc[:,columns-2]=df_p.iloc[:,columns-2]/(df_p.iloc[:,columns-2]+df_p.iloc[:,columns-1]) #Crear RATIO_DH_F

    for i in range(0, df_p.shape[0],1):
      df_p.iloc[i,columns-1]=6000
      
    modeloRN=keras.models.load_model('modelRN_'+ variable_modelo +'.h5') #(variable_modelo)
    modeloRF=joblib.load('modeloRF_' + variable_modelo + '.pkl')
    modeloXG=joblib.load('modeloXG_' + variable_modelo + '.pkl')

    
    X=df_p.values
    X_RN=np.concatenate([numpy1,X]) #Para la RN
    X=X[:,0:df_p.shape[1]-1] # Para RF y XG
    
    #Redes Neuronales
    scaler, X_scale, Y_scale=preprocesamientoRN(X_RN)
    y_hat_scale=modeloRN.predict(X_scale[(len(X_RN)-len(X)):,:])
    y_hat_RN=scaler.inverse_transform(y_hat_scale).ravel()

    #Random Forest
    y_hat_RF=modeloRF.predict(X)

    #XGBoost
    y_hat_XG=modeloXG.predict(X)

    #Promedio
    y_hat_prom=(y_hat_RN+y_hat_RF+y_hat_XG)/3

    st.markdown('**Pronóstico**') 
    resultados=pd.DataFrame({'Fecha':index_pron,'Redes Neuronales': np.around(y_hat_RN), 'Random Forest':np.around(y_hat_RF), 'XGBoost':np.around(y_hat_XG), 'Promedio':np.around(y_hat_prom)})
    resultados['Fecha']=resultados.Fecha.apply(lambda x: x.strftime('%d/%m/%Y'))
    resultados.set_index('Fecha',inplace=True)
    #Descargar los resultados
    csv = resultados.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}">aqui</a>'
    st.markdown('Si desea descargar el pronóstico de click '+href, unsafe_allow_html=True)
    #st.write('Si desea descargar la tabla con el pronóstico, de click en el link de la parte inferior')

   
    st.write(resultados)

    errores_RN = np.load('error_RNN_actual_'+variable+'.npy')
    errores_RF = np.load('error_RF_actual_'+variable+'.npy')
    errores_XG = np.load('error_XG_actual_'+variable+'.npy')

    errores=pd.DataFrame()
    errores['Errores']=['MAE','MAPE']
    errores['Redes Neuronales']=[int(errores_RN[0]), str(round(errores_RN[1]*100,2))+'%']
    errores['Random Forest']=[int(errores_RF[0]), str(round(errores_RF[1]*100,2))+'%']
    errores['XGBoost']=[int(errores_XG[0]), str(round(errores_XG[1]*100,2))+'%']
    errores.set_index('Errores',inplace=True)
    st.markdown('**Errores**')

    st.write(errores)
    
    #Descargar los resultados
    #csv = resultados.to_csv(index=False)
    #b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    #href = f'<a href="data:file/csv;base64,{b64}">Descargue pronóstico</a> (defina nombre y agregue la extensión del archivo .csv)'
    #st.markdown(href, unsafe_allow_html=True)

    #Gráfica 1  
    graficar=dataset2[-60:]
    total=pd.concat([graficar[visualizar], resultados])
    total.rename(columns={0: visualizar},inplace=True) #esta variable tiene estacionalidad
    total=total.reset_index()
    df_melt = total.melt(id_vars='index', value_vars=[visualizar,'Redes Neuronales','Random Forest','XGBoost','Promedio'])
    px.defaults.width = 1100
    px.defaults.height = 500
    fig = px.line(df_melt, x='index',y='value', color='variable', labels={"index": "Fecha",  "value": "Runt"})
    st.plotly_chart(fig)



def limpieza_rezago_yamaha(df):
  df=df.dropna()
  df=df.reset_index(drop=True)
  dataset=df.copy()
  dataset=dataset.drop(['PIB GROWTH','ISM','ICC','IEE','ICE','MES','TEMPORADA','VACACIONES','CLIMA','INGRESOS','PRECIOS','IMPUESTOS','WHOLESALE'],axis=1)
  dataset2=dataset.copy() # tiene como index las fechas
  dataset2.set_index('FECHA',inplace=True)
  dataset['DIAS HABILES']=dataset['DIAS HABILES']/(dataset['DIAS HABILES']+dataset['FESTIVOS'])
  dataset=dataset.drop(['FESTIVOS'],axis=1)
  dataset.rename(columns={'DIAS HABILES': 'RATIO_DH_F'},inplace=True) #esta variable tiene estacionalidad
  dataset1=dataset.copy() # tiene como index número enteros
  dataset1=pd.concat([dataset1, dataset1['RUNT YAMAHA']], axis=1)
  dataset1=dataset1.drop(['FECHA'],axis=1)
  numpy1=dataset1.values
  return dataset1, numpy1, dataset2

def rezago_yamaha():
  st.subheader('Estimar demanda con datos reales último año - Yamaha')
  st.write('Por favor ingrese el dato de las variables un año atras del periodo que desea estimar, es decir, si desea estimar Junio de 2021, ingrese los datos de Junio de 2020 (para el caso de los días hábiles y festivos, sí se deben ingresar los valores reales para el mes que se desea pronosticar)')
  st.write('Tome como guía de los valores y formatos la tabla que se muestra a continuación.')
    
  dataset1,numpy1,dataset2=limpieza_rezago_yamaha(df)
  dataset2=dataset2.rename(columns={"DESEMPLEO": "Desempleo", "INFLATION": "Inflación", "IEE":'IEC','OIL PRICE':'Precio petróleo', 'DIAS HABILES':'Días hábiles', 'FESTIVOS':'Días festivos'})
  st.write(dataset2.tail(6))
  

  col1, col2=st.beta_columns(2)
  ini=2020
  year = col1.selectbox('Fecha a estimar (Año)', range(ini, ini+11))
  month = col2.selectbox('Fecha a estimar (Mes)', range(1, 13))
  DESEMPLEO = st.number_input("Desempleo", format="%.3f")
  INFLATION = st.number_input("Inflación", format="%.3f")
  TRM = st.number_input("Tasa de cambio representativa del mercado [TRM]", format="%.3f")
  SMMLV_TTE = st.number_input("Salario mínimo y aux transporte [SMMLV&TTE] (t)", format="%.3f")
  OIL_PRICE = st.number_input("Precio del crudo (en dólares)", format="%.3f")
  DIAS_HABILES = st.number_input("Dias hábiles (t+12)", format="%.0f")
  FESTIVOS = st.number_input("Festivos (t+12)", format="%.0f")
  RUNT_MERCADO = st.number_input("Runt Mercado", format="%.0f")
  RUNT_YAMAHA = st.number_input("Runt Yamaha", format="%.0f")

  if (st.button('Pronosticar')):
    modelRN_r_yamaha = keras.models.load_model('modelRN_r_yamaha.h5')
    modeloRF_r_yamaha = joblib.load('modeloRF_r_yamaha.pkl')
    modeloXG_r_yamaha = joblib.load('modeloXG_r_yamaha.pkl')

    X=np.array([[DESEMPLEO, INFLATION, TRM, SMMLV_TTE, OIL_PRICE, (DIAS_HABILES)/(DIAS_HABILES+FESTIVOS), RUNT_MERCADO, RUNT_YAMAHA]])
    X_RN=np.concatenate([numpy1,np.reshape(np.append(X, [7000]), (1,-1))])

    #Redes Neuronales
    scaler, X_scale, Y_scale=preprocesamientoRN(X_RN)
    y_hat_scale=modelRN_r_yamaha.predict(np.reshape(X_scale[-1],(1,-1)))
    y_hat_RN=scaler.inverse_transform(y_hat_scale).ravel()

    #Random Forest
    y_hat_RF=modeloRF_r_yamaha.predict(X)

    #XGBoost
    y_hat_XG=modeloXG_r_yamaha.predict(X)

    #Promedio
    y_hat_prom=(y_hat_RN+y_hat_RF+y_hat_XG)/3
    
    index=['1/'+str(month)+'/'+str(year)]
    resultados=pd.DataFrame({'Redes Neuronales': np.around(y_hat_RN), 'Random Forest':np.around(y_hat_RF), 'XGBoost':np.around(y_hat_XG), 'Promedio':np.around(y_hat_prom)}, index=index)

    st.write(resultados)

    errores_RN = np.load('error_RNN_rez_Yamaha.npy')
    errores_RF = np.load('error_RF_rez_Yamaha.npy')
    errores_XG = np.load('error_XG_rez_Yamaha.npy')

    errores=pd.DataFrame()
    errores['Errores']=['MAE','MAPE']
    errores['Redes Neuronales']=[int(errores_RN[0]), str(round(errores_RN[1]*100,2))+'%']
    errores['Random Forest']=[int(errores_RF[0]), str(round(errores_RF[1]*100,2))+'%']
    errores['XGBoost']=[int(errores_XG[0]), str(round(errores_XG[1]*100,2))+'%']
    errores.set_index('Errores',inplace=True)

    st.write(errores)

    #Para graficar los resultados
    y_hat=resultados.values
    y_hat=np.delete(y_hat,3)
    index=['Redes Neuronales', 'Random Forest', 'XGBoost']
    resultados2=pd.DataFrame({'Resultados': y_hat},index=index)
    
    promedio=resultados['Promedio'].values
    promedio=np.array([promedio, promedio, promedio])
    promedio=promedio.ravel()


def rezago_yamaha_lote():
  st.subheader('Estimar la demanda con datos reales último año - Yamaha')
  st.write('Por favor, suba un archivo con los valores de los últimos 12 meses disponibles, con ellos se pronosticarán los siguientes 12 meses, tenga en cuenta que el archivo que adjunte debe tener las mismas variables y en el mismo orden de la tabla.')
  st.write('Para facilitar el cargue de los datos, utilice la **plantilla** que aparece a continuación y una vez diligenciada vuelva a cargarla. ')
  dataset1,numpy1,dataset2=limpieza_rezago_yamaha(df)
  dataset2=dataset2.rename(columns={"DESEMPLEO": "Desempleo", "INFLATION": "Inflación", "IEE":'IEC','OIL PRICE':'Precio petróleo', 'DIAS HABILES':'Días hábiles', 'FESTIVOS':'Días festivos'})
  st.write(dataset2.tail(6))

  # Cargar plantilla para que usuario descargue
  plantilla_lote_r_yamaha=pd.read_excel('plantilla_lote_r_yamaha.xlsx')
  csv = plantilla_lote_r_yamaha.to_csv(index=False)
  b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
  href = f'<a href="data:file/csv;base64,{b64}">Descargue plantilla</a> una vez diligenciada vuelva a subir a la aplicación (nombre como quiera y agregue la extensión del archivo .csv)'
  st.markdown(href, unsafe_allow_html=True)

  st.markdown('**Subir plantilla**')
  st.write(":warning: El archivo que suba debe tener extensión 'xlsx'")
  data_file=st.file_uploader('Archivo',type=['xlsx'])
  if data_file is not None:
    df_p=pd.read_excel(data_file)  
    index_pron=df_p['Fecha']
    df_p=df_p.drop(['Fecha'],axis=1)
    columns1=df_p.shape[1]
    df_p.iloc[:,5]=df_p.iloc[:,5]/(df_p.iloc[:,5]+df_p.iloc[:,6]) #Crear RATIO_DH_F
    df_p=df_p.drop(['Festivos'],axis=1)
        
    vector=pd.DataFrame((np.ones((df_p.shape[0],1), dtype=int))*(7000))
    df_p = pd.concat([df_p,vector], axis=1)
      
    modelRN_r_yamaha = keras.models.load_model('modelRN_r_yamaha.h5')
    modeloRF_r_yamaha = joblib.load('modeloRF_r_yamaha.pkl')
    modeloXG_r_yamaha = joblib.load('modeloXG_r_yamaha.pkl')

    X=df_p.values
    X_RN=np.concatenate([numpy1,X]) #Para la RN
    X=X[:,0:df_p.shape[1]-1] # Para RF y XG

    #Redes Neuronales
    scaler, X_scale, Y_scale=preprocesamientoRN(X_RN)
    y_hat_scale=modelRN_r_yamaha.predict(X_scale[(len(X_RN)-len(X)):,:])
    y_hat_RN=scaler.inverse_transform(y_hat_scale).ravel()
      
    #Random Forest
    y_hat_RF=modeloRF_r_yamaha.predict(X)

    #XGBoost
    y_hat_XG=modeloXG_r_yamaha.predict(X)

    #Promedio
    y_hat_prom=(y_hat_RN+y_hat_RF+y_hat_XG)/3

    st.write('**Pronóstico**')
    resultados=pd.DataFrame({'Fecha':index_pron,'Redes Neuronales': np.around(y_hat_RN), 'Random Forest':np.around(y_hat_RF), 'XGBoost':np.around(y_hat_XG), 'Promedio':np.around(y_hat_prom)})
    resultados['Fecha']=resultados['Fecha'].apply(lambda x: (x+relativedelta(months=+12)).strftime('%d/%m/%Y'))
    resultados.set_index('Fecha',inplace=True)
    csv = resultados.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}">aquí</a>'
    st.markdown('Si desea descargar el pronóstico de click '+href, unsafe_allow_html=True)
    #st.write('Si desea descargar la tabla con el pronóstico, de click en el link de la parte inferior')

    st.write(resultados)
    
    errores_RN = np.load('error_RNN_rez_Yamaha.npy')
    errores_RF = np.load('error_RF_rez_Yamaha.npy')
    errores_XG = np.load('error_XG_rez_Yamaha.npy')

    errores=pd.DataFrame()
    errores['Errores']=['MAE','MAPE']
    errores['Redes Neuronales']=[int(errores_RN[0]), str(round(errores_RN[1]*100,2))+'%']
    errores['Random Forest']=[int(errores_RF[0]), str(round(errores_RF[1]*100,2))+'%']
    errores['XGBoost']=[int(errores_XG[0]), str(round(errores_XG[1]*100,2))+'%']
    errores.set_index('Errores',inplace=True)
    st.markdown('**Errores**')
    st.write(errores)

    #Descargar los resultados

    #v = resultados.to_csv(index=False)
    #b64 = base64.b64encode(v.encode()).decode()  # some strings <-> bytes conversions necessary here
    #href = f'<a href="data:application/octet-stream;base64,{b64}" download="resultados.csv">Descarga los resultados en formato CSV</a>'
    #href = f'<a href="data:file/csv;base64,{b64}">Descargue pronóstico</a> (defina nombre y agregue la extensión del archivo .csv)'
    #st.markdown(href, unsafe_allow_html=True)

    #Gráfica 1  
    graficar=dataset2[-60:]
    total=pd.concat([graficar['RUNT YAMAHA'], resultados])
    total.rename(columns={0: 'RUNT YAMAHA'},inplace=True) #esta variable tiene estacionalidad
    total=total.reset_index()
    df_melt = total.melt(id_vars='index', value_vars=['RUNT YAMAHA','Redes Neuronales','Random Forest','XGBoost','Promedio'])
    px.defaults.width = 1100
    px.defaults.height = 500
    fig = px.line(df_melt, x='index',y='value', color='variable', labels={"index": "Fecha",  "value": "Runt"})
    st.plotly_chart(fig)
    

def limpieza_rezago_mercado(df):
  df=df.dropna()
  df=df.reset_index(drop=True)
  dataset=df.copy()
  dataset=dataset.drop(['PIB GROWTH','ISM','ICC','IEE','ICE','MES','TEMPORADA','VACACIONES','CLIMA','INGRESOS','PRECIOS','IMPUESTOS','WHOLESALE','RUNT YAMAHA'],axis=1)
  dataset2=dataset.copy() # tiene como index las fechas
  dataset2.set_index('FECHA',inplace=True)
  dataset['DIAS HABILES']=dataset['DIAS HABILES']/(dataset['DIAS HABILES']+dataset['FESTIVOS'])
  dataset=dataset.drop(['FESTIVOS'],axis=1)
  dataset.rename(columns={'DIAS HABILES': 'RATIO_DH_F'},inplace=True) #esta variable tiene estacionalidad
  dataset1=dataset.copy() # tiene como index número enteros
  dataset1=pd.concat([dataset1, dataset1['RUNT MERCADO']], axis=1)
  dataset1=dataset1.drop(['FECHA'],axis=1)
  numpy1=dataset1.values
  return dataset1, numpy1, dataset2

def rezago_mercado():
  st.subheader('Estimar la demanda con datos reales último año - Mercado')
  st.write('Por favor ingrese el dato de las variables un año atras del periodo que desea estimar, es decir, si desea estimar Junio de 2021, ingrese los datos de Junio de 2020 (para el caso de los días hábiles y festivos, sí se deben ingresar los valores reales para el mes que se desea pronosticar)')
  st.write('Tome como guía de los valores y formatos la tabla que se muestra a continuación.')
  dataset1,numpy1,dataset2=limpieza_rezago_mercado(df)
  dataset2=dataset2.rename(columns={"DESEMPLEO": "Desempleo", "INFLATION": "Inflación", "IEE":'IEC','OIL PRICE':'Precio petróleo', 'DIAS HABILES':'Días hábiles', 'FESTIVOS':'Días festivos'})
  st.write(dataset2.tail(6))

  col1, col2=st.beta_columns(2)
  ini=2020
  year = col1.selectbox('Fecha a estimar (Año)', range(ini, ini+11))
  month = col2.selectbox('Fecha a estimar (Mes)', range(1, 13))

  DESEMPLEO = st.number_input("Desempleo (t)", format="%.3f")
  INFLATION = st.number_input("Inflación (t)", format="%.3f")
  TRM = st.number_input("Tasa de cambio representativa del mercado [TRM] (t)", format="%.3f")
  SMMLV_TTE = st.number_input("Salario mínimo [SMMLV&TTE] (t)", format="%.3f")
  OIL_PRICE = st.number_input("Precio del crudo (t, en dólares)", format="%.3f")
  DIAS_HABILES = st.number_input("Dias hábiles (t+12)", format="%.0f")
  FESTIVOS = st.number_input("Festivos (t+12)", format="%.0f")
  RUNT_MERCADO = st.number_input("Runt Mercado (t)", format="%.0f")

  if (st.button('Pronosticar')):
    modelRN_r_yamaha = keras.models.load_model('modelRN_r_mercado.h5')
    modeloRF_r_yamaha = joblib.load('modeloRF_r_mercado.pkl')
    modeloXG_r_yamaha = joblib.load('modeloXG_r_mercado.pkl')

    X=np.array([[DESEMPLEO, INFLATION, TRM, SMMLV_TTE, OIL_PRICE, (DIAS_HABILES)/(DIAS_HABILES+FESTIVOS), RUNT_MERCADO]])
    X_RN=np.concatenate([numpy1,np.reshape(np.append(X, [7000]), (1,-1))])

    #Redes Neuronales
    scaler, X_scale, Y_scale=preprocesamientoRN(X_RN)
    y_hat_scale=modelRN_r_yamaha.predict(np.reshape(X_scale[-1],(1,-1)))
    y_hat_RN=scaler.inverse_transform(y_hat_scale).ravel()

    #Random Forest
    y_hat_RF=modeloRF_r_yamaha.predict(X)

    #XGBoost
    y_hat_XG=modeloXG_r_yamaha.predict(X)

    #Promedio
    y_hat_prom=(y_hat_RN+y_hat_RF+y_hat_XG)/3

    index=['1/'+str(month)+'/'+str(year)]
    resultados=pd.DataFrame({'Redes Neuronales': np.around(y_hat_RN), 'Random Forest':np.around(y_hat_RF), 'XGBoost':np.around(y_hat_XG), 'Promedio':np.around(y_hat_prom)}, index=index)

    st.write(resultados)

    errores_RN = np.load('error_RNN_rez_Mercado.npy')
    errores_RF = np.load('error_RF_rez_Mercado.npy')
    errores_XG = np.load('error_XG_rez_Mercado.npy')

    errores=pd.DataFrame()
    errores['Errores']=['MAE','MAPE']
    errores['Redes Neuronales']=[int(errores_RN[0]), str(round(errores_RN[1]*100,2))+'%']
    errores['Random Forest']=[int(errores_RF[0]), str(round(errores_RF[1]*100,2))+'%']
    errores['XGBoost']=[int(errores_XG[0]), str(round(errores_XG[1]*100,2))+'%']
    errores.set_index('Errores',inplace=True)

    st.write(errores)

    #Para graficar los resultados
    y_hat=resultados.values
    y_hat=np.delete(y_hat,3)
    index=['Redes Neuronales', 'Random Forest', 'XGBoost']
    resultados2=pd.DataFrame({'Resultados': y_hat},index=index)
    
    promedio=resultados['Promedio'].values
    promedio=np.array([promedio, promedio, promedio])
    promedio=promedio.ravel()

def rezago_mercado_lote():
  st.subheader('Estimar la demanda con doce meses de rezago - Mercado')
  st.write('Por favor, suba un archivo con los valores de los últimos 12 meses disponibles, con ellos se pronosticarán los siguientes 12 meses, tenga en cuenta que el archivo que adjunte debe tener las mismas variables y en el mismo orden de la tabla.')
  st.write('Para facilitar el cargue de los datos, utilice la **plantilla** que aparece a continuación y una vez diligenciada vuelva a cargarla. ')
  
  dataset1,numpy1,dataset2=limpieza_rezago_mercado(df)
  dataset2=dataset2.rename(columns={"DESEMPLEO": "Desempleo", "INFLATION": "Inflación", "IEE":'IEC','OIL PRICE':'Precio petróleo', 'DIAS HABILES':'Días hábiles', 'FESTIVOS':'Días festivos'})
  st.write(dataset2.tail(6))

  # Cargar plantilla para que usuario descargue
  plantilla_lote_r_mercado=pd.read_excel('plantilla_lote_r_mercado.xlsx')
  csv = plantilla_lote_r_mercado.to_csv(index=False)
  b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
  href = f'<a href="data:file/csv;base64,{b64}">Descargue plantilla</a> una vez diligenciada vuelva a subir a la aplicación (nombre como quiera y agregue la extensión del archivo .csv)'
  st.markdown(href, unsafe_allow_html=True)

  st.markdown('**Subir plantilla**')
  st.write(":warning: El archivo que suba debe tener extensión 'xlsx'")
  data_file=st.file_uploader('Archivo',type=['xlsx'])
  if data_file is not None:
    df_p=pd.read_excel(data_file)  
    index_pron=df_p['Fecha']
    df_p=df_p.drop(['Fecha'],axis=1)
    columns1=df_p.shape[1]
    df_p.iloc[:,5]=df_p.iloc[:,5]/(df_p.iloc[:,5]+df_p.iloc[:,6]) #Crear RATIO_DH_F
    df_p=df_p.drop(['Festivos'],axis=1)

    vector=pd.DataFrame((np.ones((df_p.shape[0],1), dtype=int))*(7000))
    df_p = pd.concat([df_p,vector], axis=1)
      
    modelRN_r_yamaha = keras.models.load_model('modelRN_r_mercado.h5')
    modeloRF_r_yamaha = joblib.load('modeloRF_r_mercado.pkl')
    modeloXG_r_yamaha = joblib.load('modeloXG_r_mercado.pkl')

    X=df_p.values
    X_RN=np.concatenate([numpy1,X]) #Para la RN
    X=X[:,0:df_p.shape[1]-1] # Para RF y XG

    #Redes Neuronales
    scaler, X_scale, Y_scale=preprocesamientoRN(X_RN)
    y_hat_scale=modelRN_r_yamaha.predict(X_scale[(len(X_RN)-len(X)):,:])
    y_hat_RN=scaler.inverse_transform(y_hat_scale).ravel()
      
    #Random Forest
    y_hat_RF=modeloRF_r_yamaha.predict(X)

    #XGBoost
    y_hat_XG=modeloXG_r_yamaha.predict(X)

    #Promedio
    y_hat_prom=(y_hat_RN+y_hat_RF+y_hat_XG)/3

    st.write('**Pronóstico**')
    resultados=pd.DataFrame({'Fecha':index_pron,'Redes Neuronales': np.around(y_hat_RN), 'Random Forest':np.around(y_hat_RF), 'XGBoost':np.around(y_hat_XG), 'Promedio':np.around(y_hat_prom)})
    resultados['Fecha']=resultados['Fecha'].apply(lambda x: (x+relativedelta(months=+12)).strftime('%d/%m/%Y'))
    resultados.set_index('Fecha',inplace=True)
    csv = resultados.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}">aquí</a>'
    st.markdown('Si desea descargar el pronóstico de click '+href, unsafe_allow_html=True)
    #st.write('Si desea descargar la tabla con el pronóstico, de click en el link de la parte inferior')

    st.write(resultados)

    errores_RN = np.load('error_RNN_rez_Mercado.npy')
    errores_RF = np.load('error_RF_rez_Mercado.npy')
    errores_XG = np.load('error_XG_rez_Mercado.npy')

    errores=pd.DataFrame()
    errores['Errores']=['MAE','MAPE']
    errores['Redes Neuronales']=[int(errores_RN[0]), str(round(errores_RN[1]*100,2))+'%']
    errores['Random Forest']=[int(errores_RF[0]), str(round(errores_RF[1]*100,2))+'%']
    errores['XGBoost']=[int(errores_XG[0]), str(round(errores_XG[1]*100,2))+'%']
    errores.set_index('Errores',inplace=True)
    st.markdown('**Errores**')
    st.write(errores)

    #Descargar los resultados

    #v = resultados.to_csv(index=False)
    #b64 = base64.b64encode(v.encode()).decode()  # some strings <-> bytes conversions necessary here
    #href = f'<a href="data:application/octet-stream;base64,{b64}" download="resultados.csv">Descarga los resultados en formato CSV</a>'
    #st.markdown(href, unsafe_allow_html=True)


    #Gráfica 1  
    graficar=dataset2[-60:]
    total=pd.concat([graficar['RUNT MERCADO'], resultados])
    total.rename(columns={0: 'RUNT MERCADO'},inplace=True) #esta variable tiene estacionalidad
    total=total.reset_index()
    df_melt = total.melt(id_vars='index', value_vars=['RUNT MERCADO','Redes Neuronales','Random Forest','XGBoost','Promedio'])
    px.defaults.width = 1100
    px.defaults.height = 500
    fig = px.line(df_melt, x='index',y='value', color='variable', labels={"index": "Fecha",  "value": "Runt"})
    st.plotly_chart(fig)

def model_eval(y, predictions):

    # Import library for metrics
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    mae = mean_absolute_error(y, predictions)     # Mean absolute error (MAE) 
    mse = mean_squared_error(y, predictions)  # Mean squared error (MSE)    
    SMAPE = np.mean(np.abs((y - predictions) / ((y + predictions)/2))) * 100 # SMAPE is an alternative for MAPE when there are zeros in the testing data. It
    # scales the absolute percentage by the sum of forecast and observed values
    rmse = np.sqrt(mean_squared_error(y, predictions)) # Root Mean Squared Error
    MAPE = np.mean(np.abs((y - predictions) / y)) * 100  
    mfe = np.mean(y - predictions) # mean_forecast_error
    NMSE = mse / (np.sum((y - np.mean(y)) ** 2)/(len(y)-1)) # normalised_mean_squared_error
    # Print metrics
    #st.write('Mean Squared Error:', round(mse, 3))
    #st.write('Root Mean Squared Error:', round(rmse, 3))
    #st.write('Scaled Mean absolute percentage error:', round(SMAPE, 3))
    #st.write('Mean forecast error:', round(mfe, 3))
    #st.write('Normalised mean squared error:', round(NMSE, 3))
    st.write('Mean Absolute Error:', round(mae, 1))
    st.write('Mean absolute percentage error:', round(MAPE, 1))

def exp_smoothing_configs(seasonal=[None]):
    models = list()
    # define config lists
    t_params = ['add', 'mul', None]
    d_params = [True, False]
    s_params = ['add', 'mul', None]
    p_params = seasonal
    b_params = [True, False]
    r_params = [True, False]
    # create config instances
    for t in t_params:
        for d in d_params:
            for s in s_params:
                for p in p_params:
                    for b in b_params:
                        for r in r_params:
                            cfg = [t,d,s,p,b,r]
                            models.append(cfg)
    return models

def HoltWinters(variable): 
  if variable == 'Yamaha':
    data='RUNT YAMAHA'
  elif variable == 'Mercado':
    data='RUNT MERCADO'
  
  st.write('Por favor, ingrese cuantos meses hacia adelante desea estimar la demanda de ' + variable)
  MES = st.number_input("Meses",value=12)
  MES=int(MES)
  

  df=pd.read_excel('BD_actualizado.xlsx')
  df3=df.copy()
  df3=df3.reset_index(drop=True)
  df3.set_index('FECHA', inplace=True)
  df3.index.freq = 'MS'
  df3=df3[data]
  df3=df3.dropna()
  if variable == 'Mercado':
    df3=df3[48:] #los primeros datos de 2001 a 2004 de mercado son tan bajitos que generan mucho error al tratar de ajustar un modelo teniendolos en cuenta
  
  cfg_list = exp_smoothing_configs(seasonal=[12]) #[0,6,12]

  if (st.button('Pronosticar')):
    train_size = int(len(df3) * 0.85) #4000
    test_size = len(df3) - train_size #1000
    ts = df3.iloc[0:train_size].copy()
    ts_v = df3.iloc[train_size:len(df3)].copy()
    ind = df3.index[-test_size:]  # this will select last 12 months' indexes
    #st.write(ind)
    #st.write(ts_v)

    best_RMSE = np.inf
    best_config = []
    t1 = d1 = s1 = p1 = b1 = r1 = None
    mape=[]
    y_forecast=[]
    model=()
    
    my_bar = st.progress(0)
    status_text = st.empty()
    for j in range(len(cfg_list)):
        try:
            cg = cfg_list[j]
            t,d,s,p,b,r = cg
            # define model
            if (t == None):
                model = HWES(ts, trend=t, seasonal=s, seasonal_periods=p)
            else:
                model = HWES(ts, trend=t, damped=d, seasonal=s, seasonal_periods=p)
            # fit model
            model_fit = model.fit(optimized=True, use_boxcox=b, remove_bias=r)
            y_forecast = model_fit.forecast(test_size)
            pred_ = pd.Series(data=y_forecast, index=ind)
            #df_pass_pred = pd.concat([ts_v, pred_.rename('pred_HW')], axis=1)
            #st.write(df_pass_pred)
            mape=mean_absolute_percentage_error(ts_v,y_forecast)
            #rmse = np.sqrt(mean_squared_error(ts_v,y_forecast))
             
            if mape < best_RMSE: # lo cambie por MAPE en vez de RMSE
                best_RMSE = mape
                best_config = cfg_list[j]
        except:
          continue
        time.sleep(0.1)
        status_text.write('Calculando')
        if j==(len(cfg_list)-1):
          j=100
        my_bar.progress(j)
        
    status_text.write('Listo!')
    #st.write(best_config)
    t1,d1,s1,p1,b1,r1 = best_config

    # Entreno el modelo con los parametros hallados (uno esta entrenando con el set de train -hw_model1- para obtener errores y otro con todo 
    # el dataset para obtener pronosticos -hw-)
    if t1 == None:
        hw_model1 = HWES(ts, trend=t1, seasonal=s1, seasonal_periods=p1)
        hw = HWES(df3, trend=t1, seasonal=s1, seasonal_periods=p1)
    else:
        hw_model1 = HWES(ts, trend=t1, seasonal=s1, seasonal_periods=p1, damped=d1)
        hw = HWES(df3, trend=t1, seasonal=s1, seasonal_periods=p1, damped=d1)
     
    fit2 = hw_model1.fit(optimized=True, use_boxcox=b1, remove_bias=r1)
    pred_HW = fit2.predict(start=pd.to_datetime(ts_v.index[0]), end = pd.to_datetime(ts_v.index[len(ts_v)-1]))
    pred_HW = pd.Series(data=pred_HW, index=ind)

     
    fitted = hw.fit(optimized=True, use_boxcox=b1, remove_bias=r1)
    y_hat=fitted.forecast(steps=MES)
    
    modelo = HWES(ts,seasonal_periods = 12,trend = 'add',seasonal = 'add')
    fitted_wo = modelo.fit(optimized=True, use_brute=True)
    pred = fitted_wo.predict(start=pd.to_datetime(ts_v.index[0]), end = pd.to_datetime(ts_v.index[len(ts_v)-1]))
    pred = pd.Series(data=pred, index=ind)
    

    model=HWES(df3,seasonal_periods = 12,trend = 'add',seasonal = 'add')
    fit=model.fit(optimized=True, use_boxcox=True, remove_bias=True)
    y_hat2=fit.forecast(steps=MES)

    tiempo=[]
    nuevo_index=[]
    for i in range(0,MES,1):
      a=df3.index[len(df3)-1]+relativedelta(months=+(1+i)) 
      b=a.strftime('%d/%m/%Y')
      nuevo_index.append(a)
      tiempo.append(b)

    st.markdown('**Pronósticos: **')
    resultados=pd.DataFrame({'Resultados optimizados': np.around(y_hat).ravel(),'Resultados sin optimizar': np.around(y_hat2).ravel()}, index=tiempo)
    csv = resultados.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}">aquí</a>'
    st.markdown('Si desea descargar el pronóstico de click '+href, unsafe_allow_html=True)
    st.write(resultados)

    #Agregar descarga csv
    #csv = resultados.to_csv(index=False)
    #b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    #href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (right-click and save as &lt;some_name&gt;.csv)'
    #href = f'<a href="data:file/csv;base64,{b64}">Descargue pronóstico</a> (defina nombre y agregue la extensión del archivo .csv)'
    #st.markdown(href, unsafe_allow_html=True)

    st.write('**Errores**')
    MAE_Opt="{:.0f}".format(mean_absolute_error(ts_v, pred_HW))
    MAPE_Opt="{:.2%}".format(mean_absolute_percentage_error(ts_v, pred_HW))

    MAE_SinOpt="{:.0f}".format(mean_absolute_error(ts_v, pred))
    MAPE_SinOpt="{:.2%}".format(mean_absolute_percentage_error(ts_v, pred))

    errores=pd.DataFrame()
    errores['Errores']=['MAE','MAPE']
    errores['Optimizado']=[MAE_Opt,MAPE_Opt]
    errores['Sin optimizar']=[MAE_SinOpt, MAPE_SinOpt]
    errores.set_index('Errores',inplace=True)
    st.write(errores)

    #Gráfica 1
    anio='2015' #para determinar desde que año se va a graficar
    agrupados=pd.DataFrame({'Optimizado': np.around(y_hat).ravel(),'Sin_optimizar': np.around(y_hat2).ravel()}, index=nuevo_index)
    total=pd.concat([df3[anio:], agrupados])
    total.rename(columns={0: 'Runt real'},inplace=True) #esta variable tiene estacionalidad
    total=total.reset_index()
    df_melt = total.melt(id_vars='index', value_vars=['Runt real','Optimizado','Sin_optimizar'])
    px.defaults.width = 1100
    px.defaults.height = 500
    fig = px.line(df_melt, x='index',y='value', color='variable', labels={"index": "Fecha",  "value": "Runt"})
    st.plotly_chart(fig)


    #Gráfica 2
    ajustados=pd.DataFrame({'Ajustado_optimizado': np.around(fitted.fittedvalues).ravel(),'Ajustado_sin_optimizar': np.around(fit.fittedvalues).ravel()}, index=df3.index)
    ajustados_total=pd.concat([df3[anio:], ajustados[anio:]], axis=1)
    ajustados_total=ajustados_total.reset_index()
    df_melt_fitted = ajustados_total.melt(id_vars='FECHA', value_vars=[data,'Ajustado_optimizado','Ajustado_sin_optimizar'])
    px.defaults.width = 1100
    px.defaults.height = 500
    fig = px.line(df_melt_fitted, x='FECHA',y='value', color='variable', labels={"FECHA": "Fecha",  "value": "Runt"})
    st.plotly_chart(fig)

    st.write('Aunque en las gráficas se observa el runt desde '+anio+' los modelos de predicción están construidos con datos desde el 2001 en el caso de yamaha y 2005 en el caso de Mercado')

def Holt_Winters():
  st.subheader("**Otra demanda a pronosticar**")
  st.write('En esta opción puede pronosticar demanda de categorías, modelos, marcas, wholesale, zonas, distribuidores, referencias de productos, etc., dependiendo de su necesidad, sin embargo debe tener en cuenta:')
  text3 = """
            * Los datos deben ser mensuales.
            * Los datos se deben subir como se muestra en la plantilla, no cambie el formato de fechas.
            * El pronóstico se calcula para una variable, si desea calcular multiples variables debe repetir el proceso para cada una de ellas.
              **Ejemplo:** si desea pronosticar categoría ON/OFF y SCOOTER, debe diligenciar la plantilla para ON/OFF, ejecutar el proceso y luego repetirlo para SCOOTER.
            * Tenga en cuenta que si usted tiene datos atípicos dentro del histórico es probable que obtenga modelos con errores muy altos y además que reflejen poco la realidad, considere siempre analizar si hubo eventos que condicionaron la demanda y debe modificar algún valor para llevarlo a algo más "normal".

            """
  st.markdown(text3)    
  # Cargar plantilla para que usuario descargue
  plantilla_otros=pd.read_excel('plantilla_otros.xlsx')
  csv = plantilla_otros.to_csv(index=False)
  b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
  href = f'<a href="data:file/csv;base64,{b64}">Descargue plantilla</a> (nombre como quiera y agregue la extensión del archivo .csv)'
  st.markdown(href, unsafe_allow_html=True)

  st.markdown('**Subir plantilla**')
  st.write(":warning: Una vez diligenciada la plantilla vuelva a subir a la aplicación el archivo con formato 'xlsx' :warning:")
  #st.write(":warning: El archivo que suba debe tener extensión 'xlsx'")
  data_file=st.file_uploader('Archivo',type=['xlsx'])
  if data_file is not None:
    df_p=pd.read_excel(data_file)  
    st.write('Por favor, ingrese cuantos meses hacia adelante desea estimar la demanda')
    MES = st.number_input("Meses",value=12)
    MES=int(MES)

    df_p.set_index('Fecha', inplace=True)
    df_p.index.freq = 'MS'
    df_p=df_p.dropna()
    cfg_list = exp_smoothing_configs(seasonal=[12]) #[0,6,12]

    if (st.button('Pronosticar')):
      train_size = int(len(df_p) * 0.85) #4000
      test_size = len(df_p) - train_size #1000
      ts = df_p.iloc[0:train_size].copy()
      ts_v = df_p.iloc[train_size:len(df_p)].copy()
      ind = df_p.index[-test_size:]  # this will select last 12 months' indexes
      #st.write(ind)
      #st.write(ts_v)

      best_RMSE = np.inf
      best_config = []
      t1 = d1 = s1 = p1 = b1 = r1 = None
      mape=[]
      y_forecast=[]
      model=()
      
      my_bar = st.progress(0)
      status_text = st.empty()
      for j in range(len(cfg_list)):
          try:
              cg = cfg_list[j]
              t,d,s,p,b,r = cg
              # define model
              if (t == None):
                  model = HWES(ts, trend=t, seasonal=s, seasonal_periods=p)
              else:
                  model = HWES(ts, trend=t, damped=d, seasonal=s, seasonal_periods=p)
              # fit model
              model_fit = model.fit(optimized=True, use_boxcox=b, remove_bias=r)
              y_forecast = model_fit.forecast(test_size)
              pred_ = pd.Series(data=y_forecast, index=ind)
              #df_pass_pred = pd.concat([ts_v, pred_.rename('pred_HW')], axis=1)
              #st.write(df_pass_pred)
              mape=mean_absolute_percentage_error(ts_v,y_forecast)
              #rmse = np.sqrt(mean_squared_error(ts_v,y_forecast))
               
              if mape < best_RMSE: # lo cambie por MAPE en vez de RMSE
                  best_RMSE = mape
                  best_config = cfg_list[j]
          except:
            continue
          time.sleep(0.1)
          status_text.write('Calculando')
          if j==(len(cfg_list)-1):
            j=100
          my_bar.progress(j)
          
      status_text.write('Listo!')
      #st.write(best_config)
      t1,d1,s1,p1,b1,r1 = best_config

      # Entreno el modelo con los parametros hallados (uno esta entrenando con el set de train -hw_model1- para obtener errores y otro con todo 
      # el dataset para obtener pronosticos -hw-)
      if t1 == None:
          hw_model1 = HWES(ts, trend=t1, seasonal=s1, seasonal_periods=p1)
          hw = HWES(df_p, trend=t1, seasonal=s1, seasonal_periods=p1)
      else:
          hw_model1 = HWES(ts, trend=t1, seasonal=s1, seasonal_periods=p1, damped=d1)
          hw = HWES(df_p, trend=t1, seasonal=s1, seasonal_periods=p1, damped=d1)
       
      fit2 = hw_model1.fit(optimized=True, use_boxcox=b1, remove_bias=r1)
      pred_HW = fit2.predict(start=pd.to_datetime(ts_v.index[0]), end = pd.to_datetime(ts_v.index[len(ts_v)-1]))
      pred_HW = pd.Series(data=pred_HW, index=ind)

       
      fitted = hw.fit(optimized=True, use_boxcox=b1, remove_bias=r1)
      y_hat=fitted.forecast(steps=MES)
      
      modelo = HWES(ts,seasonal_periods = 12,trend = 'add',seasonal = 'add')
      fitted_wo = modelo.fit(optimized=True, use_brute=True)
      pred = fitted_wo.predict(start=pd.to_datetime(ts_v.index[0]), end = pd.to_datetime(ts_v.index[len(ts_v)-1]))
      pred = pd.Series(data=pred, index=ind)
      

      model=HWES(df_p,seasonal_periods = 12,trend = 'add',seasonal = 'add')
      fit=model.fit(optimized=True, use_boxcox=True, remove_bias=True)
      y_hat2=fit.forecast(steps=MES)

      tiempo=[]
      nuevo_index=[]
      for i in range(0,MES,1):
        a=df_p.index[len(df_p)-1]+relativedelta(months=+(1+i)) 
        b=a.strftime('%d/%m/%Y')
        nuevo_index.append(a)
        tiempo.append(b)

      st.markdown('**Pronósticos: **')
      resultados=pd.DataFrame({'Resultados optimizados': np.around(y_hat).ravel(),'Resultados sin optimizar': np.around(y_hat2).ravel()}, index=tiempo)
      csv = resultados.to_csv(index=False)
      b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
      href = f'<a href="data:file/csv;base64,{b64}">aquí</a>'
      st.markdown('Si desea descargar el pronóstico de click '+href, unsafe_allow_html=True)
      st.write(resultados)

      st.write('**Errores**')
      MAE_Opt="{:.0f}".format(mean_absolute_error(ts_v, pred_HW))
      MAPE_Opt="{:.2%}".format(mean_absolute_percentage_error(ts_v, pred_HW))

      MAE_SinOpt="{:.0f}".format(mean_absolute_error(ts_v, pred))
      MAPE_SinOpt="{:.2%}".format(mean_absolute_percentage_error(ts_v, pred))

      errores=pd.DataFrame()
      errores['Errores']=['MAE','MAPE']
      errores['Optimizado']=[MAE_Opt,MAPE_Opt]
      errores['Sin optimizar']=[MAE_SinOpt, MAPE_SinOpt]
      errores.set_index('Errores',inplace=True)
      st.write(errores)

      #Gráfica 1
      anio='2015' #para determinar desde que año se va a graficar
      agrupados=pd.DataFrame({'Optimizado': np.around(y_hat).ravel(),'Sin_optimizar': np.around(y_hat2).ravel()}, index=nuevo_index)
      total=pd.concat([df_p[anio:], agrupados])
      total.rename(columns={0: 'Demanda'},inplace=True) #esta variable tiene estacionalidad
      total=total.reset_index()
      df_melt = total.melt(id_vars='index', value_vars=['Demanda','Optimizado','Sin_optimizar'])
      px.defaults.width = 1100
      px.defaults.height = 500
      fig = px.line(df_melt, x='index',y='value', color='variable', labels={"index": "Fecha",  "value": "Demanda"})
      st.plotly_chart(fig)


      #Gráfica 2
      ajustados=pd.DataFrame({'Ajustado_optimizado': np.around(fitted.fittedvalues).ravel(),'Ajustado_sin_optimizar': np.around(fit.fittedvalues).ravel()}, index=df_p.index)
      ajustados_total=pd.concat([df_p[anio:], ajustados[anio:]], axis=1)
      ajustados_total=ajustados_total.reset_index()
      df_melt_fitted = ajustados_total.melt(id_vars='Fecha', value_vars=['Demanda','Ajustado_optimizado','Ajustado_sin_optimizar'])
      px.defaults.width = 1100
      px.defaults.height = 500
      fig = px.line(df_melt_fitted, x='Fecha',y='value', color='variable', labels={"Fecha": "Fecha",  "value": "Demanda"})
      st.plotly_chart(fig)

    



#APLICACIÓN

st.title("Pronósticos motocicletas - Incolmotos Yamaha")
#img = Image.open(get_file_content_as_string("YAMAHA.PNG"))
#st.sidebar.image(img, width=250)
st.sidebar.image(
    "https://raw.githubusercontent.com/Analiticadatosiy/Pronosticos/master/YAMAHA.PNG?token=ATEVFY6JYIBS3BZZKNKD5ADAWVFBY", width=250
    #"https://raw.githubusercontent.com/Analiticadatosiy/Pronosticos/master/YAMAHA.PNG", width=250
    #"https://rasahq.github.io/rasa-nlu-examples/square-logo.svg", width=100
)

status = st.sidebar.radio("Cual es su objetivo", ("Informativo", "Pronosticar"))


if status=="Informativo":
  st.markdown("---")
  st.write('Esta aplicación se construye con el propósito de soportar las decisiones relacionadas con las proyecciones de motocicletas (MTP - Medium Term Plan), a continuación se detalla la metodología y los datos asociados:')

  st.markdown("\n ## Datos")
  st.write('Estos pronósticos se han construido con datos históricos del runt desde 2001, además de otras variables macroeconómicas (en algunas metodologías):')
  text3 = """
            * Desempleo
            * Inflación
            * Salario mínimo (SMMLV) + auxilio TTE
            * Indice de expectativas del consumidor (IEC) *
            * Indice de condiciones económicas (ICE) *
            * Precio promedio mensual del petroleo (WTI)
            * Tasa representativa del mercado (TRM)
            * Proporción días hábiles y festivos (en el mes)

            *Estos índices se utilizan para el cálculo del índice de confianza del consumidor (ICC) de Fedesarrollo
            """
  st.markdown(text3)    
  st.markdown("\n ## Recomendaciones de uso")
  st.write('Dependiendo de las necesidades del usuario y los datos que tenga disponible, existen tres opciones para la creación de los pronósticos:')
  #st.markdown("### Estos pronósticos se han construido con datos históricos desde 2001, teniendo en cuenta variables macroeconómicas\n\n")
  text2 = """
            * **Trabajando con los últimos 12 meses (t-12):** se utilizan los datos tanto de ventas como de las variables macroeconómicas de los últimos 12 meses
              para predecir los próximos 12 meses hacia adelante.
            * **Suponiendo el comportamiento futuro de las variables macroeconómicas:** en este caso se debe ingresar a la aplicación el valor que tomarán las variables
              macroeconomicas en los n meses hacia adelante que se quieran pronosticar.
            * **Sólo con el histórico de ventas:** el usuario sólo debe ingresar cuántos meses hacia adelante quiere pronosticar, pero dichos pronósticos sólo dependerán del
              histórico.
            """
  st.markdown(text2)    
  st.markdown("\n ## Metodologías")
  st.write('Se construyeron estos pronósticos alrededor de cuatro metodologías:')
  text = """
            * Redes neuronales artificiales (RNN)
            * XGBoost (XGB)
            * Random Forest (RF)
            * Holt Winters (HW) \n
            De éstas, las tres primeras tienen asociadas variables externas como la TRM, desempleo, inflación, etc. Mientras que la última (HW) sólo pronostica basándose en el comportamiento histórico de la demanda.
            """
  st.markdown(text)  
  st.markdown("\n ## Errores")
  st.write('Para entender que tan acertado es un método de pronóstico se utilizan -en este caso- dos medidas de error (MAE y MAPE):')
  text = """
            * **Error medio absoluto (MAE)**: es el promedio de las diferencias absolutas entre los valores reales y los valores pronosticados.
            """
  st.markdown(text) 
  #img_MAE = Image.open("MAE.jpg")
  st.image("https://raw.githubusercontent.com/Analiticadatosiy/Pronosticos/master/MAE.JPG?token=ATEVFY3U2WJFFMHXGNGTU73AWVFF4", width=200)
  
  text = """
            * **Error medio absoluto porcentual (MAPE)**: es el porcentaje promedio de desviación respecto al valor real.\n
                        """
  st.markdown(text)
  #img_MAPE= Image.open("MAPE.jpg")
  st.image("https://raw.githubusercontent.com/Analiticadatosiy/Pronosticos/master/MAPE.JPG?token=ATEVFYZAOBQ5DZK5LAYIKKTAWVFII", width=200)
  st.markdown('Sin embargo, es importante entender que muchas veces el error puede estar inducido por factores externos que condicionan el valor real, por ejemplo si un mes se pronostica vender 3.000 motocicletas pero no tenemos inventario y sólo vendemos 1.500, impactará mucho al error porque el pronostico se alejó mucho de la realidad, por tanto se sugiere un análisis a los datos usados para el pronóstico y a los valores pronosticados además de los errores calculados. ')


else:
  st.markdown("---")
  opciones1=['Seleccione demanda','Yamaha','Mercado', 'Otra demanda']
  opciones2=['Seleccione dinámica','Suponiendo indicadores económicos','Datos reales último año', 'Sólo con el histórico de ventas']
  opciones3=['Seleccione alcance','Predicción de un sólo mes', 'Predicción varios meses']
  # Add a selectbox to the sidebar:
  selectbox1 = st.sidebar.selectbox('¿Qué demanda desea estimar?',opciones1)

  if selectbox1 == "Yamaha":
    selectbox2 = st.sidebar.selectbox('Cómo desea hacer la estimación: ', opciones2)
    if selectbox2 == 'Suponiendo indicadores económicos':
      selectbox3 = st.sidebar.selectbox('Alcance: ', opciones3)
      if selectbox3 == 'Predicción de un sólo mes':
        actual_individual('Yamaha')
      elif selectbox3 == 'Predicción varios meses':
        actual_lote('Yamaha')
    elif selectbox2 == 'Datos reales último año':
      selectbox3 = st.sidebar.selectbox('Alcance:', opciones3)
      if selectbox3 == 'Predicción de un sólo mes':
        rezago_yamaha()
      elif selectbox3 == 'Predicción varios meses':
        rezago_yamaha_lote()    
    elif selectbox2 == 'Sólo con el histórico de ventas':
      HoltWinters('Yamaha')
  elif selectbox1 == "Mercado":
    selectbox2 = st.sidebar.selectbox('Cómo desea hacer la estimación: ', opciones2)
    if selectbox2 == 'Suponiendo indicadores económicos':
      selectbox3 = st.sidebar.selectbox('Alcance: ', opciones3)
      if selectbox3 == 'Predicción de un sólo mes':
        actual_individual('Mercado')
      elif selectbox3 == 'Predicción varios meses':
        actual_lote('Mercado')
    elif selectbox2 == 'Datos reales último año':
      selectbox3 = st.sidebar.selectbox('Alcance: ', opciones3)
      if selectbox3 == 'Predicción de un sólo mes':
        rezago_mercado()
      elif selectbox3 == 'Predicción varios meses':
        rezago_mercado_lote()
    elif selectbox2 == 'Sólo con el histórico de ventas':
      HoltWinters('Mercado')
  elif selectbox1 == "Otra demanda":
    Holt_Winters()
  
st.sidebar.subheader('Creado por:')
st.sidebar.write('Analítica de Datos') 