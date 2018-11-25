#Import Flask
from flask import Flask, request
from cnn_executor import cargarModelo,cargarModelo_3v,cargarModelo_4v
import pandas as pd
import datetime
import numpy as np
from keras.models import model_from_json
import numpy as np
# Escalamiento/Norm de Features (Parallel Calculations - Highly Important)
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing.imputation import Imputer

#Initialize the application service
app = Flask(__name__)
global loaded_model, graph
loaded_model, graph = cargarModelo()

#Define a route
@app.route('/')
def main_page():
	return 'Bienvenido Prediccion de Generacion de valor de las empresas !'

@app.route('/genera_valor/', methods=['GET','POST'])
def default():
	return 'Modelo de prediccion Generacion de valor de las empresas !'

@app.route('/genera_valor/modelo_7v/', methods=['GET','POST'])
def modelo_7v():
	print (request.args)
	# dimensions of our images.
	
	# Show
	datatest_name = request.args.get("datacsv")
	data_path='../samples/'+datatest_name+'.csv'
		
	dataset 	= pd.read_csv(data_path,delimiter='\t')
	# imp = SimpleImputer(missing_values=np.nan, strategy='mean')
	sc = StandardScaler()
	#imputacion de datos(datos nulos)
	imp = Imputer()

	X_ID 		= dataset.iloc[:, 0].values
	X_testing 	= dataset.iloc[:, 1:8].values
	#imputacion de datos(datos nulos)
	imp = Imputer()
	imp.fit(X_testing)
	X_test = imp.transform(X_testing)
	X_test = sc.fit_transform(X_test,)

	#prediccion
	
	with graph.as_default():
		y_pred = loaded_model.predict(X_test)
		resultado_final=''
		for i in range(0,len(y_pred)):
			
			if y_pred[i] > 0.5:
				print(X_ID[i], ' --> Genera Valor!')
				resultado = str(X_ID[i])+ ' --> Genera Valor!! '
			else:
				print(X_ID[i], ' --> No genera Valor ')
				resultado = str(X_ID[i])+ ' --> No genera Valor '
			resultado_final = resultado_final+resultado+'\n'

		#print('Prediccion:', score, ' Gato ' if score < 0.5 else ' Perro')
		return resultado_final

@app.route('/genera_valor/modelo_4v/', methods=['GET','POST'])
def modelo_4v():
	print (request.args)
	loaded_model, graph = cargarModelo_4v()
	# dimensions of our images.
	
	# Show
	datatest_name = request.args.get("datacsv")
	data_path='../samples/'+datatest_name+'.csv'
		
	dataset 	= pd.read_csv(data_path,delimiter='\t')
	# imp = SimpleImputer(missing_values=np.nan, strategy='mean')
	sc = StandardScaler()
	#imputacion de datos(datos nulos)
	imp = Imputer()

	X_ID 		= dataset.iloc[:, 0].values
	X_testing 	= dataset.iloc[:, 1:5].values
	#imputacion de datos(datos nulos)
	imp = Imputer()
	imp.fit(X_testing)
	X_test = imp.transform(X_testing)
	X_test = sc.fit_transform(X_test,)

	#prediccion
	
	with graph.as_default():
		y_pred = loaded_model.predict(X_test)
		resultado_final=''
		for i in range(0,len(y_pred)):
			
			if y_pred[i] > 0.5:
				print(X_ID[i], ' --> Genera Valor!')
				resultado = str(X_ID[i])+ ' --> Genera Valor!! '
			else:
				print(X_ID[i], ' --> No genera Valor ')
				resultado = str(X_ID[i])+ ' --> No genera Valor '
			resultado_final = resultado_final+resultado+'\n'

		#print('Prediccion:', score, ' Gato ' if score < 0.5 else ' Perro')
		return resultado_final


@app.route('/genera_valor/modelo_3v/', methods=['GET','POST'])
def modelo_3v():
	print (request.args)
	loaded_model, graph = cargarModelo_3v()
	# dimensions of our images.
	
	# Show
	datatest_name = request.args.get("datacsv")
	data_path='../samples/'+datatest_name+'.csv'
		
	dataset 	= pd.read_csv(data_path,delimiter='\t')
	# imp = SimpleImputer(missing_values=np.nan, strategy='mean')
	sc = StandardScaler()
	#imputacion de datos(datos nulos)
	imp = Imputer()

	X_ID = dataset.iloc[:, 0].values
	X_testing = dataset.iloc[:, 5:8].values
	#imputacion de datos(datos nulos)
	imp = Imputer()
	imp.fit(X_testing)
	X_test = imp.transform(X_testing)
	X_test = sc.fit_transform(X_test,)

	#prediccion
	
	with graph.as_default():
		y_pred = loaded_model.predict(X_test)
		resultado_final=''
		for i in range(0,len(y_pred)):
			
			if y_pred[i] > 0.5:
				print(X_ID[i], ' --> Genera Valor!')
				resultado = str(X_ID[i])+ ' --> Genera Valor!! '
			else:
				print(X_ID[i], ' --> No genera Valor ')
				resultado = str(X_ID[i])+ ' --> No genera Valor '
			resultado_final = resultado_final+resultado+'\n'

		#print('Prediccion:', score, ' Gato ' if score < 0.5 else ' Perro')
		return resultado_final
		
# Run de application
app.run(host='0.0.0.0',port=5200)
