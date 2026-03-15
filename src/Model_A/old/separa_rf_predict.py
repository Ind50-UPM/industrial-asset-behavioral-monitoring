import numpy as np
import pandas as pd
import glob
import pickle
import datetime
from pyrle import Rle

from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

digitales = pd.read_pickle("digitales.pkl")
print("leidos %d registros digitales" % len(digitales))
print(digitales.index[0], digitales.index[-1])

analogicas = pd.read_pickle("analogicas_nonans.pkl")
print("leidos %d registros analógicos" % len(analogicas))
print(analogicas.index[0], analogicas.index[-1])

threshold = 50
maxs = analogicas[['RP1','RP2','RP3']].max(axis=1)
zeros = maxs.values < threshold
analogicas.loc[zeros,'estado']=0

# asignamos el estado DIGITAL a los valores analógicos
# cogemos el instante más cercano
idx = digitales[digitales.estado!=0].index[digitales[digitales.estado!=0].index.get_indexer(analogicas[~zeros].index, method='pad')]
analogicas.loc[~zeros, 'estado'] = digitales.estado[idx].values
analogicas['estado'] = analogicas['estado'].astype(np.int32)

# cargamos los modelos entrenados previamente

model_vars, scaler, clf = pickle.load(open('separa_rf.pkl', 'rb'))

# preparamos las variable para la predicción

timefmt = '%Y-%m-%d %H:%M:%S.%f%z'

for i in np.arange(1,29):
	pred_ini = "2022-02-%02d 00:00:00.000000+0100" % i
	pred_end = (datetime.datetime.strptime(pred_ini, timefmt)+datetime.timedelta(hours=24)).strftime(timefmt)

	df2 = digitales.loc[(digitales.index > pred_ini) & (digitales.index < pred_end)]
	adf2 = analogicas.loc[(analogicas.index > pred_ini) & (analogicas.index < pred_end)]

	print("PREDICT START:", pred_ini)
	print("PREDICT END:", pred_end)

	print("leidos %d registros digitales" % len(df2))
	print("leidos %d registros analógicos" % len(adf2))

	# predecimos los estados, se supone que NO los conocemos
	# quito todos los "ceros", valores de consumo por debajo de un umbral.
	maxs = adf2[['RP1','RP2','RP3']].max(axis=1)
	zeros = maxs.values < threshold
	print("%d registros nulos" % sum(zeros))

	X2 = adf2[~zeros][model_vars]
	y2 = adf2[~zeros].estado
	print("RF: %d elementos" % len(X2))

	if len(X2):
		X3 = scaler.transform(X2)
		y3 = clf.predict(X3)
		print("prediction accuracy (%d):" % sum(y2==y3), accuracy_score(y2, y3))
	else:
		print("no elements to predict")
