import numpy as np
import pandas as pd
import glob
import pickle
from pyrle import Rle

#train_ini = "2022-01-31 00:00:00.000000+01:00"
#train_end = "2022-02-01 00:00:00.000000+01:00"

#train_ini = "2022-01-25 00:00:00.000000+01:00"
#train_end = "2022-02-08 00:00:00.000000+01:00"

#train_ini = "2022-02-15 00:00:00.000000+01:00"
#train_end = "2022-02-16 00:00:00.000000+01:00"

train_ini = "2022-01-18 00:00:00.000000+01:00"
train_end = "2022-02-18 00:00:00.000000+01:00"

print("TRAIN START:", train_ini)
print("TRAIN END:", train_end)

digitales = pd.read_pickle("digitales.pkl")
df = digitales.loc[(digitales.index > train_ini) & (digitales.index < train_end)]
print("Seleccionados %d registros digitales de %d" % (len(df), len(digitales)))

# Leo las variables analógicas

analogicas = pd.read_pickle("analogicas_nonans.pkl")

threshold = 50 # consumo mínimo (W)
maxs = analogicas[['RP1','RP2','RP3']].max(axis=1)
zeros = maxs.values < threshold
analogicas.loc[zeros,'estado']=0

# asignamos el estado DIGITAL a los valores analógicos
# cogemos el instante más cercano
idx = digitales[digitales.estado!=0].index[digitales[digitales.estado!=0].index.get_indexer(analogicas[~zeros].index, method='pad')]
analogicas.loc[~zeros, 'estado'] = digitales.estado[idx].values
analogicas['estado'] = analogicas['estado'].astype(np.int32)

adf = analogicas.loc[(analogicas.index > train_ini) & (analogicas.index < train_end)]
print("Seleccionados %d registros analógicos de %d" % (len(adf), len(analogicas)))

# preparamos las variable para la predicción

#pred_ini = "2022-02-13 00:00:00.000000+01:00"
#pred_end = "2022-02-14 00:00:00.000000+01:00"

#pred_ini = "2022-02-14 00:00:00.000000+01:00"
#pred_end = "2022-02-15 00:00:00.000000+01:00"

pred_ini = "2022-02-19 00:00:00.000000+01:00"
pred_end = "2022-02-20 00:00:00.000000+01:00"

pred_ini = "2022-02-21 00:00:00.000000+01:00"
pred_end = "2022-02-22 00:00:00.000000+01:00"

df2 = digitales.loc[(digitales.index > pred_ini) & (digitales.index < pred_end)]
adf2 = analogicas.loc[(analogicas.index > pred_ini) & (analogicas.index < pred_end)]

# agrupamos los estados y vemos cuantos hay de cada

gdf = adf.groupby(adf.estado).size().reset_index(name='counts')

signals = [
	'DIVING PUMP 1',
	'DIVING PUMP 2',
	'FEEDBACK PUMP 1',
	'FEEDBACK PUMP 2',
	'BOMBA FLOCULANTE',
	'BOMBA BALSA'
]

lstatus = []

for index, row in gdf.iterrows():
	cad = "{0:b}".format(int(row['estado']))

	if row['estado']:
		status = ''

		for i,c in enumerate(cad[::-1]):
			if c != '0':
				status += ("," if status else "") + signals[i]
	else:
		status = 'PARADO'

	lstatus.append(status)

gdf['signals'] = lstatus
print(gdf.sort_values(by=['counts'], ascending=False))

from scipy import stats

# valores medios de los consumos para cada estado
for v in ['RP1', 'RP2', 'RP3']:
	gdf[v] = [stats.mode(adf[adf.estado==i][v], nan_policy='omit').mode for i in gdf.estado]

# estados que nos interesan, a la vista del funcionamiento de la instalación
#  3: DIVING PUMP 1 + DIVING PUMP 2
# 12: FEEDBACK PUMP 1 + FEEDBACK PUMP 2
# 16: BOMBA FLOCULANTE
# 32: BOMBA BALSA
#
# Normalmente aparecen combinados:
# 19 = 16+3
# 31 = 12+16+3
# 44 = 32+12
# 51 = 32+16+3

# creamos modelos de clasificación para los estados en función de los consumos

from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

model_vars = ['RP1','RP2','RP3','Vrms1','Vrms2','Vrms3','Irms1','Irms2','Irms3','PF1','PF2','PF3']
model_vars = ['Vrms1','Vrms2','Vrms3','Irms1','Irms2','Irms3','PF1','PF2','PF3']

X0 = adf[adf.estado!=0][model_vars]
print(round(X0.describe(),2))
y = adf[adf.estado!=0].estado

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(X0)
X = scaler.transform(X0)

print("RF: %d elementos (%d estados) / MODEL_VARS=%s" % (len(X), len(gdf), ",".join(model_vars)))

clf = RandomForestClassifier(n_estimators=100, criterion="entropy", verbose=False, n_jobs=8)
clf.fit(X, y)

pickle.dump([model_vars, scaler, clf], open('separa_rf.pkl', 'wb'))
print("training accuracy:", accuracy_score(y, clf.predict(X)))

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

X3 = scaler.transform(X2)
y3 = clf.predict(X3)
print("prediction accuracy (%d):" % sum(y2==y3), accuracy_score(y2, y3))

res = pd.DataFrame(index=X2.index.values, data=np.c_[y2.values,y3], columns=('y2','y3'))
res.index.names=['Time']
res.to_csv('separa_rf.txt')

import matplotlib.pylab as plt

plt.scatter(X2.index,y2, label="originales", marker='o')
plt.scatter(X2.index,y3, label="estimados", marker='+')
plt.title("FROM %s TO %s (%d)" % (
	adf2.index[0].strftime("%Y-%m-%d %H:%M:%S"),
	adf2.index[-1].strftime("%Y-%m-%d %H:%M:%S"),
	len(X2)))

plt.legend()
plt.gcf().set_size_inches((17., 8.25))
plt.gcf().set_tight_layout(True)
plt.show()
