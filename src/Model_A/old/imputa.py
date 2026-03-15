import numpy as np
import pandas as pd
import glob
import pickle
from pyrle import Rle

# Leo las variables analógicas
analogicas = pd.read_pickle("analogicas.pkl")
print("leidos %d registros analógicos" % len(analogicas))

# hacemos la imputación de valores analógicos

blocks = [
	["Vrms1", "Vrms2", "Vrms3"],
	["RP1", "RP2", "RP3"],
	["Irms1", "Irms2", "Irms3"],
	["PF1", "PF2", "PF3"]
]

# quitamos los NANs de todas las variables.
# LeChacal se atasca con esta velocidad de muestreo

# se podrían quitar sin más, pero vamos a intentar
# la imputación con los valores a derecha o izquierda.

for cols in blocks:
	for v in cols:
		nans = np.isnan(analogicas[v].values)
		print("%s: %d NaNs a imputar " % (v, sum(nans)))
		analogicas[v].values[nans] = analogicas[cols][nans].min(axis=1,skipna=True)

		nans = np.isnan(analogicas[v].values)
		print("%s: %d NaNs después de imputar MIN (%s)" % (v, sum(nans), ",".join(cols)))

		if sum(nans):
			analogicas[v].values[nans] = analogicas[v].iloc[1+np.where(nans)[0]]
			nans = np.isnan(analogicas[v].values)
			print("%s: %d NaNs después de imputar NEXT RP#" % (v, sum(nans)))

		if sum(nans):
			analogicas[v].values[nans] = analogicas[v].iloc[np.where(nans)[0] - 1]
			nans = np.isnan(analogicas[v].values)
			print("%s: %d NaNs después de imputar PRIOR RP#" % (v, sum(nans)))

blocks = [
	["Vrms4"],["RP4"],["Irms4"],["PF4"]
]

for cols in blocks:
	for v in cols:
		nans = np.isnan(analogicas[v].values)
		print("%s: %d NaNs a imputar " % (v, sum(nans)))

		if sum(nans):
			res = Rle(nans).to_frame().astype('int32')
			pos = 0

			for i,row in res.iterrows():
				# para cada ventana de nans asignamos el anterior/posterior más cercano
				if row.Values:
					for j in np.arange(pos, pos+row.Runs):
						d1 = j-pos+1
						d2 = pos+row.Runs-j
						analogicas[v].values[j] = (analogicas[v].values[pos-1]*d2 + analogicas[v].values[pos+row.Runs]*d1)/(d1+d2)

				pos+=row.Runs

		nans = np.isnan(analogicas[v].values)
		print("%s: %d NaNs a imputar " % (v, sum(nans)))

analogicas = analogicas.dropna()
print("imputados %d registros analógicos" % len(analogicas))

#
analogicas.to_pickle("analogicas_nonans.pkl")
