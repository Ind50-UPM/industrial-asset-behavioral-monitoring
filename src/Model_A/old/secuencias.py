import numpy as np
import pandas as pd
import pickle,time

# 6	38A5  ARRANQUE DIVING PUMP 1
# 7	38A6  ARRANQUE DIVING PUMP 2
# 8	38A7  ARRANQUE FEED BACK PUMP 1
# 9	38B0  ARRANQUE FEED BACK PUMP 2
# 10  38B1  ARRANQUE BOMBA FLOCULANTE
# 11  38B2  BOMBA MOTOR NO BALSA

digitales = pd.read_pickle("digitales.pkl")
digitales.columns = ['38A5','38A6','38A7','38B0','38B1','38B2', 'estado']
print("Leídos %d registros digitales" % len(digitales))

# agrupamos los estados y vemos cuantos hay de cada

df2 = digitales.groupby(digitales.estado).size().reset_index(name='counts')
df2 = df2.set_index('estado')
df3 = df2.sort_values(by=['counts'], ascending=False)

#print(df2)
print(df3)

from pyrle import Rle

estados = Rle(digitales.estado.values[:-1]).to_frame().astype('int32')

# calculo los tiempos de inicio,final de cada secuencia de estados
start = np.insert(np.cumsum(estados.Runs.values[:-1]),0,0)
stop = np.cumsum(estados.Runs.values)

# añado la información al dataframe con el resto de información
estados.index=digitales.index[start]
estados['span']=(digitales.index[stop]-digitales.index[start]).total_seconds()
estados['start']=start
estados['stop']=stop

print("clústers estados =", len(estados))

# juntamos los estados con duración inferior al segundo con los siguientes
cond = (estados.span<1) & (estados.Runs.values==1)
sueltos = estados[cond]
digitales.loc[sueltos.index, 'estado'] = estados.loc[estados.iloc[1+np.where(cond)[0]].index, 'Values'].values

from numba import njit
@njit
def n_ranges_nb(t1, t2):
	a = np.arange(np.max(t2)+1)
	n = (t2 - t1).sum()
	out = np.zeros(n, dtype=np.int64)
	l, l_old = 0, 0
	for i,j in zip(t1, t2):
		l += j-i
		out[l_old:l] = a[i:j]
		l_old = l
	return out

cond = (estados.span<1.5) & (estados.Runs.values>1)
sueltos = estados[cond]
restado = np.repeat(estados.loc[estados.iloc[1+np.where(cond)[0]].index, 'Values'].values, sueltos.Runs.values)
idxs = digitales.iloc[n_ranges_nb(sueltos.start.values, sueltos.stop.values)].index
digitales.loc[idxs, 'estado'] = restado

# recalculamos los clústers
nestados = Rle(digitales.estado.values[:-1]).to_frame().astype('int32')
start = np.insert(np.cumsum(nestados.Runs.values[:-1]),0,0)
stop = np.cumsum(nestados.Runs.values)
nestados.index=digitales.index[start]
nestados['span']=(digitales.index[stop]-digitales.index[start]).total_seconds()

print("clústers estados corregidos =", len(nestados))

# calculo los arranques de las operaciones: 38A6

opers = Rle(digitales['38A6'].values).to_frame().astype('int32')
start = np.insert(np.cumsum(opers.Runs.values[:-1]),0,0)
opers.index = digitales.index[start]

opers = opers[opers.Values == 1]

#print("operaciones= ",len(opers))

# verifico el número de operaciones diarias
df4 = opers.groupby(opers.index.strftime("%Y-%m-%d")).size().reset_index(name='counts')
df4 = df4.rename(columns={0: 'day'})
df4 = df4.set_index('day')

# redondeamos la fecha al MES y contamos cada grupo
df5 = opers.groupby(opers.index.strftime("%Y/%m")).size().reset_index(name='counts')
df5 = df5.rename(columns={0: 'month'})
df5 = df5.set_index('month')

#print(df4)
#print(df5)

# analizamos las ventanas temporales vinculadas a las operaciones
# seleccionamos los estados correspondientes a cada operacion.
#
# se toma el tiempo de referencia de la operación y se resta/suma un offset
# definido previamente. Con estos estados se pueden entrenar modelos o
# extraer el patrón de funcionamiento de la instalación. Comparando los
# patrones (a medida que se generan) con los valores históricos.


seqs = Rle(nestados.Values>0).to_frame().astype('int32')
start = np.insert(np.cumsum(seqs.Runs.values[:-1]),0,0)
seqs.index = nestados.index[start]
end=start+seqs.Runs.values
seqs['start'] = start
seqs['end'] = end

def f(x):
	return nestados.iloc[x.start:x.end].Values.tolist(), np.sum(nestados.iloc[x.start:x.end].span.values)

words = pd.DataFrame()
word,span = zip(*seqs[seqs.Runs>1].apply(f, axis=1))
words['word'] = np.asarray(word, dtype=object)
words['span'] = span
words.index = seqs[seqs.Runs>1].index

print("palabras =", len(words))

# 

print(words.word.value_counts())
