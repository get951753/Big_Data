import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('C:/Users/get951753/Downloads/data.csv', index_col=0)

fig, axs = plt.subplots(1, 7, sharey=True)
data.plot(kind='scatter', x='money', y='ppl', ax=axs[0], figsize=(22, 7))
data.plot(kind='scatter', x='merry', y='ppl', ax=axs[1])
data.plot(kind='scatter', x='unmerry', y='ppl', ax=axs[2])
data.plot(kind='scatter', x='lostjob', y='ppl', ax=axs[3])
data.plot(kind='scatter', x='jobfight', y='ppl', ax=axs[4])
data.plot(kind='scatter', x='PM', y='ppl', ax=axs[5])
data.plot(kind='scatter', x='violent', y='ppl', ax=axs[6])

xx = data.money
xx2 = data.merry
xx3 = data.unmerry
xx4 = data.lostjob
xx5 = data.jobfight
xx6 = data.PM
xx7 = data.violent
yy = data.ppl

A = np.vstack([xx, np.ones(len(xx))]).T
A2 = np.vstack([xx2, np.ones(len(xx2))]).T
A3 = np.vstack([xx3, np.ones(len(xx3))]).T
A4 = np.vstack([xx4, np.ones(len(xx4))]).T
A5 = np.vstack([xx5, np.ones(len(xx5))]).T
A6 = np.vstack([xx6, np.ones(len(xx6))]).T
A7 = np.vstack([xx7, np.ones(len(xx7))]).T

m, c = np.linalg.lstsq(A, yy, rcond=None)[0]
m2, c2 = np.linalg.lstsq(A2, yy, rcond=None)[0]
m3, c3 = np.linalg.lstsq(A3, yy, rcond=None)[0]
m4, c4 = np.linalg.lstsq(A4, yy, rcond=None)[0]
m5, c5 = np.linalg.lstsq(A5, yy, rcond=None)[0]
m6, c6 = np.linalg.lstsq(A6, yy, rcond=None)[0]
m7, c7 = np.linalg.lstsq(A7, yy, rcond=None)[0]

fig = plt.figure(figsize=(23,6))

plt.subplot(1,7,1)
plt.plot(xx, yy, 'o', label='Original data', markersize=5)
plt.plot(xx, m*xx + c, 'r', label='Fitted line')

plt.subplot(1,7,2)
plt.plot(xx2, yy, 'o', label='Original data', markersize=5)
plt.plot(xx2, m2*xx2 + c2, 'r', label='Fitted line')

plt.subplot(1,7,3)
plt.plot(xx3, yy, 'o', label='Original data', markersize=5)
plt.plot(xx3, m3*xx3 + c3, 'r', label='Fitted line')

plt.subplot(1,7,4)
plt.plot(xx4, yy, 'o', label='Original data', markersize=5)
plt.plot(xx4, m4*xx4 + c4, 'r', label='Fitted line')

plt.subplot(1,7,5)
plt.plot(xx5, yy, 'o', label='Original data', markersize=5)
plt.plot(xx5, m5*xx5 + c5, 'r', label='Fitted line')

plt.subplot(1,7,6)
plt.plot(xx6, yy, 'o', label='Original data', markersize=5)
plt.plot(xx6, m6*xx6 + c6, 'r', label='Fitted line')

plt.subplot(1,7,7)
plt.plot(xx7, yy, 'o', label='Original data', markersize=5)
plt.plot(xx7, m7*xx7 + c7, 'r', label='Fitted line')

plt.legend()
plt.show()