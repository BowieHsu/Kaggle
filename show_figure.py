#encoding:utf-8
__author__ = 'bowiehsu'
import pandas as pd
import matplotlib.pyplot as plt

img = pd.read_csv('test.csv')
pi = img.values[100]
pix = []
for i in range(28):
    pix.append([])
    for j in range(28):
        pix[i].append(pi[i*28+j])

plt.imshow(pix)
plt.show()