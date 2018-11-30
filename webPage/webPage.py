#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 02:13:58 2017

@author: zz
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import Series

fig = plt.figure()
ax = fig.add_subplot(211)

data = pd.read_csv('time.txt', sep='\n', header=None)
data = Series(data[0])

data = data[data<300]

data=data.sort_values()
#data.to_csv('zz.txt', index=False)

t = data.values

ax.hist(x=t, bins=100, normed=True)

l = float(1)/data.mean()
p = l*np.power(math.e, -l*t)

ax.plot(t, p)
print data.describe()

fig.show()

x1 = data.quantile(0.25)
x3 = data.quantile(0.75)

print '中间1/2的均值： ',data[data>x1][data<x3].mean()
