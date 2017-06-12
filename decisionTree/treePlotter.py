#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  3 00:20:45 2017

@author: zz
"""

import matplotlib.pyplot as plt
import pandas as pd
import decisionTree as dt

def xy(tree, y):
    global n
    tree.y = y
    
    if tree.x != None:
        return tree.x
    
    if tree.children == None:
        x = n*x0
        n += 1
    else:
        xSum = 0
        k = len(tree.children)
        for node in tree.children.values():
            xSum += xy(node, y-y0)
        x = float(xSum) / k

    tree.x = x
    return x

xx = 0.05
yy = 0.08

def pp(tree, ax):
    if tree == None:
        return
    
    if tree.children != None:
        for key in tree.children.keys():
            ax.annotate(tree.attribute,
                        xy=(tree.children[key].x+xx, tree.children[key].y+yy),
                        xytext=(tree.x, tree.y),
                        xycoords='figure fraction',
                        textcoords='figure fraction',
                        arrowprops=dict(arrowstyle='->')
                           )
            ax.text((tree.children[key].x+tree.x)/2, (tree.children[key].y+tree.y)/2, key, transform=ax.transAxes)
            pp(tree.children[key], ax)
    else:
        ax.text(tree.x, tree.y, tree.label, transform=ax.transAxes)
        
        
df = pd.read_csv('贷款申请样本.txt', encoding='utf-8', index_col='ID')
tree = dt.decisionTree(df, df.columns.drop('clazz'), 'clazz', 0, method='c4.5')

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

h = dt.getTreeDepth(tree)
w = dt.getLeavesNum(tree)

y0 = float(1)/(h+1)
x0 = float(1)/(w+1)

n = 1

xy(tree, 1-y0)
pp(tree, ax)

plt.show()

