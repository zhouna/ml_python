#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 26 00:27:00 2017

@author: zz
"""
import pandas as pd
import numpy as np
import math

class Node:
    def __init__(self, label):
        self.label = label
        self.attribute = None
        self.children = None
    def setAttribute(self, attribute):
        self.attribute = attribute
    def setChildren(self, children):
        self.children = children
    def p(self):
        print 'label:', self.label, ', attribute:', self.attribute

def entropy(obj):
    # obj: pandas.Series
    N = float(obj.count())
    C = obj.value_counts().values
    P = C / N
    
    H = 0.0
    for p in P:
        H += -p*math.log(p,2)
    return H

def conditionEntropy(obj, attribute, clazz):
    """
    obj: Pandas.DataFrame
    attribute: string
    clazz: string
    """
    CC = obj[attribute].value_counts()
    index = CC.index
    counts = CC.values
    N = float(obj.index.size)
 
    H = 0.0
    for i in range(index.size):
        Hi = entropy(obj[obj[attribute] == index[i]][clazz])
        H += counts[i] / N * Hi
    return H

def ID3(obj, attributes, clazz, threshold):
    """
    obj: pandas.DataFrame
    attributes: pandas.Series
    clazz: string
    threshold: float
    """
    clazz_value_counts = obj[clazz].value_counts()
    label = clazz_value_counts.index[0]
    node = Node(label)
    
    if clazz_value_counts.size == 1 or attributes.size == 0:
        return node
    
    condition_entropy = np.zeros(attributes.size)
    h = entropy(obj[clazz])
    for i in range(attributes.size):
        condition_entropy[i] = conditionEntropy(obj, attributes[i], clazz)
    condition_entropy = h - condition_entropy
    
    index = condition_entropy.argmax()
    attr = attributes[index]
    node.setAttribute(attr)
    
    if condition_entropy.max < threshold:
        return node
    
    attr_value_counts = obj[attr].value_counts()
    children = np.empty([attr_value_counts.index.size], dtype=np.object)
    for i in range(attr_value_counts.size):
        attrs = attributes.drop(attr)
        children[i] = ID3(obj[obj[attr]==attr_value_counts.index[i]], attrs, clazz, threshold)
    
    node.setChildren(children)
    return node

df = pd.read_csv('贷款申请样本.txt', encoding='utf-8', index_col='ID')
tree = ID3(df, df.columns.drop('clazz'), 'clazz', 0)
print '-------'