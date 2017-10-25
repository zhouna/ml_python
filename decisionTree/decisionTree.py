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
        self.x = None
        self.y = None
    def setAttribute(self, attribute):
        self.attribute = attribute
    def setChildren(self, children):
        self.children = children
    def p(self):
        print ('label:', self.label, ', attribute:', self.attribute)

#计算熵(以2为底)
def entropy(obj):
    # obj: pandas.Series
    N = float(obj.count())
    C = obj.value_counts().values
    P = C / N
    
    H = 0.0
    for p in P:
        H += -p*math.log(p,2)
    return H

#计算条件熵
def conditionEntropy(obj, attribute, clazz):
    """
    obj: Pandas.DataFrame
    attribute: string
    clazz: string
    """
    CC = obj[attribute].value_counts()   #属性的数量
    index = CC.index                     
    counts = CC.values                   #属性值
    N = float(obj.index.size)
 
    H = 0.0 # empirical entropy
    HA = 0.0 # 训练数据集D关于特征A的值的熵,经验条件熵
    for i in range(index.size):
        Hi = entropy(obj[obj[attribute] == index[i]][clazz])
        p = counts[i] / N
        H += p * Hi
        HA += -p * math.log(p, 2)
    return [H, HA]

#决策树
def decisionTree(obj, attributes, clazz, threshold, method='id3'):
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
    
    #判断条件熵的大小构建决策树
    condition_entropy = np.zeros(attributes.size)
    h = entropy(obj[clazz])
    for i in range(attributes.size):
        [condition_entropy[i], ha]= conditionEntropy(obj, attributes[i], clazz)
        condition_entropy[i] = h - condition_entropy[i]
        if method == 'c4.5':
            condition_entropy[i] = condition_entropy[i] / ha
    
    index = condition_entropy.argmax()
    attr = attributes[index]
    node.setAttribute(attr)
    
    if condition_entropy.max < threshold:
        return node
    
    attr_value_counts = obj[attr].value_counts()
    attrs = attributes.drop(attr)
    children = dict()
    for i in range(attr_value_counts.size):
        children[attr_value_counts.index[i]] = decisionTree(obj[obj[attr]==attr_value_counts.index[i]], attrs, clazz, threshold, method)
    
    node.setChildren(children)
    return node

df = pd.read_csv('贷款申请样本.txt', encoding='utf-8', index_col='ID')
tree = decisionTree(df, df.columns.drop('clazz'), 'clazz', 0)


"""
return tree depth
"""
def getTreeDepth(tree):
    if tree.children == None:
        return 1
    
    ds = [None]*len(tree.children)
    i = 0
    for node in tree.children.values():
        ds[i] = getTreeDepth(node)
        i = i + 1
    
    return max(ds)+1

"""
return the tree's leaves's number
"""
def getLeavesNum(tree):
    if tree.children == None:
        return 1
    
    num = 0
    for node in tree.children.values():
        num += getLeavesNum(node)

    return num
