"""
周志华的 西瓜数据 2.0 的显示
"""
from MLInAction.ca4 import loaddata, tree, treePlotter
from pprint import pprint

# 导入西瓜数据集
X, labels = loaddata.read2_0()
descTree = tree.createTree(X, labels)
pprint(descTree)
treePlotter.createPlot(descTree)