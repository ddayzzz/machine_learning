# coding=utf-8
# 图形绘制来自于《机器学习实践》
__doc__ = '这个用于绘制图形'
import matplotlib.pyplot as plt


decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")


def plotNode(nodetxt, centptr, parentptr, nodetype):
    createPlot.ax1.annotate(nodetxt, xy=parentptr, xycoords='axes fraction', xytext=centptr, textcoords='axes fraction',
                            va='center', ha="center", bbox=nodetype, arrowprops=arrow_args)


def createPlot():
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    createPlot.ax1 = plt.subplot(111, frameon=False)
    plotNode('决策节点', (0.5, 0.1), (0.1, 0.5), decisionNode)
    plotNode('叶子节点', (0.8, 0.1), (0.3, 0.8), leafNode)
    plt.show()


def getNumLeafs(mytree):
    """
    获取叶子的个数
    :param mytree:
    :return:
    """
    numLeafs = 0
    firstStr = tuple(mytree.keys())[0]  # 获取所有的子键
    secondDict = mytree[firstStr]
    for key in secondDict.keys():
        if isinstance(secondDict[key], dict):
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs





def getTreeDepth(mytree):
    """
    获取树的深度
    :param mytree: 树
    :return:
    """
    maxDepth = 0
    firstStr = tuple(mytree.keys())[0]
    secondDict = mytree[firstStr]
    for key in secondDict.keys():
        if isinstance(secondDict[key], dict):
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth


def retrievetree(i):
    listOfTrees = [{'no surfacing':{0: 'no',1:{'flippers': {0:'no',1:'yes'}}}},
                   {'no surfacing':{0:'no', 1:{'flippers':{0:{'head':{0:'no', 1:'yes'}}, 1:'no'}}}}]
    return listOfTrees[i]


def plotMidText(cntrPtr, parentPtr, txtString):
    xmid = (parentPtr[0] - cntrPtr[0]) / 2.0 + cntrPtr[0]
    ymid = (parentPtr[1] - cntrPtr[1]) / 2.0 + cntrPtr[1]
    createPlot.ax1.text(xmid, ymid, txtString)


def plotTree(mytree, parentpt, nodetxt):
    numLeafs = getNumLeafs(mytree)
    depth = getTreeDepth(mytree)
    firstStr = tuple(mytree.keys())[0]
    cntPtr = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW, plotTree.yOff)
    plotMidText(cntPtr, parentpt, nodetxt)
    plotNode(firstStr, cntPtr, parentpt, decisionNode)
    secondDict = mytree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
    for key in secondDict.keys():
        if isinstance(secondDict[key], dict):
            plotTree(secondDict[key], cntPtr, str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntPtr, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntPtr, str(key))
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD


def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5 / plotTree.totalW;
    plotTree.yOff = 1.0
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()

