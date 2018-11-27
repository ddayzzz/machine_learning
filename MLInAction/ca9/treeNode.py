# coding=utf-8
"""
定义树的节点
"""
class TreeNode():

    def __init__(self, feat, val, right, left):
        """
        构造函数
        :param feat: 保存待切分的特征
        :param val: 保存带切分的特征值
        :param right: 右子树
        :param left: 左子树
        """
        self.featureToSplitOn = feat
        self.valueOfSplit = val
        self.right = right
        self.left = left