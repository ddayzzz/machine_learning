# coding=utf-8
"""
读取数据
"""

def read(txtfile, splitch):
    """
    读取西瓜数据
    :param txtfile: 文本文件（第一行必须是标题）
    :param splitch: 分隔符
    :return:
    """
    labels = None
    data = []
    with open(txtfile, 'r', encoding='utf-8') as fp:
        s = fp.readline().strip()
        # 标题
        labels = s.split(splitch)
        while True:
            s = fp.readline()
            if not s:
                break
            data.append(s.strip().split(splitch)[1:])  # 跳过编号
    # 已经读取完毕了
    return data, labels


def read2_0():
    d, l = read('watermelon20.txt', ',')
    return d, l[1:]