"""
学习到了奇异值分解，学习一下如何处理图像
"""
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib as mpl
from pprint import pprint

def restore(sigma, u, v, p):
    """
    这个是复原的过程 U*Sigma*V. 其中 U \in R^{m*p} Sigma 是 diag(sigma_1,...,sigma_p) 按照其大小的排列顺序. V是 \in R^{p x m}
    :param sigma: 奇异值
    :param u: 左特征向量
    :param v:右特征向量
    :param p: 最后一个累加项的编号
    :return:
    """
    m = len(u)
    n = len(v)
    a = np.zeros((m, n))  # mxn的全零矩阵
    for k in range(p):
        uk = u[:, k].reshape(m, 1)  # m 行
        vk = v[k].reshape(1, n)  # n列
        a += sigma[k] * np.dot(uk, vk)  # 选择行列相乘
    # 去噪
    a = a.clip(0, 255)
    return np.rint(a).astype('uint8')  # 取上界


if __name__ == '__main__':
    A = Image.open(os.path.realpath('./Lenna_svd.png'))
    output_path = r'./svd_output'
    pprint(A)
    A = np.array(A)
    # 输出奇异值的数量
    p = 50
    u_r, sigma_r, v_r = np.linalg.svd(A[:,:,0])  # R分量
    u_g, sigma_g, v_g = np.linalg.svd(A[:, :, 1])  # G分量
    u_b, sigma_b, v_b = np.linalg.svd(A[:, :, 2])  # B分量
    plt.figure(figsize=(11,9), facecolor='w')
    mpl.rcParams['font.sans-serif'] = ['simHei']
    mpl.rcParams['axes.unicode_minus'] = False
    for k in range(1, p + 1):
        print(k)
        R = restore(sigma_r, u_r, v_r, k)
        G = restore(sigma_g, u_g, v_g, k)
        B = restore(sigma_b, u_b, v_b, k)
        I = np.stack((R, G, B), axis=2)
        Image.fromarray(I).save('%s\\svd_%d.png' % (output_path, k))  # 将输出的图片保存
        # 也可以Image.fromarray(I).save("svd_"+str(k)+".png")
        if k <= 12:  # 显示前12个
            plt.subplot(3, 4, k)
            plt.imshow(I)
            plt.axis('off')
            plt.title('奇异值个数：%d' % k)
    plt.suptitle('SVD与图像分解', fontsize=20)
    plt.tight_layout(0.3, rect=(0, 0, 1, 0.92))
    # plt.subplots_adjust(top=0.9)
    plt.show()