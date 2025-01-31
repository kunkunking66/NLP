"""
import time

import numpy as np

_w = np.asarray([0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65])
_b = np.asarray([0.35, 0.65])

# 假定就一条样本
_x = np.asarray([5.0, 10.0])
_y = np.asarray([0.01, 0.99])

lr = 0.5


def w(i):
    # i下标完全按照ppt给定
    return _w[i - 1]


def b(i):
    return _b[i - 1]


def x(i):
    return _x[i - 1]


def y(i):
    return _y[i - 1]


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def set_w(i, gd):
    _w[i - 1] = _w[i - 1] - lr * gd


def training():
    # 1. FP前向过程 -- 计算前向预测结果+损失值
    h1 = sigmoid(z=w(1) * x(1) + w(2) * x(2) + b(1))
    h2 = sigmoid(z=w(3) * x(1) + w(4) * x(2) + b(1))
    h3 = sigmoid(z=w(5) * x(1) + w(6) * x(2) + b(1))
    o1 = sigmoid(z=w(7) * h1 + w(9) * h2 + w(11) * h3 + b(2))
    o2 = sigmoid(z=w(8) * h1 + w(10) * h2 + w(12) * h3 + b(2))
    loss = 0.5 * (y(1) - o1) ** 2 + 0.5 * (y(2) - o2) ** 2
    # print(h1, h2, h3)
    # print(o1, o2)
    # print(loss)

    # 2. BP反向过程 -- 基于loss求解梯度，然后更新参数
    t1 = (o1 - y(1)) * o1 * (1 - o1)  # loss对net_o1的导数
    t2 = (o2 - y(2)) * o2 * (1 - o2)  # loss对net_o2的导数

    set_w(1, gd=(t1 * w(7) + t2 * w(8)) * h1 * (1 - h1) * x(1))
    set_w(2, gd=(t1 * w(7) + t2 * w(8)) * h1 * (1 - h1) * x(2))
    set_w(3, gd=(t1 * w(9) + t2 * w(10)) * h2 * (1 - h2) * x(1))
    set_w(4, gd=(t1 * w(9) + t2 * w(10)) * h2 * (1 - h2) * x(2))
    set_w(5, gd=(t1 * w(11) + t2 * w(12)) * h3 * (1 - h3) * x(1))
    set_w(6, gd=(t1 * w(11) + t2 * w(12)) * h3 * (1 - h3) * x(2))

    set_w(7, gd=t1 * h1)
    set_w(9, gd=t1 * h2)
    set_w(11, gd=t1 * h3)
    set_w(8, gd=t2 * h1)
    set_w(10, gd=t2 * h2)
    set_w(12, gd=t2 * h3)

    return loss, o1, o2


if __name__ == '__main__':
    # (0.000159964002265472,   0.022968708310631587, 0.9776816961685182)
    # (0.00016006984718992034, 0.022970938311877513, 0.9776754532055207)
    r_ = training()
    print(_w)

    t = time.time()
    for j in range(1000):
        r_ = training()
        print(_w)
    print(f"总耗时:{time.time() - t}")
    print(_w)
    print(r_)
"""


import time

import numpy as np

_w = np.asarray([0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65])
_b = np.asarray([0.35, 0.65])

lr = 0.5


def w(i):
    # i下标完全按照ppt给定
    return _w[i - 1]


def b(i):
    return _b[i - 1]


def x(i):
    return _x[i - 1]


def y(i):
    return _y[i - 1]


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def set_w(i, gd):
    _w[i - 1] = _w[i - 1] - lr * gd


def training():
    # 1. FP前向过程 -- 计算前向预测结果+损失值
    h1 = sigmoid(z=w(1) * x(1) + w(2) * x(2) + b(1))
    h2 = sigmoid(z=w(3) * x(1) + w(4) * x(2) + b(1))
    h3 = sigmoid(z=w(5) * x(1) + w(6) * x(2) + b(1))
    o1 = sigmoid(z=w(7) * h1 + w(9) * h2 + w(11) * h3 + b(2))
    o2 = sigmoid(z=w(8) * h1 + w(10) * h2 + w(12) * h3 + b(2))
    loss = 0.5 * (y(1) - o1) ** 2 + 0.5 * (y(2) - o2) ** 2
    # print(h1, h2, h3)
    # print(o1, o2)
    # print(loss)

    # 2. BP反向过程 -- 基于loss求解梯度，然后更新参数
    t1 = (o1 - y(1)) * o1 * (1 - o1)  # loss对net_o1的导数
    t2 = (o2 - y(2)) * o2 * (1 - o2)  # loss对net_o2的导数

    set_w(1, gd=(t1 * w(7) + t2 * w(8)) * h1 * (1 - h1) * x(1))
    set_w(2, gd=(t1 * w(7) + t2 * w(8)) * h1 * (1 - h1) * x(2))
    set_w(3, gd=(t1 * w(9) + t2 * w(10)) * h2 * (1 - h2) * x(1))
    set_w(4, gd=(t1 * w(9) + t2 * w(10)) * h2 * (1 - h2) * x(2))
    set_w(5, gd=(t1 * w(11) + t2 * w(12)) * h3 * (1 - h3) * x(1))
    set_w(6, gd=(t1 * w(11) + t2 * w(12)) * h3 * (1 - h3) * x(2))

    set_w(7, gd=t1 * h1)
    set_w(9, gd=t1 * h2)
    set_w(11, gd=t1 * h3)
    set_w(8, gd=t2 * h1)
    set_w(10, gd=t2 * h2)
    set_w(12, gd=t2 * h3)

    return loss, o1, o2


if __name__ == '__main__':
    # 假定就5条样本
    _xs = [
        [5.0, 10.0],
        [2.0, 8.0],
        [3.0, 12.0],
        [3.0, 11.0],
        [16.0, 2.0]
    ]
    _ys = [
        [0.95, 0.12],
        [0.93, 0.01],
        [0.23, 0.77],
        [0.53, 0.45],
        [0.01, 0.99]
    ]

# # # # # 随机梯度下降 # # # # #
    for epoch in range(10000):
        # 假定一个批次就一个样本
        _rs = []
        for j in range(len(_xs)):
            _x = _xs[j]
            _y = _ys[j]
            _rs.append(training())
        if epoch == 0:
            print("=" * 100)
            print(_rs)
        elif epoch == 9999:
            print("=" * 100)
            print(_rs)


