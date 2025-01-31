import os
# 它提供了许多与操作系统交互的功能，比如文件路径操作
import numpy as np
import matplotlib.pyplot as plt


def f():
    print(f"当前文件夹:{os.getcwd()}")
    # 构造了一个文件路径，指向当前脚本所在目录下的datas文件夹中的image2.png文件
    path = os.path.join(os.path.dirname(__file__), "datas", "image4.png")
    # os.path.dirname(__file__)：获取当前执行文件的目录路径。
    # os.path.join(os.path.dirname(__file__), "datas", "image4.png")：将当前文件的目录路径与"datas"子目录和"image4.png"文件名拼接起来，形成一个完整的文件路径。
    print(path)
    print(os.path.abspath(path))  # 打印文件的绝对路径
    img = plt.imread(path)  # 使用matplotlib的imread函数读取图像文件
    print(img)
    print(type(img))
    print(img.shape)  # 打印图像数据的形状，通常是一个三元组，表示图像的高度、宽度和颜色通道数
    img1 = np.reshape(img, -1)  # 使用numpy的reshape函数将图像数据重塑为一维数组
    print(img1)
    print(img1.shape)


# if __name__ == '__main__':下面的代码块只有在文件被直接运行时才会执行。
# 如果文件被导入到另一个文件中，__name__变量会被设置为文件名（不包含.py扩展名），所以if __name__ == '__main__':下面的代码块不会执行。
if __name__ == '__main__':
    f()
