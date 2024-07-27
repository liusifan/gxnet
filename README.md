# gxnet

一个玩具级的神经网络训练框架，用于自学神经网络基础知识

[![Build Status](https://github.com/liusifan/gxnet/actions/workflows/c-cpp.yml/badge.svg)](https://github.com/liusifan/gxnet/actions?query=workflow:ci)

编译和运行
--------
可以参考项目自带的 github workflow 脚本 [c-cpp workflow](.github/workflows/c-cpp.yml)

或者按照以下步骤执行
```
git clone git@github.com:liusifan/gxnet.git
cd gxnet/gxnet
make
pip install Pillow
pip install numpy
sh launch_mnist.sh
```

开发过程
-------
用 C++ 从零开始实现一个玩具级的神经网络训练框架

这个项目的想法来自于 [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) 这本书。书上以经典的 MNIST 数据集作为目标，用不到 100 行的 Python 代码实现了一个可以运行的神经网络，并且在测试集上可以达到 95% 以上的准确率，非常令人惊讶。

这本书在 2017 年的时候就看过，最近偶然又翻到了这本书，就冒出用 C++ 从零开始实现一个玩具级神经网络训练框架的想法，主要的目的是为了更深入地理解神经网络。最近用了几个周末的时间，算是完成了目标。下面按顺序记录一下开发过程和遇到的问题。

## 按自己的理解实现初始版本
为了更好地理解神经网络训练的每一步，也为了真正地从零开始实现，就没有像书上那样引入矩阵算法库，而是用最基础的 C++ for 循环在做向量乘法，因此 C++ 代码和书上的 Python 代码不是一一对应。一开始不知道怎么从单元测试的角度去验证实现的正确性，就投机取巧直接拿 [seeds dataset](https://www.kaggle.com/datasets/rwzhang/seeds-dataset) 这个数据集做 demo，结果碰了一鼻子灰；因为训练之后相比训练之前，在测试集上的准确率没有提升，有时还下降。这个时候就很懵逼了，因为分不清楚是自己的训练框架写的有问题，还是数据处理有问题，还是超参数有问题，反正就是一团浆糊。

## 寻找单元测试的方法验证实现的正确性
在投机取巧浪费了一天之后，终于静下心来调研如何从单元测试的角度去验证实现的正确性，最终找到了这篇文章 [A worked example of backpropagation](https://alexander-schiendorfer.github.io/2020/02/24/a-worked-example-of-backprop.html)。这篇文章挽救了我的周末，对着这篇文章给的例子，实现了同样功能的代码，然后跟文章里面给出的每一步的计算结果做对比，从而验证了训练框架实现的正确性。

## 把神经网络技术发展早期存在的经典的坑都踩了一遍
这里的主要问题是看书不仔细，书上其实在关键章节都有提示，但看书的时候没有看进去。在确认训练框架的实现没问题之后，seeds 的 demo 仍然没有进展；在做了好多无效的尝试之后，忽然想起来，这个数据集的数据是按 label 的顺序排放的，在训练的时候，应该要把数据打散。果然，在加入 shuffle 功能之后，这个 demo 总算跑成功了。

然后就开始搞 MNIST 数据集。一开始以为 seeds demo 能跑成功，MNIST 也自然不在话下，哪知一上来也是啪啪打脸，也是在训练之后在测试集上的准确率没有提升，反而还下降。再次进入懵逼状态。这里整整耗了一个周末没有任何进展，一度想暂时放弃了。中间再次怀疑训练框架实现的正确性，想拿书上的 Python 代码和 C++ 代码用同样的 weights 和 biases 初始化，再去对比中间结果，不过不熟悉 Python 的矩阵库，最终作罢。在做了好多无效的尝试之后，重新回去看书，然后看到书中对于 weights 和 biases 的初始化提到用了均值为 0，标准差为 1 的高斯分布的随机数；在修改几行代码之后，终于把 MNIST 跑通了。

## 努力通过 UAT(User Acceptance Test) 测试
在 MNIST 测试集上达到 95% 准确率之后，就想拿现写的数字来试试。请家里的小朋友帮忙写了一些数字（[uat/ian](gxnet/uat/ian)），小朋友很有创意，坚持要用彩色笔来写，而且尽量每个数字用不同的颜色。模型准确率到了 95%，现写的数字也很工整，本以为 UAT 测试应该手到擒来，哪知道准确率居然一半都不到。用几个周末尝试了几种数据增强方案（把训练数据做图片居中和旋转（[trans_mnist.py](gxnet/trans_mnist.py)）），还是不能全部识别出来。最后在这个周末向一个计算机视觉方面的专家请教之后，终于发现问题就出在这个彩色笔上面，:(。

原因在于：1）这里使用的是一个简单的全连接神经网络，这个神经网络本质上并不能识别图片的形状，它其实只是对输入数据的统计特征进行识别；2)MNIST 数据集是黑白图片，而且非常黑白分明；而之前对彩色数字图片的处理，只是简单地把图片变成灰度图，不够黑白分明，生成的灰度图和 MNIST 的数据集不是同样的数据分布，相当于违反了这个模型的假设，因此识别准确率就低。知道问题之后，解决起来就比较简单了，在对 UAT 图片预处理的时候，对于非白色部分一律置成黑色（[conv2mnist.py](gxnet/conv2mnist.py)），加了这个之后彩色图片就全部识别出来了。

## 优化训练耗时
在前面几个阶段，每次调试需要的时间不需要很长；即使在第三阶段用了 6 万张 MNIST 图片做训练，每个 epoch 需要 76 秒，勉强可以接受。但在第四阶段，为了能通过 UAT 测试，做了数据增强，把 6 万张图片，扩增成 22 万张（首先把图片做 ( -15, 15 ) 角度的随机旋转生成 6 万张；其次对一共 12 万张图片做居中处理，但有些图片已经居中，因此最终生成新的 10 万张居中的图片），每个 epoch 的时间上涨到 860 秒，这就太影响效率了。因此在通过 UAT 的测试案例之后，就想把代码的执行性能提升一下。第一步是简单地在编译参数加上 -O3，没想到有奇效，每个 epoch 的耗时下降到了 58 秒，看来 -O3 对于计算密集型代码有奇效。第二步用苹果 xcode 的 cpu profiler 做分析，看到的结果是在最内层计算每条训练数据梯度的时候，有一个用于临时保存梯度的 matrix 被反复创建，占用了程序执行时间的一半；找到问题之后，改起来就不难了，把这个 matrix，还有连带的其他几个类似的 matrix/vector 一起移到所有循环之外进行初始化，相应用到的地方做一些小的调整就可以了；改完之后，每个 epoch 的耗时下降到 30 秒。这个耗时目前可以接受。

还有一点，就是用 [std::valarray](https://en.cppreference.com/w/cpp/numeric/valarray) 替代了 std::vector，在部分合适的场景下，用 valarray 的批量计算能力可以提升性能。在开始下面的工作之前，对代码打了 v0.1 的 tag 。

## 实现卷积层和池化层
想进一步尝试 [EMNIST](https://www.nist.gov/itl/products-and-services/emnist-dataset) 数据集，里面包含了英文字母和数字，只是用全连接层效果很差，因此想用卷积神经网络来实现。实现过程中，主要的问题也是在验证实现的正确性上面，这次使用了 [pytorch](cnn/testconv.py) 来作为参考对照。另外的问题就是需要一些技巧来简化代码实现，一个是 ActFunc/Layer/Network 的分层设计，一个是借鉴 C++23 里面的 [std::mdspan](https://en.cppreference.com/w/cpp/container/mdspan) 把一维数组映射为多维数组（没有直接使用 mdspan，是希望本项目保持用 C++11）。

[to be continued]

