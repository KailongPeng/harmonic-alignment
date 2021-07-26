# we use this script in the paper to make the corruption experiment.
from sklearn import datasets, neighbors
from scipy import stats
import numpy as np
import harmonicalignment
import harmonicalignment.math

from shutil import copyfileobj
from six.moves import urllib
import os


def knnclassifier(X, X_labels, Y, Y_labels, knn): #输出的是测试的准确率
    knn_op = neighbors.KNeighborsClassifier(knn)
    knn_op.fit(X, X_labels)
    return knn_op.score(Y, Y_labels)


np.random.seed(42)
digits = datasets.fetch_openml("mnist_784")
labels = digits["target"]
# imgs = digits["data"]
imgs = np.asarray(digits["data"])

n_samples = 1000
n_features = 784
n_iters = 1
n_percentages = 3
n_wavelets = 4

colreplace_probs = np.linspace(0, 1, n_percentages) if n_percentages > 1 else [1]
# scale of wavelets(eg n_filters) to use 要使用的小波规模（例如n_filters）。
wavelet_scales = [2, 8, 16, 64]
# kernel params 内核参数
knn_1 = 20
decay_1 = 20
pca_1 = 100
knn_2 = knn_1
decay_2 = decay_1
pca_2 = pca_1
# Z = transformed
knn_transform = 10
decay_transform = 10
pca_transform = None
# diffusion time for final embedding 最终嵌入的扩散时间
diffusion_t = 1

output = np.zeros((n_percentages, n_iters, n_wavelets, 2))
# store metrics in here 在这里存储指标


# 在一系列的概率分布[0. , 0.5, 1. ]中选一个，使用这一个比例从总共的 784 的n_features的列 不进行旋转； 当pct=0的时候完全不正交旋转，当pct=1的时候所有行正交旋转。
for p in range(n_percentages): 
    # build random matrix and replace prct of columns with I 构建随机矩阵，用I替换各列的prct
    pct = colreplace_probs[p] # array([0. , 0.5, 1. ]) p=0,1,2
    random_rotation = stats.ortho_group.rvs(n_features) # 返回一个随机的正交矩阵，从O(N) Haar分布（O(N)上唯一的均匀分布）抽取一个比例的列变成单位矩阵的对应列。
    # random orthogonal rotation 随机正交旋转
    colReplace = np.random.choice(
        n_features, np.floor(pct * n_features).astype(int), replace=False
    )
    random_rotation[:, colReplace] = np.eye(n_features)[:, colReplace] #只有 pct （百分值）的pixel的值没有被破坏。只有p%保留像素的损坏图像
    for iter_idx in range(n_iters):
        #  sample two sets of digits from MNIST 从MNIST中抽取两组数字
        sample_idx = np.random.choice(len(labels), n_samples * 2, replace=False)
        X1_idx = sample_idx[:n_samples]
        X2_idx = sample_idx[n_samples:]
        #  slice the digits 分割数字
        X1 = imgs[X1_idx, :]
        X2 = imgs[X2_idx, :]
        #  transform X2 转化X2
        X2_rotate = X2 @ random_rotation.T # 这里的random_rotation 当pct=1的时候完全就是单位矩阵；当pct=10时候完全就是正交矩阵。当pct=（0，1）的时候是一个非正交矩阵。
        import numpy.linalg as LA
        print(f"LA.norm(X2)={LA.norm(X2)}")
        print(f"LA.norm(X2_rotate)={LA.norm(X2_rotate)}")
        print(f"np.sum((X2_rotate != X2)*1)={np.sum((X2_rotate != X2)*1)}")
        X_combined = np.vstack([X1, X2_rotate])
        U_combined, S_combined = harmonicalignment.math.diffusionCoordinates(
            X_combined, decay_1, knn_1, pca_1
        ) # 根据原始数据，构建graph，然后形成SVD，输出SVD的输出，也就是三个矩阵
        # this is for evaluating unaligned data.  You can also plot this. 这是为评估不对齐的数据。 你也可以绘制这个。
        #  slice the labels 切分标签
        X1_labels = labels[X1_idx]
        X2_labels = labels[X2_idx]
        combined_labels = np.concatenate([X1_labels, X2_labels])
        #  run pca and classify 运行PCA并且分类
        DM_combined = U_combined @ np.diag(np.exp(-S_combined)) # 对于 SVD 的输出的乘积就是 PCA 的结果
        beforeprct = knnclassifier(
            DM_combined[:n_samples, :],
            X1_labels,
            DM_combined[n_samples:, :],
            X2_labels,
            5,
        ) # 对于 PCA 的结果进行knn分类，前1000个数据作为训练，后面的
        for scale_idx in range(n_wavelets):
            n_filters = wavelet_scales[scale_idx]
            align_op = harmonicalignment.HarmonicAlignment(
                n_filters,
                t=diffusion_t, # 1
                overlap=2,
                verbose=1,
                knn_X=knn_1, # 20
                knn_Y=knn_2, # 20
                knn_XY=knn_transform, # 10
                decay_X=decay_1, # 20
                decay_Y=decay_2, # 20
                decay_XY=decay_transform, # 10
                n_pca_X=pca_1, # 100
                n_pca_Y=pca_2, # 100
                n_pca_XY=pca_transform, # None
            )
            align_op.align(X1, X2_rotate)
            Z = align_op.diffusion_map()
            afterprct = knnclassifier(
                Z[:n_samples, :], X1_labels, Z[n_samples:, :], X2_labels, 5
            )
            output[p, iter_idx, scale_idx, 0] = beforeprct
            output[p, iter_idx, scale_idx, 1] = afterprct

print(output)


# Harmonic alignment

# Parameters
# ----------
# n_filters : int Number of wavelets
#     小波的数量
# overlap : float, optional (default: 2) Amount of overlap between wavelets
#     小波之间的重叠量
# t : int, optional (default: 1) Amount of diffusion
#     可选（默认：1） 扩散量
# knn : int, optional (default: 5) Default number of nearest neighbors
#     可选（默认：5）默认的近邻数量
# decay : float, optional (default: 20) Default value of alpha decay
#     α衰变的默认值
# n_pca : int, optional (default: 100) Default number of principal components on which to build graph. If 0, no PCA is performed.
#     建立图形的主成分的默认数量。如果是0，则不执行PCA。
# n_eigenvectors : int, optional (default: None) Number of eigenvectors of the normalized Laplacian on which to perform alignment. Defaults to all eigenvectors.
#     归一化Laplacian的特征向量的数量，用来进行对齐。默认为所有特征向量。
# n_jobs : int, optional (default: 1) Number of threads. -1 uses all available
#     (默认: 1) 线程数。-1：使用所有可用的
# verbose : int or bool, optional (default: 0) Verbosity of logging output. 0 is silent, 1 is standard, 2 is debug.
#     int或bool, 可选 (默认: 0) 日志输出的粗略程度。0是沉默，1是标准，2是调试。
# random_state : int or np.RandomState, optional (default: None) Sets the random seed
#     int 或 np.RandomState, 可选 (默认: None) 设置随机种子
# knn_{X,Y,XY} （也就是 knn_X knn_Y 以及 knn_XY）: int, optional (default: None) If not None, overrides `knn`
#     如果不是无，则覆盖`knn`。
# decay_{X,Y,XY} （也就是 decay_X decay_Y 以及 decay_XY）: int, optional (default: None) If not None, overrides `decay`
#     如果不是无，则覆盖`decay`。
# n_pca_{X,Y,XY} （也就是 n_pca_X n_pca_Y 以及 n_pca_XY）: int, optional (default: None) If not None, overrides `n_pca`
# 如果不是无，将覆盖`n_pca`。