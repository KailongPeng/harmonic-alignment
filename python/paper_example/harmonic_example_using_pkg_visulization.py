# we use this script in the paper to make the corruption experiment.
from sklearn import datasets, neighbors
from scipy import stats
import numpy as np
import harmonicalignment
import harmonicalignment.math

from shutil import copyfileobj
from six.moves import urllib
import os
import matplotlib.pyplot as plt

def knnclassifier(X, X_labels, Y, Y_labels, knn): #输出的是测试的准确率
    knn_op = neighbors.KNeighborsClassifier(knn)
    knn_op.fit(X, X_labels)
    return knn_op.score(Y, Y_labels)
def overlapRatio(x1,x2,title='before alignment',nthFeature=0):
    plt.figure()
    a = x1[:,nthFeature]
    b = x2[:,nthFeature]
    _max=max(max(a),max(b))
    _min=min(min(a),min(b))
    bins = np.arange(_min,_max,(_max-_min)/40)
    hist1=plt.hist(a,bins=bins,fc=(0, 0, 1, 0.5)) #np.arange(-2,2,0.1)
    # plt.figure()
    hist2=plt.hist(b,bins=bins,fc=(1, 0, 0, 0.5))
    plt.title(f"{title}, x1 x2 dimension 0 distribution")

    t = np.concatenate([np.expand_dims(hist1[0], axis=0) , np.expand_dims(hist2[0], axis=0)],axis=0)
    _overlapRatio = np.sum(np.min(t,axis=0)) / np.sum(hist1[0])
    return _overlapRatio

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

p = 1 #for p in range(n_percentages): 
# build random matrix and replace prct of columns with I 构建随机矩阵，用I替换各列的prct
pct = colreplace_probs[p] # array([0. , 0.5, 1. ]) p=0,1,2
random_rotation = stats.ortho_group.rvs(n_features) # 返回一个随机的正交矩阵，从O(N) Haar分布（O(N)上唯一的均匀分布）抽取一个比例的列变成单位矩阵的对应列。
# random orthogonal rotation 随机正交旋转
colReplace = np.random.choice(
    n_features, np.floor(pct * n_features).astype(int), replace=False
)
random_rotation[:, colReplace] = np.eye(n_features)[:, colReplace] #只有 pct （百分值）的pixel的值没有被破坏。只有p%保留像素的损坏图像


iter_idx=0 # for iter_idx in range(n_iters):
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
# for scale_idx in range(n_wavelets):
n_filters = 2 #n_filters = wavelet_scales[scale_idx]
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
xb1_aligned = Z[:n_samples, :]
xb2_aligned = Z[n_samples:, :]
print(f"beforeprct={beforeprct}")
print(f"afterprct={afterprct}")
_overlapRatio = overlapRatio(xb1_aligned,xb2_aligned,title=f'n_filters={n_filters}')
print(f"n_filters={n_filters}, overlapRatio={_overlapRatio}")
# output[p, iter_idx, scale_idx, 0] = beforeprct
# output[p, iter_idx, scale_idx, 1] = afterprct

# print(output)


# 展示align前后的图像
# align之前的图像
plt.figure()
plt.imshow(X1[0].reshape(28,-1))
plt.title("before alignment x1")

plt.figure()
plt.imshow(X2_rotate[0].reshape(28,-1))
plt.title("before alignment X2_rotate")

plt.figure()
plt.imshow(Z[0,:784].reshape(28,-1))
plt.title("after alignment x1")

plt.figure()
plt.imshow(Z[n_samples,:784].reshape(28,-1))
plt.title("after alignment X2_rotate")

