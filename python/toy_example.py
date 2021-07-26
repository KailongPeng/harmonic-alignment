import harmonicalignment
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn import datasets
import matplotlib
import matplotlib.cm as cmx
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import zscore

def scatter3d(x,y,z, cs, colorsMap='jet'):
    cm = plt.get_cmap(colorsMap)
    cNorm = matplotlib.colors.Normalize(vmin=min(cs), vmax=max(cs))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(x, y, z, c=scalarMap.to_rgba(cs))
    scalarMap.set_array(cs)
    fig.colorbar(scalarMap)
    plt.show()

def normalize(X): # 默认是对每一列单独标准化 mean=0 std=1
    _X=X.copy()
    _X = zscore(_X, axis=0)
    _X[np.isnan(_X)]=0
    return _X

data=sklearn.datasets.make_swiss_roll(n_samples=2000, noise=0.1)    
scatter3d(data[0][:,0],data[0][:,1],data[0][:,2],data[1])


# 分割数据成为x1 和 x2
x1=data[0][:1000,:]
label1=data[1][:1000]
x2=data[0][1000:,:]
label2=data[1][1000:]
x=data[0]
label=data[1]

# 分别作图验证
scatter3d(x1[:,0],x1[:,1],x1[:,2],label1)
scatter3d(x2[:,0],x2[:,1],x2[:,2],label2)

# 在align之前画出某一个feature的distribution，然后计算x1和x2 overlap的比例
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

_overlapRatio = overlapRatio(x1,x2,title='before alignment')

print(f"before alignment, overlapRatio={_overlapRatio}")

# 标准化，怀疑内置在harmonic 函数中，因此去掉归一化
# x1=normalize(x1)
# x2=normalize(x2)

# 运行协调对齐
# scale of wavelets(eg n_filters) to use 要使用的小波规模（例如n_filters）。
wavelet_scales = [2, 4, 8, 16, 64]
# wavelet_scales = [64, 128, 256]
# wavelet_scales = np.arange(64,128,8)

for n_filters in wavelet_scales:
    # align_op = harmonicalignment.HarmonicAlignment(n_filters=int(n_filters))
    align_op = harmonicalignment.HarmonicAlignment(
                int(n_filters),
                t=1, # 1
                overlap=4,
                verbose=0,
                knn_X=20, # 20
                knn_Y=20, # 20
                knn_XY=10, # 10
                decay_X=20, # 20
                decay_Y=20, # 20
                decay_XY=10, # 10
                n_pca_X=100, # 100
                n_pca_Y=100, # 100
                n_pca_XY=None, # None
            )
    align_op.align(x1, x2)
    XY_aligned = align_op.diffusion_map()

    xb1_aligned=XY_aligned[:1000]
    xb2_aligned=XY_aligned[1000:]

    # # 分别作图
    # plt.figure()
    # plt.scatter(xb1_aligned[:,0],xb1_aligned[:,1],c=label1)
    # plt.title(f'x1 aligned ; n_filters={n_filters}')
    # plt.figure()
    # plt.scatter(xb2_aligned[:,0],xb2_aligned[:,1],c=label2)
    # plt.title(f'x2 aligned ; n_filters={n_filters}')
    # plt.figure()
    # plt.scatter(XY_aligned[:,0],XY_aligned[:,1],c=label[:])
    # plt.title(f'x1x2 aligned ; n_filters={n_filters}')
    # # 发现效果非常差，因为原始数据是瑞士卷的数据的原始版x1和放大版x2
    # # 但是我得到的xb1_aligned 和 xb2_aligned 看到的效果是完全不重合

    _overlapRatio = overlapRatio(xb1_aligned,xb2_aligned,title=f'n_filters={n_filters}')
    print(f"n_filters={n_filters}, overlapRatio={_overlapRatio}")
