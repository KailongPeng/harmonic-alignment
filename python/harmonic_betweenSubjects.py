# import os
# import sys
# import csv
# import numpy as np
# import pandas as pd
# import nibabel as nib
# import datetime
# import matplotlib.pyplot as plt
# import pickle
# from sklearn.decomposition import PCA
# import harmonicalignment
# import phate
# def load_obj(name):
#     with open(name + '.pkl', 'rb') as f:
#         return pickle.load(f)

# def removeColumnWithNan(a): # 删除包含缺失值的列。
#     #print(~np.isnan(a).any(axis=0))
#     t=a[:, ~np.isnan(a).any(axis=0)]
#     # print(t)
#     return t

# class activation_PCA(): #使用PCA降维到1000维度，降维前归一化。
#     def fit(self, X_train=0, X_test=0, num_components=100):
#         n_samples, n_features = X_train.shape
#         num_components = min(n_samples, n_features) if num_components > min(n_samples, n_features) else num_components

#         # scale the data
#         # scaler = StandardScaler()
#         # scaler.fit(X_train)
#         # X_train = scaler.transform(X_train)
#         # X_test = scaler.transform(X_test)

#         self.model = PCA(num_components)
#         self.model.fit(X_train)
#         self.pcs_train = self.model.transform(X_train)  # principle components are the coefficients that transform original data into principal vectors space.
#         self.pcs_test = self.model.transform(X_test)

# def pca(X_train=train_sub1, X_test=test_sub1):
#     PCA_sub1 = activation_PCA()
#     PCA_sub1.fit(X_train=train_sub1, X_test=test_sub1, num_components=100)
#     training_sub1 = PCA_sub1.pcs_train
#     testing_sub1 = PCA_sub1.pcs_test
#     return training_sub1 , testing_sub1

# def PhateShow(X,label=[],title=''):
#     plt.figure()
#     phate_operator = phate.PHATE(verbose=0)
#     tree_phate = phate_operator.fit_transform(X)
#     if len(label)==0:
#         phate.plot.scatter2d(tree_phate,title=f"{title} knn={5} decay={40} t=auto")
#     else:
#         phate.plot.scatter2d(tree_phate,title=f"{title} knn={5} decay={40} t=auto",c=label)
#     return tree_phate

# def loadBold500SubjectBrainData(subject1):
#     # subject1='CSI1'
#     destDir = '/gpfs/milgram/scratch60/turk-browne/kp578/harmonic/brain'
#     subspace = False
#     destMeta = '{}/{}_meta.csv'.format(destDir, subject1)
#     destFeat = '{}/{}_feat.npy'.format(destDir, subject1) if subspace else '{}/{}_std.npy'.format(destDir, subject1)
#     y_sub1 = pd.read_csv(destMeta)
#     X_sub1 = np.load(destFeat)

#     X_sub1 = np.transpose(X_sub1,(3,0,1,2))

#     # 为了测试的时候节约内存，只使用前200个数据
#     y_sub1=y_sub1[:200]
#     X_sub1=X_sub1[:200]

#     # 将 N x voxel x voxel x voxel 的数据变成适应PCA的 n x voxel 的数据
#     X_sub1 = X_sub1.reshape(X_sub1.shape[0],-1)

#     # 删除包含缺失值的列。
#     X_sub1 = removeColumnWithNan(X_sub1)

#     # 根据session或者run来选择训练集和测试集
#     # trainingID = y_sub1['sess']!=y_sub1['sess'].iloc[-1]
#     # testingID = y_sub1['sess']==y_sub1['sess'].iloc[-1]
#     trainingID_sub1 = y_sub1['run']!=y_sub1['run'].iloc[-1]
#     testingID_sub1 = y_sub1['run']==y_sub1['run'].iloc[-1]

#     train_sub1 = X_sub1[trainingID_sub1]
#     train_label_sub1 = y_sub1[trainingID_sub1]
#     test_sub1 = X_sub1[testingID_sub1]
#     test_label_sub1 = y_sub1[testingID_sub1]

#     print(f'train_sub1.shape={train_sub1.shape}')
#     print(f'test_sub1.shape={test_sub1.shape}')

#     # PCA 降低维度到100维
#     training_sub1 , testing_sub1 = pca(X_train=train_sub1, X_test=test_sub1)
    
#     # 输出降维后的数据和标签信息
#     return training_sub1 , train_label_sub1, testing_sub1, test_label_sub1

# def harmonicBetweenSubjects(subject1='CSI1',subject2='CSI2'):
#     training_sub1 , train_label_sub1, testing_sub1, test_label_sub1 = loadBold500SubjectBrainData(subject1)
#     training_sub2 , train_label_sub2, testing_sub2, test_label_sub2 = loadBold500SubjectBrainData(subject2)

#     # 进行不同被试数据之间的harmonic alignment
#     x1,x2=training_sub1,training_sub2
#     n_filters = 8
#     align_op = harmonicalignment.HarmonicAlignment(
#                 int(n_filters),
#                 t=1, # 1
#                 overlap=n_filters,
#                 verbose=0,
#                 knn_X=20, # 20
#                 knn_Y=20, # 20
#                 knn_XY=10, # 10
#                 decay_X=20, # 20
#                 decay_Y=20, # 20
#                 decay_XY=10, # 10
#                 n_pca_X=100, # 100
#                 n_pca_Y=100, # 100
#                 n_pca_XY=None, # None
#             )
#     align_op.align(x1, x2)
#     XY_aligned = align_op.diffusion_map()

#     # 获取align完成之后的统一空间
#     xb1_aligned=XY_aligned[:training_sub1.shape[0]]
#     xb2_aligned=XY_aligned[training_sub1.shape[0]:]

#     label=[0]*training_sub1.shape[0]+[1]*training_sub1.shape[0]
#     _=PhateShow(XY_aligned,label=label,title=f'phate n_filters={n_filters}')

# subject1 = 'CSI2'
# subject2 = 'CSI3'
# harmonicBetweenSubjects(subject1=subject1,subject2=subject2)


# # subject1='CSI1'
# # subject2='CSI2'

# # destDir = '/gpfs/milgram/scratch60/turk-browne/kp578/harmonic/brain'
# # subspace = False

# # destMeta = '{}/{}_meta.csv'.format(destDir, subject1)
# # destFeat = '{}/{}_feat.npy'.format(destDir, subject1) if subspace else '{}/{}_std.npy'.format(destDir, subject1)
# # y_sub1 = pd.read_csv(destMeta)
# # X_sub1 = np.load(destFeat)

# # destMeta = '{}/{}_meta.csv'.format(destDir, subject2)
# # destFeat = '{}/{}_feat.npy'.format(destDir, subject2) if subspace else '{}/{}_std.npy'.format(destDir, subject2)
# # y_sub2 = pd.read_csv(destMeta)
# # X_sub2 = np.load(destFeat)

# # X_sub1 = np.transpose(X_sub1,(3,0,1,2))
# # X_sub2 = np.transpose(X_sub2,(3,0,1,2))


# # # 为了测试的时候节约内存，只使用前200个数据
# # y_sub1=y_sub1[:200]
# # X_sub1=X_sub1[:200]
# # y_sub2=y_sub2[:200]
# # X_sub2=X_sub2[:200]


# # # trainingID = y_sub1['sess']!=y_sub1['sess'].iloc[-1]
# # # testingID = y_sub1['sess']==y_sub1['sess'].iloc[-1]
# # trainingID_sub1 = y_sub1['run']!=y_sub1['run'].iloc[-1]
# # testingID_sub1 = y_sub1['run']==y_sub1['run'].iloc[-1]

# # trainingID_sub2 = y_sub2['run']!=y_sub2['run'].iloc[-1]
# # testingID_sub2 = y_sub2['run']==y_sub2['run'].iloc[-1]

# # X_sub1=X_sub1.reshape(X_sub1.shape[0],-1)
# # X_sub2=X_sub2.reshape(X_sub2.shape[0],-1)


# # X_sub1 = removeColumnWithNan(X_sub1)
# # X_sub2 = removeColumnWithNan(X_sub2)

# # train_sub1 = X_sub1[trainingID_sub1]
# # train_sub2 = X_sub2[trainingID_sub2]

# # test_sub1 = X_sub1[testingID_sub1]
# # test_sub2 = X_sub2[testingID_sub2]


# # print(f'sub2_train.shape={train_sub1.shape}')
# # print(f'sub2_test.shape={test_sub1.shape}')

# # training_sub1 , testing_sub1 = pca(X_train=train_sub1, X_test=test_sub1)
# # training_sub2 , testing_sub2 = pca(X_train=train_sub2, X_test=test_sub2)



