# import os
# import sys
# import csv
# import numpy as np
# import pandas as pd
# import nibabel as nib
# import datetime
# import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA
# import harmonicalignment
# import phate
# import scipy
# import pickle
# from tqdm import tqdm
# from scipy.stats import zscore
# # sys.path.append("/Users/kailong/Desktop/rtEnv/harmonic/harmonic-alignment/python/")
# # from get_brain_features_all_regions import removeColumnWithNan

# def removeColumnWithNan(a): # 删除包含缺失值的列。
#     #print(~np.isnan(a).any(axis=0))
#     t=a[:, ~np.isnan(a).any(axis=0)]
#     # print(t)
#     return t

# def kp_and(a,b):
#     return np.asarray(a) * np.asarray(b)

# def normalize(_X):
#     # _X=X.copy()
#     _X = zscore(_X, axis=0)
#     # _X[np.isnan(_X)]=0
#     return _X

# def zscoreEachRun(X_sub2, y_sub2):
#     if len(X_sub2.shape)>2:
#         X_sub2 = X_sub2.reshape(X_sub2.shape[0],-1)
#     X_sub2_normalized=np.zeros((2)) # 初始化标准化结果的容器
#     SessionList = np.unique(list(y_sub2["sess"]))
#     SessionList.sort()
#     for sess in tqdm(SessionList):
#         runList = np.unique(y_sub2[y_sub2['sess']==sess]['run'])
#         # print(runList)
#         for run in runList:
#             ID = kp_and(y_sub2['sess']==sess, y_sub2['run']==run)
#             # y_sub2[ID]
#             # print(X_sub2[ID].shape)
#             t = normalize(X_sub2[ID])
#             X_sub2_normalized = t if len(X_sub2_normalized.shape)<2 else np.concatenate((X_sub2_normalized,t),axis=0)
#     print(f"X_sub2_normalized.shape={X_sub2_normalized.shape}")
#     return X_sub2_normalized

# def findStrictCorrespondence(y_sub1="",X_sub2="",y_sub2=""):  # 根据给定的y_sub1，获取完全对应的y_sub2，以及完全对应的X_sub2，输出为X_sub2_strictAligned,y_sub2_strictAligned
#     X_sub2_strictAligned = np.zeros((2)) # 初始化标准化结果的容器
#     y_sub2_strictAligned = []
#     for img in y_sub1['item']:
#         # print(sum(1*(y_sub2['item']==img))) #每个对应的图片的数据的个数
#         # print(y_sub2[y_sub2['item']==img]['item'].iloc[0],img)
#         CorrespondingID = y_sub2['item']==img
#         t = y_sub2[CorrespondingID] # 如果有重复的图片，就删除重复的图片，重复图片的特征是’cat‘当中有’rep‘
#         notRepID = ['rep' not in i for i in t['cat']]
#         t = t[notRepID]
#         y_sub2_strictAligned = t if len(y_sub2_strictAligned)==0 else pd.concat([y_sub2_strictAligned,t])
        
#         t = X_sub2[CorrespondingID][notRepID]
#         X_sub2_strictAligned = t if len(X_sub2_strictAligned.shape)<2 else np.concatenate((X_sub2_strictAligned,t),axis=0)
        
#     print(f"X_sub2_strictAligned.shape={X_sub2_strictAligned.shape}")
#     return X_sub2_strictAligned,y_sub2_strictAligned

# def loadBold500SubjectBrainData_strict_align(subject1='CSI1', subject2='CSI2', numberOfDatapoints=-1):
#     # numberOfDatapoints：为了测试的时候节约内存，只使用前 numberOfDatapoints 个数据
#     destDir = '/gpfs/milgram/scratch60/turk-browne/kp578/harmonic/brain'

#     # 如果已经有现成的可输出的数据，就不需要再跑了，直接加载即可
#     if os.path.exists(f"{destDir}/loadBold500SubjectBrainData_strictAlign_sub-{subject1}_numberOfDatapoints-{numberOfDatapoints}.pkl"):
#         print(f"loading {destDir}/loadBold500SubjectBrainData_strictAlign_sub-{subject1}_numberOfDatapoints-{numberOfDatapoints}.pkl")
#         [training_sub1, train_label_sub1, testing_sub1, test_label_sub1] = load_obj(f"{destDir}/loadBold500SubjectBrainData_strictAlign_sub-{subject1}_numberOfDatapoints-{numberOfDatapoints}")
#     else:
#         subspace = False
#         destMeta = '{}/{}_meta.csv'.format(destDir, subject1)
#         destFeat = '{}/{}_feat.npy'.format(destDir, subject1) if subspace else '{}/{}_std.npy'.format(destDir, subject1)
#         y_sub1 = pd.read_csv(destMeta)
        
#         zscored = f'{destDir}/{subject1}_zscoredEachRun.npy'
#         if os.path.exists(zscored):
#             X_sub1 =  np.load(zscored)
#         else:
#             X_sub1 = np.load(destFeat)
#             X_sub1 = np.transpose(X_sub1,(3,0,1,2))

#             X_sub1 = zscoreEachRun(X_sub1,y_sub1)
#             print(f"X_sub1.shape after zscoreEachRun = {X_sub1.shape}")
#             np.save(zscored,X_sub1)

#         # # 为了测试的时候节约内存，只使用前200 numberOfDatapoints 个数据
#         if numberOfDatapoints == -1:
#             pass
#         else:
#             y_sub1=y_sub1[:numberOfDatapoints]
#             X_sub1=X_sub1[:numberOfDatapoints]

#         # 将 N x voxel x voxel x voxel 的数据变成适应PCA的 n x voxel 的数据
#         if len(X_sub1.shape)>2:
#             X_sub1 = X_sub1.reshape(X_sub1.shape[0],-1)

#         # 删除包含缺失值的列。
#         X_sub1 = removeColumnWithNan(X_sub1)

#         # 根据session或者run来选择训练集和测试集
#         trainingID_sub1 = y_sub1['sess']!=y_sub1['sess'].iloc[-1]
#         testingID_sub1 = y_sub1['sess']==y_sub1['sess'].iloc[-1]
#         # trainingID_sub1 = y_sub1['run']!=y_sub1['run'].iloc[-1]
#         # testingID_sub1 = y_sub1['run']==y_sub1['run'].iloc[-1]

#         train_sub1 = X_sub1[trainingID_sub1]
#         train_label_sub1 = y_sub1[trainingID_sub1]
#         test_sub1 = X_sub1[testingID_sub1]
#         test_label_sub1 = y_sub1[testingID_sub1]

#         print(f'train_sub1.shape={train_sub1.shape}')
#         print(f'test_sub1.shape={test_sub1.shape}')

#         # PCA 降低维度到100维
#         training_sub1 , testing_sub1 = pca(X_train=train_sub1, X_test=test_sub1)
        
#         # 保存pca后的数据
#         save_obj([training_sub1 , train_label_sub1, testing_sub1, test_label_sub1], f"{destDir}/loadBold500SubjectBrainData_sub_{subject1}_numberOfDatapoints_{numberOfDatapoints}")

#     # 如果已经有现成的可输出的数据，就不需要再跑了，直接加载即可
#     if os.path.exists(f"{destDir}/loadBold500SubjectBrainData_strictAlign_sub-{subject2}_numberOfDatapoints-{numberOfDatapoints}.pkl"):
#         print(f"loading {destDir}/loadBold500SubjectBrainData_strictAlign_sub-{subject2}_numberOfDatapoints-{numberOfDatapoints}.pkl")
#         [training_sub2, train_label_sub2, testing_sub2, test_label_sub2] = load_obj(f"{destDir}/loadBold500SubjectBrainData_strictAlign_sub-{subject2}_numberOfDatapoints-{numberOfDatapoints}")
#     else:
#         subspace = False
#         destMeta = '{}/{}_meta.csv'.format(destDir, subject2)
#         destFeat = '{}/{}_feat.npy'.format(destDir, subject2) if subspace else '{}/{}_std.npy'.format(destDir, subject2)
#         y_sub2 = pd.read_csv(destMeta)

#         zscored = f'{destDir}/{subject2}_zscoredEachRun.npy'
#         if os.path.exists(zscored):
#             X_sub2 =  np.load(zscored)
#         else:
#             X_sub2 = np.load(destFeat)
#             X_sub2 = np.transpose(X_sub2,(3,0,1,2))

#             X_sub2 = zscoreEachRun(X_sub2,y_sub2)
#             print(f"X_sub2.shape after zscoreEachRun = {X_sub2.shape}")
#             np.save(zscored,X_sub2)

#         # 根据y_sub1获取完全对应的大脑数据
#         X_sub2, y_sub2 = findStrictCorrespondence(y_sub1="",X_sub2="",y_sub2="")
#         print(f"X_sub2.shape={X_sub2.shape}")

#         # 为了测试的时候节约内存，只使用前200 numberOfDatapoints 个数据
#         if numberOfDatapoints == -1:
#             pass
#         else:
#             assert X_sub2.shape[0]==numberOfDatapoints
#             y_sub2=y_sub2[:numberOfDatapoints]
#             X_sub2=X_sub2[:numberOfDatapoints]

#         # 将 N x voxel x voxel x voxel 的数据变成适应PCA的 n x voxel 的数据
#         if len(X_sub2.shape)>2:
#             X_sub2 = X_sub2.reshape(X_sub2.shape[0],-1)

#         # 删除包含缺失值的列。
#         X_sub2 = removeColumnWithNan(X_sub2)

#         # 根据session或者run来选择训练集和测试集
#         trainingID_sub2 = y_sub2['sess']!=y_sub2['sess'].iloc[-1]
#         testingID_sub2 = y_sub2['sess']==y_sub2['sess'].iloc[-1]
#         # trainingID_sub2 = y_sub2['run']!=y_sub2['run'].iloc[-1]
#         # testingID_sub2 = y_sub2['run']==y_sub2['run'].iloc[-1]

#         train_sub2 = X_sub2[trainingID_sub2]
#         train_label_sub2 = y_sub2[trainingID_sub2]
#         test_sub2 = X_sub2[testingID_sub2]
#         test_label_sub2 = y_sub2[testingID_sub2]

#         print(f'train_sub2.shape={train_sub2.shape}')
#         print(f'test_sub2.shape={test_sub2.shape}')

#         # PCA 降低维度到100维
#         training_sub2 , testing_sub2 = pca(X_train=train_sub2, X_test=test_sub2)
        
#         # 保存pca后的数据
#         save_obj([training_sub2 , train_label_sub2, testing_sub2, test_label_sub2], f"{destDir}/loadBold500SubjectBrainData_sub_{subject2}_numberOfDatapoints_{numberOfDatapoints}")


#     # 输出降维后的数据和标签信息
#     return [training_sub1 , train_label_sub1, testing_sub1, test_label_sub1],[training_sub2 , train_label_sub2, testing_sub2, test_label_sub2]