import os
import sys
import csv
import numpy as np
import pandas as pd
import nibabel as nib
import datetime
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import harmonicalignment
import phate
import scipy
# import ray
# ray.init()
import pickle
def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def zscore_data(data):
    return (data-np.expand_dims(np.mean(data,3), 3))/np.expand_dims(np.std(data,3),3)

def load_data(projDir, subject, sess, run, zscore=True, native=True):
    if native:
        imloc = '{}/derivatives/fmriprep/sub-{}/{}/func'.format(projDir, subject, sess)
        imfile = 'sub-{}_{}_task-5000scenes_run-{}_bold_space-T1w_preproc.nii.gz'.format(subject, sess, run)
    else:
        imloc = '{}/standard/sub-{}/{}/func'.format(projDir, subject, sess)
        imfile = 'sub-{}_{}_task-5000scenes_run-{}_standard.nii.gz'.format(subject, sess, run)
    Im = nib.load('{}/{}'.format(imloc, imfile))
    dat = np.array(Im.get_fdata(), dtype=np.float64)
    if zscore:
        dat = zscore_data(dat)
    return dat

# 将数据从原始的bold5000变成特征阵列和元数据 TURN DATA FROM ORIGINAL BOLD5000 TO FEATURE ARRAYS AND METADATA

subspace = False

# 大脑数据的位置 where is bold5000
projDir = '/gpfs/milgram/data/bold5000'
# 保存输出变量的文件夹位置 where should output go
destDir = '/gpfs/milgram/scratch60/turk-browne/kp578/harmonic/brain'
subjects = ['CSI1', 'CSI2', 'CSI3', 'CSI4']

# 获得所有的图像对应的大脑的活动数据 Get all brain features after image onset 
# @ray.remote
def get_brain_features_for_subject(subject):
    thisSub = pd.DataFrame(columns=['subject', 'sess', 'run', 'item', 'cat', 'onsetS', 'critTR'])
    destMeta = '{}/{}_meta.csv'.format(destDir, subject)
    destFeat = '{}/{}_feat.npy'.format(destDir, subject) if subspace else '{}/{}_std.npy'.format(destDir, subject)
    sessions = [sess for sess in os.listdir('{}/sub-{}'.format(projDir, subject)) if os.path.isdir('{}/sub-{}/{}'.format(projDir, subject, sess))]
    sessions.sort()
    subFeatures = []
    for snum, sess in enumerate(sessions[:]):
        thisSess_standard = '{}/standard/sub-{}/{}/func'.format(projDir, subject, sess)
        thisSess = '{}/sub-{}/{}/func'.format(projDir, subject, sess)

        print("thisSess", thisSess_standard)
        if not os.path.isdir(thisSess_standard):
            break
        runs = np.unique([run.split('run-')[-1].split('_')[0] for run in os.listdir(thisSess_standard) if 'task-5000scenes' in run.split('_')])
        runs.sort()
        for rnum, run in enumerate(runs[:]):
            print('compiling subject {}, {}, run {} -- {}'.format(subject, sess, run, datetime.datetime.now()))
            runDat = pd.read_table('{}/sub-{}_{}_task-5000scenes_run-{}_events.tsv'.format(thisSess, subject, sess, run))
            runDat = runDat.rename(columns={"onset": "onsetS", "ImgName": "item", "ImgType": "cat"})
            runDat['subject'] = subject
            runDat['sess'] = sess
            runDat['run'] = run
            critTR = np.around(runDat['onsetS'],0)/2 + 2

            runDat['critTR'] = critTR.astype(int)
            thisSub = pd.concat([thisSub, runDat], join="inner")
            dat = load_data(projDir, subject, sess, run, native=subspace, zscore = True)

            # EXPERIMENTAL TAKE THIS OUT
            critTR = np.minimum(critTR, dat.shape[3] - 1)
            print("critTR", runDat['onsetS'][:5], critTR[:5])

            theseFeatures = dat[:,:,:,critTR.astype(int)]
            theseFeatures = zscore_data(theseFeatures)
            subFeatures.append(theseFeatures) # = theseFeatures if rnum == 0 and snum == 0 else np.concatenate((subFeatures, theseFeatures), 3)

    thisSub.to_csv(destMeta)
    np.save(destFeat, np.concatenate(subFeatures, 3))

# for subject in subjects:
#     get_brain_features_for_subject.remote(subject)

def removeColumnWithNan(a): # 删除包含缺失值的列。
    #print(~np.isnan(a).any(axis=0))
    t=a[:, ~np.isnan(a).any(axis=0)]
    # print(t)
    return t

class activation_PCA(): #使用PCA降维到1000维度，降维前归一化。
    def fit(self, X_train=0, X_test=0, num_components=100):
        n_samples, n_features = X_train.shape
        num_components = min(n_samples, n_features) if num_components > min(n_samples, n_features) else num_components

        # scale the data
        # scaler = StandardScaler()
        # scaler.fit(X_train)
        # X_train = scaler.transform(X_train)
        # X_test = scaler.transform(X_test)

        self.model = PCA(num_components)
        self.model.fit(X_train)
        self.pcs_train = self.model.transform(X_train)  # principle components are the coefficients that transform original data into principal vectors space.
        self.pcs_test = self.model.transform(X_test)

def pca(X_train=None, X_test=None):
    PCA_sub1 = activation_PCA()
    PCA_sub1.fit(X_train=X_train, X_test=X_test, num_components=100)
    training_sub1 = PCA_sub1.pcs_train
    testing_sub1 = PCA_sub1.pcs_test
    return training_sub1 , testing_sub1

def PhateShow(X,label=[],title=''):
    plt.figure()
    phate_operator = phate.PHATE(verbose=0)
    tree_phate = phate_operator.fit_transform(X)
    if len(label)==0:
        phate.plot.scatter2d(tree_phate,title=f"{title} knn={5} decay={40} t=auto")
    else:
        phate.plot.scatter2d(tree_phate,title=f"{title} knn={5} decay={40} t=auto",c=label)
    return tree_phate

def loadBold500SubjectBrainData(subject1):
    destDir = '/gpfs/milgram/scratch60/turk-browne/kp578/harmonic/brain'
    if os.path.exists(f"{destDir}/loadBold500SubjectBrainData_sub_{subject1}.pkl"):
        [training_sub1 , train_label_sub1, testing_sub1, test_label_sub1] = load_obj(f"{destDir}/loadBold500SubjectBrainData_sub_{subject1}")
    else:
        subspace = False
        destMeta = '{}/{}_meta.csv'.format(destDir, subject1)
        destFeat = '{}/{}_feat.npy'.format(destDir, subject1) if subspace else '{}/{}_std.npy'.format(destDir, subject1)
        y_sub1 = pd.read_csv(destMeta)
        X_sub1 = np.load(destFeat)

        X_sub1 = np.transpose(X_sub1,(3,0,1,2))

        # # 为了测试的时候节约内存，只使用前200个数据
        # y_sub1=y_sub1[:200]
        # X_sub1=X_sub1[:200]

        # 将 N x voxel x voxel x voxel 的数据变成适应PCA的 n x voxel 的数据
        X_sub1 = X_sub1.reshape(X_sub1.shape[0],-1)

        # 删除包含缺失值的列。
        X_sub1 = removeColumnWithNan(X_sub1)

        # 根据session或者run来选择训练集和测试集
        trainingID_sub1 = y_sub1['sess']!=y_sub1['sess'].iloc[-1]
        testingID_sub1 = y_sub1['sess']==y_sub1['sess'].iloc[-1]
        # trainingID_sub1 = y_sub1['run']!=y_sub1['run'].iloc[-1]
        # testingID_sub1 = y_sub1['run']==y_sub1['run'].iloc[-1]

        train_sub1 = X_sub1[trainingID_sub1]
        train_label_sub1 = y_sub1[trainingID_sub1]
        test_sub1 = X_sub1[testingID_sub1]
        test_label_sub1 = y_sub1[testingID_sub1]

        print(f'train_sub1.shape={train_sub1.shape}')
        print(f'test_sub1.shape={test_sub1.shape}')

        # PCA 降低维度到100维
        training_sub1 , testing_sub1 = pca(X_train=train_sub1, X_test=test_sub1)
        
        # 保存pca后的数据
        save_obj([training_sub1 , train_label_sub1, testing_sub1, test_label_sub1], f"{destDir}/loadBold500SubjectBrainData_sub_{subject1}")

    # 输出降维后的数据和标签信息
    return training_sub1 , train_label_sub1, testing_sub1, test_label_sub1

def harmonicBetweenSubjects(subject1='CSI1',subject2='CSI2'):
    training_sub1 , train_label_sub1, testing_sub1, test_label_sub1 = loadBold500SubjectBrainData(subject1)
    training_sub2 , train_label_sub2, testing_sub2, test_label_sub2 = loadBold500SubjectBrainData(subject2)

    # 进行不同被试数据之间的harmonic alignment
    x1,x2=training_sub1,training_sub2
    n_filters = 8
    align_op = harmonicalignment.HarmonicAlignment(
                int(n_filters), # 小波的数量
                t=1, # 1 # 扩散量
                overlap=n_filters, # 小波之间的重叠量
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

    # 获取align完成之后的统一空间
    xb1_aligned=XY_aligned[:training_sub1.shape[0]]
    xb2_aligned=XY_aligned[training_sub1.shape[0]:]

    alignmentMetric = getAlignmentMetric(xb1_aligned, xb2_aligned)
    print(f"alignmentMetric={alignmentMetric}")

    label=[0]*training_sub1.shape[0]+[1]*training_sub1.shape[0]
    _=PhateShow(XY_aligned,label=label,title=f'phate n_filters={n_filters}')

subject1 = 'CSI2'
subject2 = 'CSI3'
harmonicBetweenSubjects(subject1=subject1,subject2=subject2)

def loadModelData(model='Resnet',layerID=80,sub='CSI2'):
    scratch60='/gpfs/milgram/scratch60/turk-browne/kp578/harmonic/'
    [activations, imageDataset] = load_obj(f'{scratch60}model/model_{model}-layerID_{layerID}-sub_{sub}') # 保存 图片array 的模型的第i层激活

    destDir = '/gpfs/milgram/scratch60/turk-browne/kp578/harmonic/brain'
    destMeta = '{}/{}_meta.csv'.format(destDir, sub)
    y_sub = pd.read_csv(destMeta)
    y_sub

    # 为了测试的时候节约内存，只使用前200个数据
    activations=activations[:200]
    y_sub=y_sub[:200]

    # 将 N x feature x feature x feature 的数据变成适应PCA的 n x feature 的数据
    activations = activations.reshape(activations.shape[0],-1)
    activations.shape

    # 删除包含缺失值的列。
    activations = removeColumnWithNan(activations)

    # 根据session或者run来选择训练集和测试集
    # trainingID = y_sub1['sess']!=y_sub1['sess'].iloc[-1]
    # testingID = y_sub1['sess']==y_sub1['sess'].iloc[-1]
    trainingID_sub1 = y_sub['run']!=y_sub['run'].iloc[-1]
    testingID_sub1 = y_sub['run']==y_sub['run'].iloc[-1]

    train_model = activations[trainingID_sub1]
    train_label_model = y_sub[trainingID_sub1]
    test_model = activations[testingID_sub1]
    test_label_model = y_sub[testingID_sub1]

    print(f'train_model.shape={train_model.shape}')
    print(f'test_model.shape={test_model.shape}')

    # PCA 降低维度到100维
    training_model , testing_model = pca(X_train=train_model, X_test=test_model)

    return training_model , train_label_model, testing_model, test_label_model

def getAlignmentMetric(xb1_aligned, xb2_aligned): # align的越好，对应的点的距离越近，因此 metric越小越好 。metric越接近1，就说明对齐得越差，因为对角线上的值和非对角线上的值的相对大小差不多。 因此metric的range是【0，正无穷】，超过1时就很差了
    distanceMatrix = scipy.spatial.distance_matrix(xb1_aligned, xb2_aligned) # xb1_aligned 和 xb2_aligned 都有 M 个数据点，N维度，因此 distanceMatrix 就是 M x M 维度。我感兴趣的就是对应点的距离，也就是对角线的和，占全矩阵的和的比值。
    metric = (np.trace(distanceMatrix)/distanceMatrix.shape[0]) / ((np.sum(distanceMatrix)-np.trace(distanceMatrix))/(distanceMatrix.shape[0]*distanceMatrix.shape[1]-distanceMatrix.shape[0]))
    return metric

def harmonicBetweenBrainAndModel(subject='CSI2',model='Resnet',layerID=80):
    training_brain , train_label_brain, testing_brain, test_label_brain = loadBold500SubjectBrainData(subject)
    training_model , train_label_model, testing_model, test_label_model = loadModelData(model='Resnet',layerID=80,sub='CSI2')
    # 进行不同被试数据之间的harmonic alignment
    x1,x2=training_brain,training_model
    n_filters = 8
    align_op = harmonicalignment.HarmonicAlignment(
                int(n_filters),
                t=1, # 1
                overlap=n_filters,
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

    # 获取align完成之后的统一空间
    xb1_aligned=XY_aligned[:training_brain.shape[0]]
    xb2_aligned=XY_aligned[training_brain.shape[0]:]

    alignmentMetric = getAlignmentMetric(xb1_aligned, xb2_aligned)
    print(f"alignmentMetric={alignmentMetric}")

    # brain 数据label为0，model数据label为1
    # label=[0]*training_brain.shape[0]+[1]*training_brain.shape[0]

    # 根据不同的图片来源的数据来标记图片，从而知道详细的align是否成功
    label=[]
    for i in range(len(train_label_brain)):
        if 'imagenet' in train_label_brain['cat'][i]:
            label.append(1)
        elif 'coco' in train_label_brain['cat'][i]:
            label.append(2)
        elif 'scenes' in train_label_brain['cat'][i]:
            label.append(3)
    for i in range(len(train_label_model)):
        if 'imagenet' in train_label_brain['cat'][i]:
            label.append(1)
        elif 'coco' in train_label_brain['cat'][i]:
            label.append(2)
        elif 'scenes' in train_label_brain['cat'][i]:
            label.append(3)

    _=PhateShow(XY_aligned,label=label,title=f'phate n_filters={n_filters}')

# harmonicBetweenBrainAndModel()




















# def harmonicBetweenSubjects(subject1='CSI1',subject2='CSI2'):
#     destDir = '/gpfs/milgram/scratch60/turk-browne/kp578/harmonic/brain'
#     subspace = False

#     destMeta = '{}/{}_meta.csv'.format(destDir, subject1)
#     destFeat = '{}/{}_feat.npy'.format(destDir, subject1) if subspace else '{}/{}_std.npy'.format(destDir, subject1)
#     y_sub1 = pd.read_csv(destMeta)
#     X_sub1 = np.load(destFeat)

#     destMeta = '{}/{}_meta.csv'.format(destDir, subject2)
#     destFeat = '{}/{}_feat.npy'.format(destDir, subject2) if subspace else '{}/{}_std.npy'.format(destDir, subject2)
#     y_sub2 = pd.read_csv(destMeta)
#     X_sub2 = np.load(destFeat)

#     X_sub1 = np.transpose(X_sub1,(3,0,1,2))
#     X_sub2 = np.transpose(X_sub2,(3,0,1,2))





# old
    # # Use region masks to create brain activity matrix by region for each subject
    # stimfeatures_directory = "/gpfs/milgram/project/turk-browne/projects/rtSynth/pls"
    # brain_regions = ['Fus', 'IT', 'LOC', 'PHC']
    # for region in brain_regions:
    #     for subject in subjects:
    #         brain = np.load('/gpfs/milgram/project/turk-browne/projects/rtSynth/features/{}_std.npy'.format(subject)) # 不存在了
    #         mask = nib.load('/gpfs/milgram/project/turk-browne/projects/rtSynth/rois/{}_{}_stan.nii.gz'.format(subject, region)) # 不存在了
    #         mask = mask.get_fdata().astype(int)

    #         mask = np.where(mask == 1)
    #         region_voxels = []
    #         for time in range((brain.shape[-1])):
    #             region_voxels.append(brain[:,:,:,time][mask])

    #         region_voxels = np.array(region_voxels)
    #         del brain
    #         np.save(f'{stimfeatures_directory}/{region}/brainfeat_{subject}.npy', region_voxels)
