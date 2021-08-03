# from classification_models.keras import Classifiers # https://github.com/qubvel/classification_models
from classification_models.classification_models.keras import Classifiers
import numpy as np
import tensorflow as tf
from keras import backend as K
import pickle, random
from collections import OrderedDict
from tqdm import tqdm
from skimage.transform import resize
import sys
import glob
import imageio, re
import os.path
from os import path
# from skimage.io import imread
from imageio import imread
from time import ctime
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
print(os.environ['CONDA_DEFAULT_ENV'])  # show the conda environment currently running

main_dir = '../'
cwd = os.getcwd()
if 'milgram' in cwd:
    scratchDir = '/gpfs/milgram/scratch60/turk-browne/kp578/'
elif 'GoogleDrive' in cwd:
    scratchDir = '../scratch60/'
else:
    scratchDir = '/gpfs/scratch60/turk-browne/kp578/'


def sizeOfArray(x): # 计算一个链表、矩阵占空间大小 # can take list, array, int and float
    import numpy
    if type(x) == numpy.ndarray:
        print('size=', x.shape)
        return x.nbytes
    elif type(x) == int or type(x) == float:
        return sys.getsizeof(x)
    elif type(x) == list:
        print('list len=', len(x))
        mem = 0
        for _ in x:
            mem = mem + sizeOfArray(_)
        return mem
    else:
        return 0

def size(x, name, localVariables=locals().items()): # 计算一个链表、矩阵占空间大小 
    print('#########################################################################################')
    print('{} is {}MB'.format(name, sizeOfArray(x) / 1e6))
    # for var, obj in localVariables:
    #     print(var, sys.getsizeof(obj) / 1e6, 'MB')
    print('#########################################################################################')

def timing(): # 输出现在的时间
    print(ctime())
    print('------------------------------------------------------------------------------------------')

def stage(text): # 输出 text 作为输出的分隔符
    print('==========================================================================================')
    print(text)
    print('==========================================================================================')

def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def pd(): # 设置断点
    import pdb
    pdb.set_trace()

def sizeOfArray(x):  # can take list, array, int and float
    import numpy
    if type(x) == numpy.ndarray:
        print('size=', x.shape)
        return x.nbytes
    elif type(x) == int or type(x) == float:
        return sys.getsizeof(x)
    elif type(x) == list:
        print('list len=', len(x))
        mem = 0
        for _ in x:
            mem = mem + sizeOfArray(_)
        return mem
    else:
        return 0

def size(x, name, localVariables=locals().items()):
    print('#########################################################################################')
    print('{} is {}MB'.format(name, sizeOfArray(x) / 1e6))
    # for var, obj in localVariables:
    #     print(var, sys.getsizeof(obj) / 1e6, 'MB')
    print('#########################################################################################')

def stage(text):
    print('==========================================================================================')
    print(text)
    print('==========================================================================================')

def timing():
    print(ctime())
    print('------------------------------------------------------------------------------------------')

class activation_PCA(): #使用PCA降维到1000维度，降维前归一化。
    def fit(self, X_train=0, X_test=0, num_components=1000):
        n_samples, n_features = X_train.shape
        num_components = min(n_samples, n_features) if num_components > min(n_samples, n_features) else num_components

        # scale the data
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        self.model = PCA(num_components)
        self.model.fit(X_train)
        self.pcs_train = self.model.transform(
            X_train)  # principle components are the coefficients that transform original data into principal vectors space.
        self.pcs_test = self.model.transform(X_test)

def bar(M, xlabel='', ylabel='', title='', xticks=''): # 绘制柱状图
    import matplotlib.pyplot as plt
    import numpy.matlib
    # plot barplot with percentage error bar
    plt.figure(figsize=(4, 3), dpi=70)
    means = np.mean(M, axis=0)
    positions = list(np.arange(M.shape[1]))
    std = np.std(M, axis=0)
    upper = np.percentile(M, 97.5, axis=0)
    lower = np.percentile(M, 2.5, axis=0)

    fig, ax = plt.subplots(figsize=(15, 15))
    #     ax.bar(positions, means, yerr=[means-lower,upper-means], align='center', alpha=0.5, ecolor='black', capsize=10)
    ax.bar(positions, means, yerr=std, align='center', alpha=0.5, ecolor='black', capsize=10)
    ax.set_ylabel(ylabel, fontsize=25)
    ax.set_xlabel(xlabel, fontsize=25)
    ax.set_xticks(positions)
    xticks = np.arange(M.shape[1]) if xticks == '' else xticks
    ax.set_xticklabels(xticks, fontsize=25)  # ,rotation=90
    ax.set_title(title, fontsize=25)
    ax.yaxis.grid(True)
    ax.plot(np.linspace(-0.5, M.shape[1] - 0.5, 100), np.matlib.repmat(1 / 8, 100, 1), '-r', label='baseline')
    plt.tight_layout()
    # plt.savefig('bar_plot_with_error_bars.png')
    plt.show()

def Svm(X, y): # 训练支持向量机并输出准确率
    from sklearn import svm
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import ShuffleSplit
    #     aPCA = activation_PCA() #use PCA to reduce dimensionality
    #     aPCA.fit(X_train=X)
    #     X=aPCA.pcs_train
    size(X, 'X')
    ss = ShuffleSplit(n_splits=5, test_size=0.5, random_state=0)
    acc = []
    for train_index, test_index in ss.split(X):
        X_train, X_test, y_train, y_test = X[train_index], X[test_index], np.ravel(y[train_index]), np.ravel(
            y[test_index])

        # scale the data
        #         scaler = StandardScaler()
        #         scaler.fit(X_train)
        #         X_train=scaler.transform(X_train)
        #         X_test=scaler.transform(X_test)

        # use PCA to reduce dimensionality
        #         aPCA = activation_PCA()
        #         aPCA.fit(X_train=X_train, X_test=X_test)
        #         X_train = aPCA.pcs_train
        #         X_test = aPCA.pcs_test

        clf = svm.SVC(gamma='scale')
        size(X_train, 'X_train')
        clf.fit(X_train, y_train)  # fit svm for training data
        acc.append(clf.score(X_test, y_test))
    return acc

def readtxt(path):
    f = open(path, "r")
    return f.read()

def getImageNameList(matfile): # 获取每一个run的图片的名称和来源的数据库，形式是list
    RunImageList=scipy.io.loadmat(matfile)['cur_img_names']
    txtfile = matfile.split('.mat')[0] + '.txt'
    txt = readtxt(txtfile)
    imageNames = txt.split('\n')

    ImageNameList_eachRun=[]
    for i in range(len(RunImageList)):
        imageName=RunImageList[i][0][0]

        if isinstance(imageName, str):
            imageDataset = RunImageList[i][1][0]
            assert imageNames[i] == imageName , f"{imageNames[i]} == {imageName}"
        else: # 如果不是string的话，就说明是一个rep，此时就读对应的txt文件，获取对应的行，去掉rep作为当前的图片
            imageName = imageNames[i]
            imageName = imageName.split('rep_')[1] #去掉 图片名字 开头的 rep_ 
            imageDataset = RunImageList[i][1][0].split('rep_')[1] #去掉 图片来源数据库 开头的 rep_

        ImageNameList_eachRun.append([imageName, imageDataset])

    return ImageNameList_eachRun

def getStimuliForSub(sub='CSI1'): # 获取每一个BOLD5000的被试['CSI1','CSI2','CSI3','CSI4']的刺激的图片的顺序
    import scipy.io
    import glob
    # subList=['CSI1','CSI2','CSI3','CSI4']
    imageDictForSub = {}
    
    Stimuli_Presentation_Lists_path='/gpfs/milgram/project/turk-browne/projects/harmonic/harmonic-alignment/python/bold5000_images/BOLD5000_Stimuli/Stimuli_Presentation_Lists/'
    sesList=glob.glob(f'{Stimuli_Presentation_Lists_path}{sub}/{sub}_sess[0-9]*')
    sesList=[i.split('/')[-1] for i in sesList]
    sesList.sort()
    imageNumber=0
    for currSes in sesList:
        imageDictForSub[currSes]={}
        if sub=='CSI1':
            runList=glob.glob(f"{Stimuli_Presentation_Lists_path}{sub}/{currSes}/CSI_sess{currSes[-2:]}_run*.mat")
        else:
            runList=glob.glob(f"{Stimuli_Presentation_Lists_path}{sub}/{currSes}/{sub}_sess{currSes[-2:]}_run*.mat")
        runList=[i.split('/')[-1] for i in runList]
        runList.sort()
        for currRun in runList:
            matfile=f"{Stimuli_Presentation_Lists_path}{sub}/{currSes}/{currRun}"
            # print(len(_['cur_img_names']))
            ImageNameList_eachRun = getImageNameList(matfile)
            # print(f"RunImageList={RunImageList}")
            imageNumber+=len(ImageNameList_eachRun)
            imageDictForSub[currSes][currRun.split('.mat')[0]] = ImageNameList_eachRun

    print(f"imageNumber for {sub} = {imageNumber}")
    return imageDictForSub

def loadImageAsArray(imageName=np.nan, imageDataset=np.nan, img_size=np.nan): #根据图片的名字加载对应的图片成为一个array
    import math
    from imageio import imread
    
    if not isinstance(imageDataset, str):
        if 'COCO' in imageName:
            imageDataset="COCO"
        elif ".JPEG" in imageName:
            imageDataset="ImageNet"
        else:
            imageDataset="Scene"
    else:
        if imageDataset=='coco':
            imageDataset="COCO"
        elif imageDataset=='imagenet':
            imageDataset="ImageNet"
        elif imageDataset=='scenes':
            imageDataset="Scene"

    stimulPath='/gpfs/milgram/project/turk-browne/projects/harmonic/harmonic-alignment/python/bold5000_images/BOLD5000_Stimuli/Scene_Stimuli/Presented_Stimuli/'
    image = imread(f"{stimulPath}/{imageDataset}/{imageName}")
    original_size = image.shape
    if not np.isnan(img_size):
        image = resize(image, [img_size, img_size, 3])
    
    return image

def loadImageAsListOfArray(imageDictForSub): # 加载图片成为一个list 的array，附送一个list包含图片来源的dataset
    # print(f"number of sessions = {len(imageDictForSub)}")
    # print(f"number of runs for each session= {len(imageDictForSub[ list(imageDictForSub.keys())[0] ])}")
    imageArray=[]
    imageDataset=[]
    for sess in tqdm(imageDictForSub):
        for run in imageDictForSub[sess]:
            # print(f"len(imageDictForSub[sess][run])={len(imageDictForSub[sess][run])}")
            for CurrImage_id in range(len(imageDictForSub[sess][run])):
                # print(f"imageName={CurrImage},imageDataset={imageDictForSub[sess][run][CurrImage]}")
                image = loadImageAsArray(imageName=imageDictForSub[sess][run][CurrImage_id][0],imageDataset=imageDictForSub[sess][run][CurrImage_id][1])
                imageArray.append(image)
                imageDataset.append(imageDictForSub[sess][run][CurrImage_id][1])
    return imageArray, imageDataset

def load_activation(imageList, imageDataset, model='Resnet',layerID=80, img_size=100, randomseed=0): # 本函数根据list of 矩阵形式的图片集，再加载训练好的CNN模型（Resnet 或者 Inception），获得模型的activation作为输出
    tf.compat.v1.disable_eager_execution()
    print(f"tf.executing_eagerly()={tf.executing_eagerly()}")
    numpy.random.seed(randomseed)

    # 确保图片的大小和模型接受的图片大小一样，都是 100 x 100 x 3
    if imageList[0].shape[0] !=img_size:
        New_imageList=[]
        for _ in tqdm(imageList):
            _ = resize(_, [img_size, img_size, 3])
            New_imageList.append(_)
        imageList = New_imageList
    _imageList=np.asarray(imageList)
    print(f"_imageList.shape={_imageList.shape}")
    size(_imageList,'_imageList') # 给出数据的内存大小
    
    # 获取 _imageList 的数据范围
    stage('Calculating Activation') ; timing()
    print('range of _imageList = {}-{}'.format(np.min(_imageList), np.max(_imageList), end='\n\n\n'))
    images = np.asarray(_imageList)

    # 加载CNN的模型（Resnet 或者 Inception） Load CNN model
    tf.keras.backend.clear_session()
    if model == "Resnet":
        ResNet18, preprocess_input = Classifiers.get('resnet18')
        model_ = ResNet18((img_size, img_size, 3), weights='imagenet')
        SelectedLayers = ['data', 'bn_data', 'zero_padding2d_1', 'conv0', 'bn0', 'relu0', 'zero_padding2d_2', 'pooling0', 'stage1_unit1_bn1', 'stage1_unit1_relu1', 'zero_padding2d_3', 'stage1_unit1_conv1', 'stage1_unit1_bn2', 'stage1_unit1_relu2', 'zero_padding2d_4', 'stage1_unit1_conv2', 'stage1_unit1_sc', 'add_1', 'stage1_unit2_bn1', 'stage1_unit2_relu1', 'zero_padding2d_5', 'stage1_unit2_conv1', 'stage1_unit2_bn2', 'stage1_unit2_relu2', 'zero_padding2d_6', 'stage1_unit2_conv2', 'add_2', 'stage2_unit1_bn1', 'stage2_unit1_relu1', 'zero_padding2d_7', 'stage2_unit1_conv1', 'stage2_unit1_bn2', 'stage2_unit1_relu2', 'zero_padding2d_8', 'stage2_unit1_conv2', 'stage2_unit1_sc', 'add_3', 'stage2_unit2_bn1', 'stage2_unit2_relu1', 'zero_padding2d_9', 'stage2_unit2_conv1', 'stage2_unit2_bn2', 'stage2_unit2_relu2', 'zero_padding2d_10', 'stage2_unit2_conv2', 'add_4', 'stage3_unit1_bn1', 'stage3_unit1_relu1', 'zero_padding2d_11', 'stage3_unit1_conv1', 'stage3_unit1_bn2', 'stage3_unit1_relu2', 'zero_padding2d_12', 'stage3_unit1_conv2', 'stage3_unit1_sc', 'add_5', 'stage3_unit2_bn1', 'stage3_unit2_relu1', 'zero_padding2d_13', 'stage3_unit2_conv1', 'stage3_unit2_bn2', 'stage3_unit2_relu2', 'zero_padding2d_14', 'stage3_unit2_conv2', 'add_6', 'stage4_unit1_bn1', 'stage4_unit1_relu1', 'zero_padding2d_15', 'stage4_unit1_conv1', 'stage4_unit1_bn2', 'stage4_unit1_relu2', 'zero_padding2d_16', 'stage4_unit1_conv2', 'stage4_unit1_sc', 'add_7', 'stage4_unit2_bn1', 'stage4_unit2_relu1', 'zero_padding2d_17', 'stage4_unit2_conv1', 'stage4_unit2_bn2', 'stage4_unit2_relu2', 'zero_padding2d_18', 'stage4_unit2_conv2', 'add_8', 'bn1', 'relu1', 'pool1', 'fc1', 'softmax']\
            [layerID] #89
    elif model == "Inception":
        Inceptionv3, preprocess_input = Classifiers.get('inceptionv3')
        model_ = Inceptionv3((img_size, img_size, 3), weights='imagenet')
        SelectedLayers = ['input_1', 'conv2d_1', 'batch_normalization_1', 'activation_1', 'conv2d_2', 'batch_normalization_2', 'activation_2', 'conv2d_3', 'batch_normalization_3', 'activation_3', 'max_pooling2d_1', 'conv2d_4', 'batch_normalization_4', 'activation_4', 'conv2d_5', 'batch_normalization_5', 'activation_5', 'max_pooling2d_2', 'conv2d_9', 'batch_normalization_9', 'activation_9', 'conv2d_7', 'conv2d_10', 'batch_normalization_7', 'batch_normalization_10', 'activation_7', 'activation_10', 'average_pooling2d_1', 'conv2d_6', 'conv2d_8', 'conv2d_11', 'conv2d_12', 'batch_normalization_6', 'batch_normalization_8', 'batch_normalization_11', 'batch_normalization_12', 'activation_6', 'activation_8', 'activation_11', 'activation_12', 'mixed0', 'conv2d_16', 'batch_normalization_16', 'activation_16', 'conv2d_14', 'conv2d_17', 'batch_normalization_14', 'batch_normalization_17', 'activation_14', 'activation_17', 'average_pooling2d_2', 'conv2d_13', 'conv2d_15', 'conv2d_18', 'conv2d_19', 'batch_normalization_13', 'batch_normalization_15', 'batch_normalization_18', 'batch_normalization_19', 'activation_13', 'activation_15', 'activation_18', 'activation_19', 'mixed1', 'conv2d_23', 'batch_normalization_23', 'activation_23', 'conv2d_21', 'conv2d_24', 'batch_normalization_21', 'batch_normalization_24', 'activation_21', 'activation_24', 'average_pooling2d_3', 'conv2d_20', 'conv2d_22', 'conv2d_25', 'conv2d_26', 'batch_normalization_20', 'batch_normalization_22', 'batch_normalization_25', 'batch_normalization_26', 'activation_20', 'activation_22', 'activation_25', 'activation_26', 'mixed2', 'conv2d_28', 'batch_normalization_28', 'activation_28', 'conv2d_29', 'batch_normalization_29', 'activation_29', 'conv2d_27', 'conv2d_30', 'batch_normalization_27', 'batch_normalization_30', 'activation_27', 'activation_30', 'max_pooling2d_3', 'mixed3', 'conv2d_35', 'batch_normalization_35', 'activation_35', 'conv2d_36', 'batch_normalization_36', 'activation_36', 'conv2d_32', 'conv2d_37', 'batch_normalization_32', 'batch_normalization_37', 'activation_32', 'activation_37', 'conv2d_33', 'conv2d_38', 'batch_normalization_33', 'batch_normalization_38', 'activation_33', 'activation_38', 'average_pooling2d_4', 'conv2d_31', 'conv2d_34', 'conv2d_39', 'conv2d_40', 'batch_normalization_31', 'batch_normalization_34', 'batch_normalization_39', 'batch_normalization_40', 'activation_31', 'activation_34', 'activation_39', 'activation_40', 'mixed4', 'conv2d_45', 'batch_normalization_45', 'activation_45', 'conv2d_46', 'batch_normalization_46', 'activation_46', 'conv2d_42', 'conv2d_47', 'batch_normalization_42', 'batch_normalization_47', 'activation_42', 'activation_47', 'conv2d_43', 'conv2d_48', 'batch_normalization_43', 'batch_normalization_48', 'activation_43', 'activation_48', 'average_pooling2d_5', 'conv2d_41', 'conv2d_44', 'conv2d_49', 'conv2d_50', 'batch_normalization_41', 'batch_normalization_44', 'batch_normalization_49', 'batch_normalization_50', 'activation_41', 'activation_44', 'activation_49', 'activation_50', 'mixed5', 'conv2d_55', 'batch_normalization_55', 'activation_55', 'conv2d_56', 'batch_normalization_56', 'activation_56', 'conv2d_52', 'conv2d_57', 'batch_normalization_52', 'batch_normalization_57', 'activation_52', 'activation_57', 'conv2d_53', 'conv2d_58', 'batch_normalization_53', 'batch_normalization_58', 'activation_53', 'activation_58', 'average_pooling2d_6', 'conv2d_51', 'conv2d_54', 'conv2d_59', 'conv2d_60', 'batch_normalization_51', 'batch_normalization_54', 'batch_normalization_59', 'batch_normalization_60', 'activation_51', 'activation_54', 'activation_59', 'activation_60', 'mixed6', 'conv2d_65', 'batch_normalization_65', 'activation_65', 'conv2d_66', 'batch_normalization_66', 'activation_66', 'conv2d_62', 'conv2d_67', 'batch_normalization_62', 'batch_normalization_67', 'activation_62', 'activation_67', 'conv2d_63', 'conv2d_68', 'batch_normalization_63', 'batch_normalization_68', 'activation_63', 'activation_68', 'average_pooling2d_7', 'conv2d_61', 'conv2d_64', 'conv2d_69', 'conv2d_70', 'batch_normalization_61', 'batch_normalization_64', 'batch_normalization_69', 'batch_normalization_70', 'activation_61', 'activation_64', 'activation_69', 'activation_70', 'mixed7', 'conv2d_73', 'batch_normalization_73', 'activation_73', 'conv2d_74', 'batch_normalization_74', 'activation_74', 'conv2d_71', 'conv2d_75', 'batch_normalization_71', 'batch_normalization_75', 'activation_71', 'activation_75', 'conv2d_72', 'conv2d_76', 'batch_normalization_72', 'batch_normalization_76', 'activation_72', 'activation_76', 'max_pooling2d_4', 'mixed8', 'conv2d_81', 'batch_normalization_81', 'activation_81', 'conv2d_78', 'conv2d_82', 'batch_normalization_78', 'batch_normalization_82', 'activation_78', 'activation_82', 'conv2d_79', 'conv2d_80', 'conv2d_83', 'conv2d_84', 'average_pooling2d_8', 'conv2d_77', 'batch_normalization_79', 'batch_normalization_80', 'batch_normalization_83', 'batch_normalization_84', 'conv2d_85', 'batch_normalization_77', 'activation_79', 'activation_80', 'activation_83', 'activation_84', 'batch_normalization_85', 'activation_77', 'mixed9_0', 'concatenate_1', 'activation_85', 'mixed9', 'conv2d_90', 'batch_normalization_90', 'activation_90', 'conv2d_87', 'conv2d_91', 'batch_normalization_87', 'batch_normalization_91', 'activation_87', 'activation_91', 'conv2d_88', 'conv2d_89', 'conv2d_92', 'conv2d_93', 'average_pooling2d_9', 'conv2d_86', 'batch_normalization_88', 'batch_normalization_89', 'batch_normalization_92', 'batch_normalization_93', 'conv2d_94', 'batch_normalization_86', 'activation_88', 'activation_89', 'activation_92', 'activation_93', 'batch_normalization_94', 'activation_86', 'mixed9_1', 'concatenate_2', 'activation_94', 'mixed10', 'avg_pool', 'predictions']\
            [layerID]  #313
    print('range of images= {}-{}, should be[0-1]'.format(np.min(images), np.max(images)))

    # 预处理图片数据集： 从0-1到0-255； 经过preprocess_input之后到 0.0-255.0
    images = images * 255  # images = (images / 2 + 0.5) * 255  # cast to [0, 255]
    print('range of images after casting to [0,255] = {}-{}'.format(np.min(images), np.max(images)))
    images = preprocess_input(images)
    print('range of images after preprocessing = {}-{}'.format(np.min(images), np.max(images), end='\n\n\n'))

    # 使用keras获取激活值 Get activations using keras
    inp = model_.input  # input placeholder
    SelectedmodelLayers = []
    for currlayer in model_.layers:
        if currlayer.name == SelectedLayers:
            SelectedmodelLayers.append(currlayer)
            print('currLayer = ', currlayer.name)
    outputs = [layer.output for layer in SelectedmodelLayers]  # 所有层的输出 all layer outputs
    functor = K.function([inp, K.learning_phase()], outputs)  # 执行函数 evaluation function
    activations = functor([images, 1])[0]
    size(activations, 'activations')

    stage('Activation already gotten') ; timing()

    return activations, imageDataset

scratch60='/gpfs/milgram/scratch60/turk-browne/kp578/harmonic/'
# 获取每一个BOLD5000的被试['CSI1','CSI2','CSI3','CSI4']的刺激的图片的顺序
for sub in ['CSI1','CSI2','CSI3','CSI4']:
    imageDictForSub = getStimuliForSub(sub=sub)
    # 加载每一个被试的图片刺激的顺序 成为一个 list 的 array，附送一个list包含 “图片来源的dataset”
    imageList, imageDataset = loadImageAsListOfArray(imageDictForSub)
    # 保存每一个被试的图片刺激的list，以及 图片刺激对应的图片来源的数据库
    save_obj([imageList, imageDataset], f'{scratch60}stimuli/stimuli_{sub}') # 保存加载的list of 图片array

# 加载 模型 Resnet 第 80 层的激活，保存在stimuli文件夹中
for sub in ['CSI1','CSI2','CSI3','CSI4']:
    model = 'Resnet'
    layerID = 80
    [imageList, imageDataset] = load_obj(f'{scratch60}stimuli/stimuli_{sub}') # 加载list of 图片array
    activations, imageDataset = load_activation(imageList, imageDataset, model=model,layerID=layerID, img_size=100, randomseed=0)
    save_obj([activations, imageDataset], f'{scratch60}model/model_{model}-layerID_{layerID}-sub_{sub}') # 保存 图片array 的模型的第i层激活

# 加载大脑的数据
'/gpfs/milgram/project/turk-browne/projects/rtSynth/features/CSI1_meta.csv'






# old
    # save_obj([activations, y, IDfromAllImages], activations_filename)


    # def load_activation(imageList, imageDataset, model='Resnet',layerID=80, img_size=100, randomseed=0): # 本函数根据list of 矩阵形式的图片集，再加载训练好的CNN模型（Resnet 或者 Inception），获得模型的activation作为输出
    #     import numpy  # randomly select images
    #     numpy.random.seed(randomseed)

    #     if imageList[0].shape[0] !=img_size:
    #         New_imageList=[]
    #         for _ in imageList:
    #             _ = resize(_, [img_size, img_size, 3])
    #             New_imageList.append(_)
    #     imageList = New_imageList

    #     stage('Calculating Activation')
    #     size(X, 'X')
    #     size(y, 'y')
    #     timing()
    #     print('range of X = {}-{}'.format(np.min(X), np.max(X), end='\n\n\n'))
        
    #     IDfromAllImages = numpy.random.choice(len(y), image_number, replace=False)
    #     # print('ID of selected images from original dataset:', ID)
    #     y = np.asarray(y)
    #     X = X[IDfromAllImages]
    #     y = y[IDfromAllImages]
    #     print('head of y = ', y[0:10])
    #     images = np.asarray(X)
    #     X = None  # clear memory of X

    #     # 加载CNN的模型（Resnet 或者 Inception） Load CNN model
    #     tf.keras.backend.clear_session()
    #     if model == "Resnet":
    #         ResNet18, preprocess_input = Classifiers.get('resnet18')
    #         model_ = ResNet18((img_size, img_size, 3), weights='imagenet')
    #         SelectedLayers = ['data', 'bn_data', 'zero_padding2d_1', 'conv0', 'bn0', 'relu0', 'zero_padding2d_2', 'pooling0', 'stage1_unit1_bn1', 'stage1_unit1_relu1', 'zero_padding2d_3', 'stage1_unit1_conv1', 'stage1_unit1_bn2', 'stage1_unit1_relu2', 'zero_padding2d_4', 'stage1_unit1_conv2', 'stage1_unit1_sc', 'add_1', 'stage1_unit2_bn1', 'stage1_unit2_relu1', 'zero_padding2d_5', 'stage1_unit2_conv1', 'stage1_unit2_bn2', 'stage1_unit2_relu2', 'zero_padding2d_6', 'stage1_unit2_conv2', 'add_2', 'stage2_unit1_bn1', 'stage2_unit1_relu1', 'zero_padding2d_7', 'stage2_unit1_conv1', 'stage2_unit1_bn2', 'stage2_unit1_relu2', 'zero_padding2d_8', 'stage2_unit1_conv2', 'stage2_unit1_sc', 'add_3', 'stage2_unit2_bn1', 'stage2_unit2_relu1', 'zero_padding2d_9', 'stage2_unit2_conv1', 'stage2_unit2_bn2', 'stage2_unit2_relu2', 'zero_padding2d_10', 'stage2_unit2_conv2', 'add_4', 'stage3_unit1_bn1', 'stage3_unit1_relu1', 'zero_padding2d_11', 'stage3_unit1_conv1', 'stage3_unit1_bn2', 'stage3_unit1_relu2', 'zero_padding2d_12', 'stage3_unit1_conv2', 'stage3_unit1_sc', 'add_5', 'stage3_unit2_bn1', 'stage3_unit2_relu1', 'zero_padding2d_13', 'stage3_unit2_conv1', 'stage3_unit2_bn2', 'stage3_unit2_relu2', 'zero_padding2d_14', 'stage3_unit2_conv2', 'add_6', 'stage4_unit1_bn1', 'stage4_unit1_relu1', 'zero_padding2d_15', 'stage4_unit1_conv1', 'stage4_unit1_bn2', 'stage4_unit1_relu2', 'zero_padding2d_16', 'stage4_unit1_conv2', 'stage4_unit1_sc', 'add_7', 'stage4_unit2_bn1', 'stage4_unit2_relu1', 'zero_padding2d_17', 'stage4_unit2_conv1', 'stage4_unit2_bn2', 'stage4_unit2_relu2', 'zero_padding2d_18', 'stage4_unit2_conv2', 'add_8', 'bn1', 'relu1', 'pool1', 'fc1', 'softmax']\
    #             [layerID] #89
    #     elif model == "Inception":
    #         Inceptionv3, preprocess_input = Classifiers.get('inceptionv3')
    #         model_ = Inceptionv3((img_size, img_size, 3), weights='imagenet')
    #         SelectedLayers = ['input_1', 'conv2d_1', 'batch_normalization_1', 'activation_1', 'conv2d_2', 'batch_normalization_2', 'activation_2', 'conv2d_3', 'batch_normalization_3', 'activation_3', 'max_pooling2d_1', 'conv2d_4', 'batch_normalization_4', 'activation_4', 'conv2d_5', 'batch_normalization_5', 'activation_5', 'max_pooling2d_2', 'conv2d_9', 'batch_normalization_9', 'activation_9', 'conv2d_7', 'conv2d_10', 'batch_normalization_7', 'batch_normalization_10', 'activation_7', 'activation_10', 'average_pooling2d_1', 'conv2d_6', 'conv2d_8', 'conv2d_11', 'conv2d_12', 'batch_normalization_6', 'batch_normalization_8', 'batch_normalization_11', 'batch_normalization_12', 'activation_6', 'activation_8', 'activation_11', 'activation_12', 'mixed0', 'conv2d_16', 'batch_normalization_16', 'activation_16', 'conv2d_14', 'conv2d_17', 'batch_normalization_14', 'batch_normalization_17', 'activation_14', 'activation_17', 'average_pooling2d_2', 'conv2d_13', 'conv2d_15', 'conv2d_18', 'conv2d_19', 'batch_normalization_13', 'batch_normalization_15', 'batch_normalization_18', 'batch_normalization_19', 'activation_13', 'activation_15', 'activation_18', 'activation_19', 'mixed1', 'conv2d_23', 'batch_normalization_23', 'activation_23', 'conv2d_21', 'conv2d_24', 'batch_normalization_21', 'batch_normalization_24', 'activation_21', 'activation_24', 'average_pooling2d_3', 'conv2d_20', 'conv2d_22', 'conv2d_25', 'conv2d_26', 'batch_normalization_20', 'batch_normalization_22', 'batch_normalization_25', 'batch_normalization_26', 'activation_20', 'activation_22', 'activation_25', 'activation_26', 'mixed2', 'conv2d_28', 'batch_normalization_28', 'activation_28', 'conv2d_29', 'batch_normalization_29', 'activation_29', 'conv2d_27', 'conv2d_30', 'batch_normalization_27', 'batch_normalization_30', 'activation_27', 'activation_30', 'max_pooling2d_3', 'mixed3', 'conv2d_35', 'batch_normalization_35', 'activation_35', 'conv2d_36', 'batch_normalization_36', 'activation_36', 'conv2d_32', 'conv2d_37', 'batch_normalization_32', 'batch_normalization_37', 'activation_32', 'activation_37', 'conv2d_33', 'conv2d_38', 'batch_normalization_33', 'batch_normalization_38', 'activation_33', 'activation_38', 'average_pooling2d_4', 'conv2d_31', 'conv2d_34', 'conv2d_39', 'conv2d_40', 'batch_normalization_31', 'batch_normalization_34', 'batch_normalization_39', 'batch_normalization_40', 'activation_31', 'activation_34', 'activation_39', 'activation_40', 'mixed4', 'conv2d_45', 'batch_normalization_45', 'activation_45', 'conv2d_46', 'batch_normalization_46', 'activation_46', 'conv2d_42', 'conv2d_47', 'batch_normalization_42', 'batch_normalization_47', 'activation_42', 'activation_47', 'conv2d_43', 'conv2d_48', 'batch_normalization_43', 'batch_normalization_48', 'activation_43', 'activation_48', 'average_pooling2d_5', 'conv2d_41', 'conv2d_44', 'conv2d_49', 'conv2d_50', 'batch_normalization_41', 'batch_normalization_44', 'batch_normalization_49', 'batch_normalization_50', 'activation_41', 'activation_44', 'activation_49', 'activation_50', 'mixed5', 'conv2d_55', 'batch_normalization_55', 'activation_55', 'conv2d_56', 'batch_normalization_56', 'activation_56', 'conv2d_52', 'conv2d_57', 'batch_normalization_52', 'batch_normalization_57', 'activation_52', 'activation_57', 'conv2d_53', 'conv2d_58', 'batch_normalization_53', 'batch_normalization_58', 'activation_53', 'activation_58', 'average_pooling2d_6', 'conv2d_51', 'conv2d_54', 'conv2d_59', 'conv2d_60', 'batch_normalization_51', 'batch_normalization_54', 'batch_normalization_59', 'batch_normalization_60', 'activation_51', 'activation_54', 'activation_59', 'activation_60', 'mixed6', 'conv2d_65', 'batch_normalization_65', 'activation_65', 'conv2d_66', 'batch_normalization_66', 'activation_66', 'conv2d_62', 'conv2d_67', 'batch_normalization_62', 'batch_normalization_67', 'activation_62', 'activation_67', 'conv2d_63', 'conv2d_68', 'batch_normalization_63', 'batch_normalization_68', 'activation_63', 'activation_68', 'average_pooling2d_7', 'conv2d_61', 'conv2d_64', 'conv2d_69', 'conv2d_70', 'batch_normalization_61', 'batch_normalization_64', 'batch_normalization_69', 'batch_normalization_70', 'activation_61', 'activation_64', 'activation_69', 'activation_70', 'mixed7', 'conv2d_73', 'batch_normalization_73', 'activation_73', 'conv2d_74', 'batch_normalization_74', 'activation_74', 'conv2d_71', 'conv2d_75', 'batch_normalization_71', 'batch_normalization_75', 'activation_71', 'activation_75', 'conv2d_72', 'conv2d_76', 'batch_normalization_72', 'batch_normalization_76', 'activation_72', 'activation_76', 'max_pooling2d_4', 'mixed8', 'conv2d_81', 'batch_normalization_81', 'activation_81', 'conv2d_78', 'conv2d_82', 'batch_normalization_78', 'batch_normalization_82', 'activation_78', 'activation_82', 'conv2d_79', 'conv2d_80', 'conv2d_83', 'conv2d_84', 'average_pooling2d_8', 'conv2d_77', 'batch_normalization_79', 'batch_normalization_80', 'batch_normalization_83', 'batch_normalization_84', 'conv2d_85', 'batch_normalization_77', 'activation_79', 'activation_80', 'activation_83', 'activation_84', 'batch_normalization_85', 'activation_77', 'mixed9_0', 'concatenate_1', 'activation_85', 'mixed9', 'conv2d_90', 'batch_normalization_90', 'activation_90', 'conv2d_87', 'conv2d_91', 'batch_normalization_87', 'batch_normalization_91', 'activation_87', 'activation_91', 'conv2d_88', 'conv2d_89', 'conv2d_92', 'conv2d_93', 'average_pooling2d_9', 'conv2d_86', 'batch_normalization_88', 'batch_normalization_89', 'batch_normalization_92', 'batch_normalization_93', 'conv2d_94', 'batch_normalization_86', 'activation_88', 'activation_89', 'activation_92', 'activation_93', 'batch_normalization_94', 'activation_86', 'mixed9_1', 'concatenate_2', 'activation_94', 'mixed10', 'avg_pool', 'predictions']\
    #             [layerID]  #313
    #     print('range of images= {}-{}, should be[0-1]'.format(np.min(images), np.max(images)))

    #     # 预处理图片数据集 Preprocess data set
    #     images = images * 255  # images = (images / 2 + 0.5) * 255  # cast to [0, 255]
    #     print('range of images after casting to [0,255] = {}-{}'.format(np.min(images), np.max(images)))
    #     images = preprocess_input(images)
    #     print('range of images after preprocessing = {}-{}'.format(np.min(images), np.max(images), end='\n\n\n'))
    #     # 使用keras获取激活值 Get activations using keras
    #     inp = model_.input  # input placeholder
    #     SelectedmodelLayers = []
    #     for currlayer in model_.layers:
    #         if currlayer.name == SelectedLayers:
    #             SelectedmodelLayers.append(currlayer)
    #             print('currLayer = ', currlayer.name)
    #     outputs = [layer.output for layer in SelectedmodelLayers]  # all layer outputs
    #     functor = K.function([inp, K.learning_phase()], outputs)  # evaluation function
    #     activations = functor([images, 1])[0]
    #     # save_obj([activations, y, IDfromAllImages], activations_filename)
    #     size(activations, 'activations')
    #     # activations = activations.reshape(image_number, -1)

    #     # if cat == 'All':
    #     #     return activations, IDfromAllImages
    #     # else:
    #     #     if original_size_flag:
    #     #         return activations, y, IDfromAllImages, original_size
    #     #     else:
    #     #         return activations, y, IDfromAllImages


    # def load_activation( # 本函数根据矩阵形式的图片集，再加载训练好的CNN模型（Resnet 或者 Inception），获得模型的activation作为输出
    #     dataset, image_number, 
    #     model, layerID, randomseed=0, cat='All', img_size=100,
    #     reCalculateActivation=False, original_size_flag=False):
    #     # load activations that has specific dataset and model and img_size and image_number
    #     global original_size
    #     activations_filename = f"{scratchDir}mappingBetweenModels/activation/activations_for_.{dataset}.{model}.{layerID}.{img_size}.{image_number} "
    #     if reCalculateActivation:
    #         try:
    #             os.remove(activations_filename + '.pkl')
    #         except:
    #             pass
    #     try:
    #         timing()
    #         [activations, y, IDfromAllImages] = load_obj(activations_filename) # if path.exists(activations_filename + '.pkl'):
    #         # activations_dict=load_obj(activations_dict_filename)
    #         stage('loading existing activation')
    #     except:



    # def load_BOLD5000(img_size=224, reCalculate=False, image_number=1000, original_size_flag=False, tag=''):
    #     print('loading BOLD5000{}'.format(img_size))
    #     randomSeed = 0
    #     BOLD5000Images = scratchDir + 'result/BOLD5000Images.{}.{}.{}'.format(
    #         image_number, img_size, tag)
    #     if reCalculate:
    #         try:
    #             os.remove(BOLD5000Images + '.pkl')
    #         except:
    #             pass
    #     if path.exists(BOLD5000Images + '.pkl'):
    #         print('loading existing BOLD5000 images')
    #         [X, y] = load_obj(BOLD5000Images)
    #     else:
    #         print('reCalculating images', img_size)

    #         meta = pd.read_csv('/gpfs/milgram/project/turk-browne/projects/rtSynth/kp_scratch/mappingBetweenModels/../../features/CSI2_sorted.csv') #/gpfs/milgram/project/turk-browne/projects/rtSynth/features
    #         imagelist = list(meta['item'])
    #         imageFiles = [main_dir + 'images/{}'.format(image) for image in imagelist] #/gpfs/milgram/project/turk-browne/projects/rtSynth/images

    #         images = []
    #         y = []
    #         original_size = []
    #         for ii, imagePath in tqdm(enumerate(imageFiles)):
    #             img, originalSize = loadImageAsArray(imagePath, img_size)
    #             images.append(img)
    #             original_size.append(originalSize)
    #             y.append(1) #BOLD5000 images are unique, use 1 as a placeholder for future usage in other function
    #         X = np.asarray(images)
    #         y = np.asarray(y)
    #         save_obj([X, y], BOLD5000Images)
    #     size(X,'X')
    #     if original_size_flag:
    #         return [X, y, original_size]
    #     else:
    #         return [X, y]

    # def New_load_BOLD5000(img_size=100, reCalculate=False): # 获取imagenet图片库中的一些图像，输出的是矩阵
    #     print('loading imagenet{}'.format(img_size))
    #     totalNumber = 1000
    #     resizedImageNetImages_1ImageEachCategory = scratchDir + 'result/resizedImageNetImages_1ImageEachCategory.{}.{}'.format(
    #         totalNumber, img_size)
    #     if reCalculate == True:
    #         try:
    #             os.remove(resizedImageNetImages_1ImageEachCategory + '.pkl')
    #         except:
    #             pass
    #     if path.exists(resizedImageNetImages_1ImageEachCategory + '.pkl'):
    #         print('loading existing imagenet images')
    #         [X, y] = load_obj(resizedImageNetImages_1ImageEachCategory)
    #     else:
    #         print('reCalculating images', img_size)
    #         dataDir = '/gpfs/milgram/project/turk-browne/projects/rtSynth/images/'
    #         images = []
    #         y = []
    #         ims = glob.glob(f'{dataDir}*jpg') + glob.glob(f'{dataDir}*JPEG') + glob.glob(f'{dataDir}*jpeg')
    #         print(f"len(ims)={len(ims)}")
    #         for img_path in ims:  # There are 1000 categories in total, select one image from each category
                
    #             img, original_size = loadImageAsArray(img_path, img_size)
    #             images.append(img)
    #             y.append(cat)

    #         X = np.asarray(images)
    #         y = np.asarray(y)
    #         save_obj([X, y], resizedImageNetImages_1ImageEachCategory)
    #     print('imageNet datashape = ', X.shape)
    #     return [X, y]



