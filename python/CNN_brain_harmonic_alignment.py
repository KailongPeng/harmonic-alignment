# 环境准备

    import numpy as np
    import matplotlib,os
    # from loader import Loader
    import matplotlib.pyplot as plt
    import matplotlib.cm
    from sklearn import preprocessing
    import pickle
    import sys
    import pandas as pd
    import random
    from timeit import default_timer as timer
    from scipy import stats
    import sklearn
    from sklearn.decomposition import PCA
    from sklearn.model_selection import cross_val_predict
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.metrics import mean_squared_error, r2_score

    def save_obj(obj, name):
        with open(name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    def load_obj(name):
        with open(name + '.pkl', 'rb') as f:
            return pickle.load(f)

    def shuffleRows(arr1):
        arr1=list(arr1)
        random.shuffle(arr1)
        arr1=np.asarray(arr1)
        return arr1

    class Scaling:
        def fit(self, train=0, test=0):
            self.scaler = sklearn.preprocessing.StandardScaler()  # z = (x - u) / s
            self.scaler.fit(train)
            self.pcs_train = self.scaler.transform(train) # equivelent with stats.zscore(b, axis=0)
            if type(test)!=int:
                self.pcs_test = self.scaler.transform(test)

    class activation_PCA:
        def fit(self, X_train=0, X_test=0, num_components=1000):
            n_samples, n_features = X_train.shape
            self.num_components = min(n_samples, n_features) if num_components > min(n_samples, n_features) else num_components
            self.model = PCA(self.num_components)
            self.model.fit(X_train)
            # print('explained_variance_ratio=',self.model.explained_variance_ratio_)
            # principle components are the coefficients that transform original data into principal vectors space.
            self.pcs_train = self.model.transform(X_train)
            if type(X_test)!=int:
                self.pcs_test = self.model.transform(X_test)

    def zscoreForEachRow(data):
        data=data.T
        return ((data-np.mean(data,0))/np.std(data,0)).T

# 图像获取，label保存
# CNN activation 获取
# 大脑数据获取
# 预处理，包括PCA等等

def get_bold5000data(model_layer=9,shuffleFlag=False,repetitionFlag=1):#0052_StrictCorrespondence_batch100_zscorePCA_leaveOutRuns_noDiscriminator_complexGenerator_savingX1X12X2X21_combine_all_ROIs_CSI3_zscore_after_PCA
    stimfeatures_directory = "/gpfs/milgram/project/turk-browne/projects/rtSynth/pls"
    network = 'robust'

    SingleSubjectFlag=False
    if SingleSubjectFlag==True:
        subjects = ['CSI3']
        testRunsNum=25
    else:
        subjects = ['CSI1', 'CSI2', 'CSI3', 'CSI4']
        testRunsNum=25

    model_train,brain_train,sessionRun_train=0,0,0
    model_test,brain_test,sessionRun_test=0,0,0
    testSubject=subjects[0]
    for i,subject in enumerate(subjects):
        print("subject=",subject)
        if subject==testSubject:
            _model=np.load(f"{stimfeatures_directory}/LOC/imfeat_layer_{model_layer}_subject_{subject}_{network}.npy")
            _model=_model.reshape((_model.shape[0],-1))

            # _brain=np.load(f"{stimfeatures_directory}/wang2014/roi{brain_roi}/brainfeat_{subject}.npy")
            _brain=0
            for brain_roi in range(1,26):
                brain=np.load(f"{stimfeatures_directory}/wang2014/roi{brain_roi}/brainfeat_{subject}.npy")
                _brain=brain if type(_brain)==int else np.concatenate((_brain,brain),axis=1)
            print(_brain.shape)

            _sessionRun=np.load(f'/gpfs/milgram/project/turk-browne/projects/rtSynth/sessionRun/subject_{subject}.npy')

            #randomly choose n=10 runs from the range of all runIDs
            np.random.seed(repetitionFlag)
            test_runID=np.random.choice(
                np.unique(_sessionRun),
                testRunsNum,
                replace=False)
            testPositive=np.mean(np.asarray([(_sessionRun==currRunID)*1 for currRunID in test_runID]),axis=0)
            testID=[bool(_) for _ in testPositive]
            trainID=[not _ for _ in testID]

            print(sum(1*trainID))
            print(sum(1*testID))
            print(len(_sessionRun))
            print(sum(1*testID)/sum(1*trainID))

            X1_train=_model[trainID]
            X1_test=_model[testID]
            X2_train=_brain[trainID]
            X2_test=_brain[testID]
            _sessionRun_train=_sessionRun[trainID]
            _sessionRun_test=_sessionRun[testID]

            # zscore across images/time
            scaler1=Scaling()
            scaler1.fit(train=X1_train,test=X1_test)
            scaler2=Scaling()
            scaler2.fit(train=X2_train,test=X2_test)

            model_train=scaler1.pcs_train if type(model_train)==int else np.concatenate((model_train,scaler1.pcs_train),axis=0)
            brain_train=scaler2.pcs_train if type(brain_train)==int else np.concatenate((brain_train,scaler2.pcs_train),axis=0)
            sessionRun_train=_sessionRun_train if type(sessionRun_train)==int else np.concatenate([sessionRun_train,_sessionRun_train+i*1000])

            model_test=scaler1.pcs_test if type(model_test)==int else np.concatenate((model_test,scaler1.pcs_test),axis=0)
            brain_test=scaler2.pcs_test if type(brain_test)==int else np.concatenate((brain_test,scaler2.pcs_test),axis=0)
            sessionRun_test=_sessionRun_test if type(sessionRun_test)==int else np.concatenate([sessionRun_test,_sessionRun_test+i*1000])

            print(model_train.shape)
            print(model_test.shape)
            print(brain_train.shape)
            print(brain_test.shape)
            print('\n')

        elif subject!=testSubject:
            _model=np.load(f"{stimfeatures_directory}/LOC/imfeat_layer_{model_layer}_subject_{subject}_{network}.npy")
            _model=_model.reshape((_model.shape[0],-1))

            # _brain=np.load(f"{stimfeatures_directory}/wang2014/roi{brain_roi}/brainfeat_{subject}.npy")
            _brain=0
            for brain_roi in range(1,26):
                brain=np.load(f"{stimfeatures_directory}/wang2014/roi{brain_roi}/brainfeat_{subject}.npy")
                _brain=brain if type(_brain)==int else np.concatenate((_brain,brain),axis=1)
            print(_brain.shape)
            _sessionRun=np.load(f'/gpfs/milgram/project/turk-browne/projects/rtSynth/sessionRun/subject_{subject}.npy')

            # zscore across images/time
            scaler1=Scaling()
            scaler1.fit(train=_model,test=0)
            scaler2=Scaling()
            scaler2.fit(train=_brain,test=0)

            model_train=scaler1.pcs_train if type(model_train)==int else np.concatenate((model_train,scaler1.pcs_train),axis=0)
            brain_train=scaler2.pcs_train if type(brain_train)==int else np.concatenate((brain_train,scaler2.pcs_train),axis=0)
            sessionRun_train=_sessionRun if type(sessionRun_train)==int else np.concatenate([sessionRun_train,_sessionRun+i*1000])

            print(model_train.shape)
            print(brain_train.shape)
            print('\n')
    print(model_train.shape)
    print(model_test.shape)
    print(brain_train.shape)
    print(brain_test.shape)

    print('testing data proportion=',brain_test.shape[0]/brain_train.shape[0])

    # shuffle the image order, which is shuffle across images/time, shuffling across rows, within columns.
    if shuffleFlag:
        model_train=shuffleRows(model_train)
        model_test=shuffleRows(model_test)

    # reordered ID for both domains
    assert model_train.shape[0] == brain_train.shape[0]
    reorder_train=list(range(model_train.shape[0]))
    reorder_test=list(range(model_test.shape[0]))
    np.random.shuffle(reorder_train)
    np.random.shuffle(reorder_test)

    X1_train=model_train[reorder_train]
    X1_test=model_test[reorder_test]

    X2_train=brain_train[reorder_train]
    X2_test=brain_test[reorder_test]

    sessionRun_train=sessionRun_train[reorder_train]
    sessionRun_test=sessionRun_test[reorder_test]

    print("Batch 1 shape: {} Batch 2 shape: {}".format(X1_train.shape, X2_train.shape))
    print("Batch 1 shape: {} Batch 2 shape: {}".format(X1_test.shape, X2_test.shape))


    # PCA from number of filters/voxels to 1000 PCs
    PCA1=activation_PCA()
    PCA1.fit(X_train=X1_train,
             X_test=X1_test)
    PCA2=activation_PCA()
    PCA2.fit(X_train=X2_train,
             X_test=X2_test)


    xb1, xb1_test=PCA1.pcs_train,PCA1.pcs_test
    xb2, xb2_test=PCA2.pcs_train,PCA2.pcs_test

    # # zscore across images/time
    # scaler1=Scaling()
    # scaler1.fit(train=xb1,test=xb1_test)
    # scaler2=Scaling()
    # scaler2.fit(train=xb2,test=xb2_test)
    #
    # xb1, xb1_test=scaler1.pcs_train,scaler1.pcs_test
    # xb2, xb2_test=scaler2.pcs_train,scaler2.pcs_test


    print("Batch 1 shape: {} Batch 2 shape: {}".format(xb1.shape, xb2.shape))
    print("Batch 1 shape: {} Batch 2 shape: {}".format(xb1_test.shape, xb2_test.shape))

    return xb1, xb2, xb1_test, xb2_test, [testSubject,test_runID]

for model_layer in range(9,-1,-1):
    # model_layer=9 #int(sys.argv[2])
    # for brain_roi in range(1,26+1):
    brain_roi=1
    # brain_roi= 1 #int(sys.argv[1])
    heldoutSubject=0 #int(sys.argv[3])
    shuffleFlag=False #bool(int(sys.argv[4]))
    savepath="/gpfs/milgram/scratch60/turk-browne/kp578/MAGAN/result/"+'0070_StrictCorrespondence_batch100_zscorePCA_leaveOutRuns_noDiscriminator_complexGenerator_savingX1X12X2X21_combine_all_ROIs_zscore_PCA_allSubjects/'
    if not os.path.isdir(savepath):
        os.mkdir(savepath)
    savefile=savepath+'roi_{}_layer{}_repetitionIndicator_{}_shuffleFlag_{}'.format(brain_roi,model_layer,heldoutSubject,shuffleFlag)

    if os.path.isfile(savefile+'.pkl'):
        raise NameError(savefile+'exist!')
    # xb1, xb2, labels1, labels2 , xb1_test, xb2_test, labels1_test, labels2_test = get_data(brain_roi,model_layer,heldoutSubject,shuffleFlag)
    xb1, xb2, xb1_test, xb2_test, test_runID = get_bold5000data(model_layer=model_layer,repetitionFlag=heldoutSubject,shuffleFlag=shuffleFlag)

    print("Batch 1 shape: {} Batch 2 shape: {}".format(xb1.shape, xb2.shape))
    np.save(f"./cpsc_final/xb1_model_layer_{model_layer}",xb1)
    np.save(f"./cpsc_final/xb1_test_model_layer_{model_layer}",xb1_test)

    np.save(f"./cpsc_final/xb2_brain",xb2)
    np.save(f"./cpsc_final/xb2_test_brain",xb2)

# harmonic alignment

import random
k=1000 # 3000 len(xb1)
print(f"k={k}")
ID1=random.choices(range(len(xb1)),k=k)
ID2=ID1 #random.choices(range(len(xb1)),k=1000)

import harmonicalignment
ha_op = harmonicalignment.HarmonicAlignment(n_filters=4)
ha_op.align(xb1[ID1,:], xb2[ID2,:])
XY_aligned = ha_op.diffusion_map()

xb1_aligned=XY_aligned[:k]
xb2_aligned=XY_aligned[k:]
# PLSR

def optimise_pls_cv(X_training, X_testing, Y_training, Y_testing, comp_min=1, comp_max=100, step=20,
                    plot_components=True):
    '''Run PLS including a variable number of components, up to n_comp,
       and calculate MSE '''
    mse = []
    component = np.arange(comp_min, comp_max, step)
    if len(component) > 1:
        for i in component:
            pls = PLSRegression(n_components=i, max_iter=1000)
            # Cross-validation
            y_cv_training = cross_val_predict(pls, X_training, Y_training, cv=10)
            mse.append(mean_squared_error(Y_training, y_cv_training))
            comp = 100 * (i + 1 - comp_min) / (comp_max - comp_min)
            # Trick to update status on the same line
            stdout.write("\r%d%% completed" % comp)
            stdout.flush()
            print(': ', i, mse)
        stdout.write("\n")
        # Calculate and print the position of minimum in MSE
        msemin = np.argmin(mse)  # ID of smallest mse
        bestComps = component[msemin]  # value of best comps
        print("Suggested number of components: ", bestComps)
        stdout.write("\n")
    else:
        bestComps = component[0]
    comps = bestComps
    # Define PLS object with optimal number of components
    pls_opt = PLSRegression(n_components=bestComps, max_iter=1000)
    # Fit to the entire dataset
    pls_opt.fit(X_training, Y_training)
    y_c_training = pls_opt.predict(X_training)
    # Cross-validation
    y_cv_training = cross_val_predict(pls_opt, X_training, Y_training, cv=10)
    # Calculate scores for calibration and cross-validation
    score_c = r2_score(Y_training, y_c_training)
    score_cv = r2_score(Y_training, y_cv_training)
    y_c_testing = pls_opt.predict(X_testing)
    score_test = pls_opt.score(X_testing, Y_testing)
    print('testing set score=', score_test)
    # my way of metric:
    corrcoef = []
    for curr_feature in range(Y_testing.shape[1]):
        corrcoef.append(np.corrcoef(Y_testing[:, curr_feature], y_c_testing[:, curr_feature])[0, 1])
    corrcoef = np.asarray(corrcoef)
    meanSquareCorr = np.nanmean(np.square(corrcoef))
    print('meanSquareCorr=', meanSquareCorr)
    from sklearn.metrics import explained_variance_score
    explainedVariance = explained_variance_score(Y_testing, y_c_testing, multioutput='uniform_average')
    print('explainedVariance=', explainedVariance)
    # Calculate mean squared error for calibration and cross validation
    mse_c = mean_squared_error(Y_training, y_c_training)
    mse_cv = mean_squared_error(Y_training, y_cv_training)
    print('R2 calib: %5.3f' % score_c)
    print('R2 CV: %5.3f' % score_cv)
    print('MSE calib: %5.3f' % mse_c)
    print('MSE CV: %5.3f' % mse_cv)
save_dir='./results'
optimise_pls_cv(X_train, X_test, Y_train, Y_test, comp_min=5, comp_max=6, step=20)

