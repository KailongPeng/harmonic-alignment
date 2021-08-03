wordingDir = '/gpfs/milgram/project/turk-browne/projects/rtSynth/pls'
import os
import sys
import csv
import numpy as np
import pandas as pd
import nibabel as nib
import datetime
import matplotlib.pyplot as plt
import ray
ray.init()

# TURN DATA FROM ORIGINAL BOLD5000 TO FEATURE ARRAYS AND METADATA

subspace = False

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


# where is bold5000
projDir = '/gpfs/milgram/data/bold5000'

# where should output go
destDir = '/gpfs/milgram/project/turk-browne/projects/rtSynth/features'

subjects = ['CSI1', 'CSI2', 'CSI3', 'CSI4']

# 获得所有的图像对应的大脑的活动数据 Get all brain features after image onset 
@ray.remote
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
            dat = load_data(projDir, subject, sess, run, native=subspace, zscore = False)


            # EXPERIMENTAL TAKE THIS OUT
            critTR = np.minimum(critTR, dat.shape[3] - 1)
            print("critTR", runDat['onsetS'][:5], critTR[:5])

            theseFeatures = dat[:,:,:,critTR.astype(int)]
            theseFeatures = zscore_data(theseFeatures)
            subFeatures.append(theseFeatures) # = theseFeatures if rnum == 0 and snum == 0 else np.concatenate((subFeatures, theseFeatures), 3)

    thisSub.to_csv(destMeta)
    np.save(destFeat, np.concatenate(subFeatures, 3))

for subject in subjects:
    get_brain_features_for_subject.remote(subject)


# Use region masks to create brain activity matrix by region for each subject
stimfeatures_directory = "/gpfs/milgram/project/turk-browne/projects/rtSynth/pls"
brain_regions = ['Fus', 'IT', 'LOC', 'PHC']
for region in brain_regions:
    for subject in subjects:
        brain = np.load('/gpfs/milgram/project/turk-browne/projects/rtSynth/features/{}_std.npy'.format(subject))
        mask = nib.load('/gpfs/milgram/project/turk-browne/projects/rtSynth/rois/{}_{}_stan.nii.gz'.format(subject, region))
        mask = mask.get_fdata().astype(int)

        mask = np.where(mask == 1)
        region_voxels = []
        for time in range((brain.shape[-1])):
            region_voxels.append(brain[:,:,:,time][mask])

        region_voxels = np.array(region_voxels)
        del brain
        np.save(f'{stimfeatures_directory}/{region}/brainfeat_{subject}.npy', region_voxels)
