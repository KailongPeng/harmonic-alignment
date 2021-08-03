import pandas as pd
import numpy as np

brain_features_dir = '/gpfs/milgram/project/turk-browne/projects/rtSynth/features' # where is the brain features for metadata
network = 'robust'
stimfeatures_directory = "/gpfs/milgram/project/turk-browne/projects/rtSynth/pls"
brain_regions = ['Fus', 'IT', 'LOC', 'PHC']
subjects = ['CSI1', 'CSI2', 'CSI3', 'CSI4']

# Open the paths of images 获得 图片数据集 COCO ImageNet 以及 Scene 的图片的名称。
with open(f"{stimfeatures_directory}/COCO.txt", 'r') as f:
    coco_fullpaths = f.readlines()
    coco_paths = [im.split("/")[-1][:] for im in coco_fullpaths]
with open(f"{stimfeatures_directory}/ImageNet.txt", 'r') as f:
    imagenet_fullpaths = f.readlines()
    imagenet_paths = [im.split("/")[-1][:] for im in imagenet_fullpaths]
with open(f"{stimfeatures_directory}/Scene.txt", 'r') as f:
    scenes_fullpaths = f.readlines()
    scenes_paths = [im.split("/")[-1][:] for im in scenes_fullpaths]

# Build image features matrix
def append_to_im_features(im_features, name, cat, IMAGENETFEATURES, COCOFEATURES, SCENEFEATURES):
    assert cat in ["imagenet", "coco", "scenes"], f"Unknown {cat} image type in BOLD5000, \
                                                must come from one of [imagenet, coco, scenes]"
    if cat == "imagenet":
        image_path = imagenet_paths
        model_central_features = IMAGENETFEATURES
    elif cat == "coco":
        image_path = coco_paths
        model_central_features = COCOFEATURES
    elif cat == "scenes":
        image_path = scenes_paths
        model_central_features = SCENEFEATURES

    is_no_match = True
    for idx, path in enumerate(image_path):
        if name in path:
            is_no_match = False
            break
    assert is_no_match == False, f"There was no image called {name} in category {cat}"
    # print("model_central_features.shape", model_central_features.shape)
    # print("model_central_features[idx].shape", model_central_features[idx].shape)
    im_features.append(model_central_features[idx])

def align_and_save_imfeatures(brain_features_dir): # 获得被试被展示的图片的id，然后根据这些id获取图片的对应的模型的激活。其中模型的激活来自于f'{stimfeatures_directory}/BOLDfeaturesCOCO/{network}_layer_{layer}.npy' 等地方
    for subject in subjects:
        destMeta = '{}/{}_meta.csv'.format(brain_features_dir, subject)
        metadata = pd.read_csv(destMeta)
        stim_category = list(metadata['cat'])
        stim_name = list(metadata['item'])
        for region in brain_regions:
            for layer in range(0, 10):
                im_features = []
                brain_features = []

                COCOFEATURES = np.load(f'{stimfeatures_directory}/BOLDfeaturesCOCO/{network}_layer_{layer}.npy')
                IMAGENETFEATURES = np.load(f'{stimfeatures_directory}/BOLDfeaturesImageNet/{network}_layer_{layer}.npy')
                SCENEFEATURES = np.load(f'{stimfeatures_directory}/BOLDfeaturesScene/{network}_layer_{layer}.npy')

                for stim_num in range(len(stim_name)):
                    name = stim_name[stim_num]
                    cat = stim_category[stim_num]
                    # some categories start with rep
                    if "rep_" in cat:
                        cat = cat.split("_")[-1]

                    append_to_im_features(im_features, name, cat, IMAGENETFEATURES, COCOFEATURES, SCENEFEATURES)
                    # deal with category
                im_features = np.stack(im_features, axis = 0)
                print("layer", layer, "im_features.shape", im_features.shape)
                np.save(f"{stimfeatures_directory}/{region}/imfeat_layer_{layer}_subject_{subject}_{network}.npy", im_features)

if __name__ == "__main__":
    align_and_save_imfeatures(brain_features_dir)