#!/bin/python

# 来自于/gpfs/milgram/project/turk-browne/projects/rtSynth/kp_scratch/parallel_get_activation/299_carChair_python.py
workingDir='/gpfs/milgram/project/turk-browne/projects/rtSynth/kp_scratch/parallel_get_activation/'
import sys
curr_iter=int(sys.argv[1])
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pickle,pdb
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tqdm import tqdm
tf.keras.backend.clear_session()  # For easy reset of notebook state.
#image_model=tf.keras.applications.inception_v3.InceptionV3(weights='imagenet', input_shape=(75,75,3),include_top=False)#input_tensor=None,include_top=True, pooling=None, classes=1000
image_model=tf.keras.applications.inception_v3.InceptionV3(weights='imagenet', input_shape=(299,299,3),include_top=False)
def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)
#try:
#    img=load_obj('../images/upsampled_img_75.75.3')
#except:
#    (x_train, _), _ = tf.keras.datasets.cifar100.load_data(label_mode='fine')
#    x_train = x_train.reshape(50000,32,32,3).astype('float32') / 255
#    train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
#    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)
#    import scipy.ndimage
#    from skimage.transform import resize
#    img = []
#    for idx, im in tqdm(enumerate(x_train)):
#        img.append(resize(im, [75,75,3]))
#    img = np.array(img)
#    save_obj(img,'../images/upsampled_img_75.75.3')


img=load_obj('../dataset/carChair299_old')
names = ["pool", "fc"]
def inside(layer):
    YN=False
    for name in names:
        #pdb.set_trace()
        if name in layer:
            YN=True
    return YN
layers = [[layer.name if inside(layer.name) else ''] for layer in image_model.layers]
layers = [x[0] for x in layers if x != ['']]

layer=layers[curr_iter]
inp = image_model.input
x = image_model.get_layer(layer).output
model1 = tf.keras.Model(inputs=inp, outputs=x)
activations=[]
for ii,curr_img in tqdm(enumerate(img)):
    print(ii)
    res = model1.predict(np.expand_dims(curr_img, axis=0))
    activations.append(res)
save_obj(activations,'../images/activation/carChair.layer.{}'.format(layer))