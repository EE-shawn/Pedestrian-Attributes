#data preprocessing:

Extract Gender images from whole RAP dataset

note:

when conducting a deep learning project, most time we have spent on data preprocessing. this file tells how to extract relevant data from database. and then do some preprocessing.

# code
```
import time
import datetime
import numpy as np
import pandas as pd
import h5py
import json
import cv2
import scipy.io as sio

h5file = r'anote_new.hdf5'
num_classes = 51
with h5py.File(h5file,'r') as mat:
    attri_eng = mat['attri_eng'][()]
    attri_exp = mat['attri_exp'][()]
    labels = mat['labels'][()]
    names = mat['names'][()]
    
attri_eng = json.loads(attri_eng)
attri_exp = json.loads(attri_exp)
attri_eng = [attri_eng[str(i)] for i in range(len(attri_eng))]
attri_exp = [attri_exp[str(i)] for i in range(len(attri_exp))]

names = [name.decode() for name in names]
if True:
    names = ['img_%d.png' % int(i) for i in range(len(names))]
    
labels = np.asarray(labels,dtype=np.int32)

gender_labels = labels[:,0]

for i, (a, b) in enumerate(zip(gender_labels, names)):
    print (i, a, b)
    
import shutil
import os
src = os.path.join('D:\Attris\Data\RAP_dataset', b)
dst_meale = r'D:\Attris\Data\RAP_dataset_seperate\Gender\male'
dst_female = r'D:\Attris\Data\RAP_dataset_seperate\Gender\female'
for i, (a, b) in enumerate(zip(gender_labels, names)):
    src = os.path.join('D:\Attris\Data\RAP_dataset', b)
    if a ==1:
        shutil.copy2(src,dst_female)
    elif a==0:
        shutil.copy2(src,dst_meale)
 ```     
 
 Extract training, validation and test images from PA100K dataset:
 
 # code for training data
 
  ```
import numpy as np
import pandas as pd
import h5py
import json
import scipy.io as sio
import shutil
import os

mat = sio.loadmat('annotation.mat')
train_img = mat['train_images_name']
train_label = mat['train_label']
attributes = mat['attributes']
attributes = [str(atr[0][0]) for atr in attributes]
train_img_names = [str(name[0][0]) for name in train_img]
attributes = json.dumps({i:s for i,s in enumerate(attributes)})
train_img_names_encode = [name.encode() for name in train_img_names]

matfile = 'annotation_'+'train'
savename = matfile.split('.')[0] + '.hdf5'
with h5py.File(savename,'w') as f:
    f.create_dataset(name='attributes',data=attributes)
    f.create_dataset(name='train_img_names',data=train_img_names_encode)
    f.create_dataset(name='train_label',data=train_label)
    f.flush()
  
  
h5file = r'annotation_train.hdf5'

with h5py.File(h5file,'r') as mat:
    attri_eng = mat['attributes'][()]
    labels = mat['train_label'][()]
    names = mat['train_img_names'][()]
    
attri_eng = json.loads(attri_eng)
attri_eng = [attri_eng[str(i)] for i in range(len(attri_eng))]
gender_labels = labels[:,0]
gender_labels = np.asarray(gender_labels,dtype=np.int32)
names = [name.decode() for name in names] 

dst_meale = r'D:\Attris\Data\PA-100K\gender\train\male'
dst_female = r'D:\Attris\Data\PA-100K\gender\train\female'
for i, (a, b) in enumerate(zip(gender_labels, names)):
    src = os.path.join(r'D:\Attris\Data\PA-100K\data\release_data\release_data', b)
    if a ==1:
        shutil.copy2(src,dst_female)
    elif a==0:
        shutil.copy2(src,dst_meale)

  ```
  
  # code for validation 
  
 ```
import numpy as np
import pandas as pd
import h5py
import json
import scipy.io as sio
import shutil
import os

mat = sio.loadmat('annotation.mat')
validation_img = mat['val_images_name']
validation_label = mat['val_label']
attributes = mat['attributes']
attributes = [str(atr[0][0]) for atr in attributes]
validation_img_names = [str(name[0][0]) for name in validation_img]
attributes = json.dumps({i:s for i,s in enumerate(attributes)})
validation_img_names_encode = [name.encode() for name in validation_img_names]

matfile = 'annotation_'+'validation'
savename = matfile.split('.')[0] + '.hdf5'
with h5py.File(savename,'w') as f:
    f.create_dataset(name='attributes',data=attributes)
    f.create_dataset(name='validation_img_names',data=validation_img_names_encode)
    f.create_dataset(name='validation_label',data=validation_label)
    f.flush()
    
 h5file = r'annotation_validation.hdf5'

with h5py.File(h5file,'r') as mat:
    attri_eng = mat['attributes'][()]
    labels = mat['validation_label'][()]
    names = mat['validation_img_names'][()]

attri_eng = json.loads(attri_eng)
attri_eng = [attri_eng[str(i)] for i in range(len(attri_eng))]
gender_labels = labels[:,0]
gender_labels = np.asarray(gender_labels,dtype=np.int32)
names = [name.decode() for name in names] 

dst_meale = r'D:\Attris\Data\PA-100K\gender\validation\male'
dst_female = r'D:\Attris\Data\PA-100K\gender\validation\female'
for i, (a, b) in enumerate(zip(gender_labels, names)):
    src = os.path.join(r'D:\Attris\Data\PA-100K\data\release_data\release_data', b)
    if a ==1:
        shutil.copy2(src,dst_female)
    elif a==0:
        shutil.copy2(src,dst_meale)   
    
 ```
 
 # code for test data
 
 ```
 import numpy as np
import pandas as pd
import h5py
import json
import scipy.io as sio
import shutil
import os

mat = sio.loadmat('annotation.mat')
test_img = mat['test_images_name']
test_label = mat['test_label']
attributes = mat['attributes']
attributes = [str(atr[0][0]) for atr in attributes]
test_img_names = [str(name[0][0]) for name in test_img]
attributes = json.dumps({i:s for i,s in enumerate(attributes)})
test_img_names_encode = [name.encode() for name in test_img_names]

matfile = 'annotation_'+'test'
savename = matfile.split('.')[0] + '.hdf5'
with h5py.File(savename,'w') as f:
    f.create_dataset(name='attributes',data=attributes)
    f.create_dataset(name='test_img_names',data=test_img_names_encode)
    f.create_dataset(name='test_label',data=test_label)
    f.flush()

h5file = r'annotation_test.hdf5'

with h5py.File(h5file,'r') as mat:
    attri_eng = mat['attributes'][()]
    labels = mat['test_label'][()]
    names = mat['test_img_names'][()]

attri_eng = json.loads(attri_eng)
attri_eng = [attri_eng[str(i)] for i in range(len(attri_eng))]
gender_labels = labels[:,0]
gender_labels = np.asarray(gender_labels,dtype=np.int32)
names = [name.decode() for name in names] 

dst_meale = r'D:\Attris\Data\PA-100K\gender\test\male'
dst_female = r'D:\Attris\Data\PA-100K\gender\test\female'
for i, (a, b) in enumerate(zip(gender_labels, names)):
    src = os.path.join(r'D:\Attris\Data\PA-100K\data\release_data\release_data', b)
    if a ==1:
        shutil.copy2(src,dst_female)
    elif a==0:
        shutil.copy2(src,dst_meale)
 
 ```
  
 # batch modify image name
 ```
 import os
 
 image_path = r'D:\Attris\Data\PA-100K\gender\train\female'
 file_list  = os.listdir(image_path)
 i = 0
 for item in file_list:
    os.rename(os.path.join(image_path,item),os.path.join(image_path,('female'+str(i)+'.jpg')))
    i+=1
    
 ```
# batch draw bounding box

```

import time
import datetime
import numpy as np
import pandas as pd
import h5py
import json
import shutil
import cv2
import scipy.io as sio
import json
import warnings


cd D:\Attris\Data\RAP_annotation
D:\Attris\Data\RAP_annotation

matfile = r'RAP_annotation.mat'
mat = sio.loadmat(matfile)
mat = mat['RAP_annotation']


mat = mat[0,0]
labels = mat[1]
attri_eng = mat[3]
names = mat[5]
position = mat[4]

attri_eng = [str(atr[0][0]) for atr in attri_eng]
names = [str(name[0][0]) for name in names]

attri_eng_json = json.dumps({i:s for i,s in enumerate(attri_eng)})
names_encode = [name.encode() for name in names]

position_copy = np.copy(position)

position_x = position_copy[:,0]

position_y = position_copy[:,1]

position_copy[:,4] = position_copy[:,4] - position_x

position_copy[:,8] = position_copy[:,8] - position_x

position_copy[:,12] = position_copy[:,12] - position_x

position_copy[:,5] = position_copy[:,5] - position_y

position_copy[:,9] = position_copy[:,9] - position_y

position_copy[:,13] = position_copy[:,13] - position_y

position_copy[:,[0,1]]=0

position_copy[position_copy < 0] = 0

position_final = position_copy[:,[4,5,6,7,8,9,10,11,12,13,14,15]]

savename = matfile.split('.')[0] + '-with_position' + '.hdf5'

with h5py.File(savename,'w') as f:
   f.create_dataset(name='attri_eng',data=attri_eng_json)
   f.create_dataset(name='names',data=names_encode)
   f.create_dataset(name='labels',data=labels)
   f.create_dataset(name='position',data=position)
   f.create_dataset(name='position_modified',data=position_copy)
   f.create_dataset(name='position_final',data=position_final)
   f.flush()
   
for i in range(5,41585):
    image_path = basedir + '\img_'+ str(i) + '.png'
    orig = cv2.imread(image_path)
    orig = cv2.rectangle(orig,(position_final[i][0],position_final[i][1]),(position_final[i][0] + position_final[i][2],position_final[i][1] + position_final[i][3]),(255,0,0),2)
    orig = cv2.rectangle(orig,(position_final[i][4],position_final[i][5]),(position_final[i][4] + position_final[i][6],position_final[i][5] + position_final[i][7]),(0,255,0),2)
    orig = cv2.rectangle(orig,(position_final[i][8],position_final[i][9]),(position_final[i][8] + position_final[i][10],position_final[i][9] + position_final[i][11]),(0,0,255),2)    
    cv2.imwrite(image_path,orig)
    
```
