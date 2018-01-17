# preprocessing Google Open Images 

# 处理validation数据部分
```
import csv
import os
import shutil
   
with open(r'D:\TF_Try\tensorflow_models\research\oid_argumented\data_csv\validation\annotations-human-bbox.csv','r',encoding='utf-8') as myFile :
    csvfr = csv.reader(myFile)
    row_val = [row for row in csvfr]
    
with open(r'D:\TF_Try\tensorflow_models\research\oid_argumented\data_csv\validation\annotations-human-bbox.csv','r',encoding='utf-8') as myFile :
    csvfr = csv.reader(myFile)
    column_0_val = [row[0] for row in csvfr]
    
with open(r'D:\TF_Try\tensorflow_models\research\oid_argumented\data_csv\validation\annotations-human-bbox.csv','r',encoding='utf-8') as myFile :
    csvfr = csv.reader(myFile)
    column_2_val = [row[2] for row in csvfr]
    
new_column = column[1:]  


for i in new_column:
   src = os.path.join(r'D:\TF_Try\tensorflow_models\research\oid\raw_images_validation',str(i) +'.jpg')
   if os.path.exists(src):
        shutil.copy2(src,dst)
   else:
       pass
```

读取文件夹中的所有图片
```
import glob as gb

img_path = gb.glob("\\*.jpg")

img_path = gb.glob('*.jpg')

img_path = gb.glob('*')

for item in column_0_val_test_num:
    if item + '.jpg' in img_path:
        pass
    else:
        print(item)
        
2675713441079e56
1625307983409413
5626576622065547
7037237751101327
70771453072e0150

```

# 处理test数据部分
```
import csv
import os
import shutil
import pandas as pd

with open(r'D:\TF_Try\tensorflow_models\research\oid_argumented\data_csv\validation\person_attributes.csv','r',encoding='utf-8') as myFile :
    csvfr = csv.reader(myFile)
    column = [row[2] for row in csvfr]
column = column[1:]

column_no_duplicate = list(set(column))

row_selected = []

with open(r'D:\TF_Try\tensorflow_models\research\oid_argumented\data_csv\test\annotations-human-bbox.csv','r',encoding='utf-8') as myFile_test :
    csvfr_test = csv.reader(myFile_test)
    for row in csvfr_test:
        for element in column_no_duplicate:
            if element == row[2]:
                row_selected.append(row)

with open(r'D:\TF_Try\tensorflow_models\research\oid_argumented\data_csv\validation\person_attributes.csv','r',encoding='utf-8') as myFile :
    csvfr = csv.reader(myFile)
    row = [row for row in csvfr]
    
row_test.insert(0,rows_test[0])

test = pd.DataFrame(data = row_test)  

test.to_csv('pedestrian_attributes_test.csv')

#计算某个类number
count = 0
for item in class_lables:
    if item =='/m/01s55n':
        count +=1
    else:
        pass
print(count)

```

找出未copy过去的图片index
```
new_column = []
for item in column_no_duplicate:
    new_column.append (item + '.jpg')
    
img_path = gb.glob('*.jpg')

for i,item in enumerate(new_column):
    if item in img_path:
        pass
    else:
        print(i,item)
```
# 处理test数据部分2
```
import csv
import os
import shutil
import pandas as pd

with open(r'D:\TF_Try\tensorflow_models\research\oid_argumented\data_csv\validation\person_attributes.csv','r',encoding='utf-8') as myFile :
    csvfr = csv.reader(myFile)
    column = [row[2] for row in csvfr]
column = column[1:]

column_no_duplicate = list(set(column))
with open(r'D:\TF_Try\tensorflow_models\research\oid_argumented\data_csv\test\annotations-human-bbox.csv','r',encoding='utf-8') as myFile_test :
    csvfr_test = csv.reader(myFile_test)
    for row in csvfr_test:
        for element in column_no_duplicate:
            if element == row[2]:
                row_selected.append(row)
                
row_selected = []
with open(r'D:\TF_Try\tensorflow_models\research\oid_argumented\data_csv\test\annotations-human-bbox.csv','r',encoding='utf-8') as myFile_test :
    csvfr_test = csv.reader(myFile_test)
    for row in csvfr_test:
        for element in column_no_duplicate:
            if element == row[2]:
                row_selected.append(row)

with open(r'D:\TF_Try\tensorflow_models\research\oid_argumented\data_csv\validation\person_attributes.csv','r',encoding='utf-8') as myFile :
    csvfr = csv.reader(myFile)
    row = [row for row in csvfr]
    
with open(r'D:\TF_Try\tensorflow_models\research\oid_argumented\data_csv\test\pedestrian_attributes_test.csv','r',encoding='utf-8') as myFile :
    csvfr = csv.reader(myFile)
    column_img_test = [row[1] for row in csvfr]
    
import glob as gb
img_path = gb.glob('*.jpg')

column_no_duplicate = list(set(column_img_test))

for i,item in enumerate(column_no_duplicate):
    if item +'.jpg' in img_path:
        pass
    else:
        dst = r'D:\TF_Try\tensorflow_models\research\oid_argumented\data\test'
        if item  =='\ufeffImageID':
            pass
        else:
            shutil.copy2(os.path.join(r'D:\TF_Try\tensorflow_models\research\oid\raw_images_test',str(item) +'.jpg'),dst)
```
