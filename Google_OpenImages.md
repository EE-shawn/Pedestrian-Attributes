# preprocessing Google Open Images 

# select images including person and its attributes
```
import csv
import os
import shutil

with open('person_attributes.csv','r',encoding='utf-8') as myFile :
    csvfr = csv.reader(myFile)
    column = [row[0] for row in csvfr]
    
new_column = column[1:]  

for i in new_column:
   src = os.path.join(r'D:\TF_Try\tensorflow_models\research\oid\raw_images_validation',str(i) +'.jpg')
   if os.path.exists(src):
        shutil.copy2(src,dst)
   else:
       pass
       
```

```
import csv
import os
import shutil
import pandas as pd

with open(r'D:\TF_Try\tensorflow_models\research\oid_argumented\data_csv\validation\person_attributes.csv','r',encoding='utf-8') as myFile :
    csvfr = csv.reader(myFile)
    column2 = [row[2] for row in csvfr]
column2_new = column2[1:]

column2_new_no_duplicate = list(set(column2_new))

row_test.insert(0,rows_test[0])

test = pd.DataFrame(data = row_test)

test.to_csv('pedestrian_attributes_test.csv')

```
