import os
from glob import glob
import pandas as pd
from steganogan import SteganoGAN


PATH = '../alaska2-image-steganalysis/SteganoGAN_Attk/*.png'
names = []
secrets = []
steganogan = SteganoGAN.load(architecture='dense')
i = 0
for filename in sorted(glob(PATH)):
#     if i==1000:
#         break
    if i%10==0:
        print(filename)
    i = i + 1
    
    try:
        secret = steganogan.decode(filename)
    except ValueError as ve:
        secret = ve
    names.append(filename.split('/')[-1])
    secrets.append(secret)

img_names, img_attk = [], []
for name in names:
    split = name.split('.')[0].split('_')
    if len(split)==1:
        img_names.append(split[0])
        img_attk.append('None')
    else:
        img_names.append(name.split('.')[0].split('_')[0] )
        img_attk.append(name.split('.')[0].split('_')[1] )

data = {'names': img_names, 'attk':img_attk, 'secrets': secrets}
data = pd.DataFrame.from_dict(data)
data.to_csv('../Evaluation/secrets.csv', index=False)  
