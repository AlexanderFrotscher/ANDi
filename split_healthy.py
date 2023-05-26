import os
import numpy as np
import nibabel as nib
import pandas as pd

def load_img(file_path):
        data = nib.load(file_path).get_fdata()
        return data


def preprocess_mask(mask):
        mask_WT = mask.copy()
        mask_WT[mask_WT == 1] = 1
        mask_WT[mask_WT == 2] = 1
        mask_WT[mask_WT == 4] = 1
        return mask_WT

df = pd.read_csv('/mnt/lustre/baumgartner/bkc035/data/BraTS2020/TrainingData/survival_info.csv')
root_path = '/mnt/lustre/baumgartner/bkc035/data/BraTS2020/TrainingData'
df2 = pd.DataFrame({'Slice':[x for x in range(15,131)]})
ids = df.loc[:,"Brats20ID"]
ages = df.loc[:,"Age"]
df = pd.merge(df,df2,'cross')
slices = [x for x in range(15,131)]
my_ids = []
my_ages = []
my_slices = []
for id,age in zip(ids,ages):
        img_path = os.path.join(root_path, id + '_seg.nii.gz')
        img = load_img(img_path)
        img = preprocess_mask(img)
        for sl in slices:
                img_slice = img[:,:,sl]
                if 1 in img_slice:
                        my_ids.append(id)
                        my_ages.append(age)
                        my_slices.append(sl)
my_dict = {"Brats20ID" : my_ids,
           "Age": my_ages,
           "Slice": my_slices}
df = df.drop('Survival_days',axis=1).drop('Extent_of_Resection',axis=1)
df_tumor = pd.DataFrame(my_dict)
df_tumor.to_csv('/mnt/lustre/baumgartner/bkc035/data/BraTS2020/TrainingData/tumor_slices.csv',index=False)
df_healthy = pd.merge(df,df_tumor,indicator=True,how='outer').query('_merge=="left_only"').drop('_merge',axis=1)
df_healthy.to_csv('/mnt/lustre/baumgartner/bkc035/data/BraTS2020/TrainingData/healthy_slices.csv',index=False)