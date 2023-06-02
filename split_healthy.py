import os
import numpy as np
import nibabel as nib
import pandas as pd


def preprocess_mask(mask):
        mask_WT = mask.copy()
        mask_WT[mask_WT == 1] = 1
        mask_WT[mask_WT == 2] = 1
        mask_WT[mask_WT == 4] = 1
        return mask_WT

df = pd.read_csv('/mnt/lustre/baumgartner/bkc035/data/BraTS2020/TrainingData/survival_info.csv')
root_path = '/mnt/lustre/baumgartner/bkc035/data/BraTS2020/TrainingData'
df2 = pd.DataFrame({'Slice':[x for x in range(15,131)]})
df = pd.merge(df,df2,'cross')
ids = df.loc[:,"Brats20ID"]
ages = df.loc[:,"Age"]
slices = df.loc[:,'Slice']
tumor_ids = []
my_ages = []
tumor_slices = []
zero_ids = []
zero_slices = []
for id,age,sl in zip(ids,ages,slices):
        img_path = os.path.join(root_path,id, id + '_seg.nii.gz')
        mask = np.asarray(nib.load(img_path).dataobj[:,:,sl],dtype=int)
        img_path = os.path.join(root_path,id, id + '_flair.nii.gz')
        img = np.asarray(nib.load(img_path).dataobj[:,:,sl],dtype=float)
        num_zeros = np.count_nonzero(img==0)
        if num_zeros > 54000:
              zero_ids.append(id)
              zero_slices.append(sl)
        else:
                mask = preprocess_mask(mask)
                if 1 in mask:
                        tumor_ids.append(id)
                        my_ages.append(age)
                        tumor_slices.append(sl)
zero_dict = {"Brats20ID" : zero_ids,
           "Slice": zero_slices}
tumor_dict = {"Brats20ID" : tumor_ids,
           "Age": my_ages,
           "Slice": tumor_slices}
df = df.drop('Survival_days',axis=1).drop('Extent_of_Resection',axis=1)
df_tumor = pd.DataFrame(tumor_dict)
df_zero = pd.DataFrame(zero_dict)
df_tumor.to_csv('/mnt/lustre/baumgartner/bkc035/data/BraTS2020/TrainingData/tumor_slices.csv',index=False)
df_zero.to_csv('/mnt/lustre/baumgartner/bkc035/data/BraTS2020/TrainingData/zero_slices.csv',index=False)
df_nonzero = pd.merge(df, df_zero,indicator=True,how='outer').query('_merge=="left_only"').drop('_merge',axis=1)
df_healthy = pd.merge(df_nonzero,df_tumor,indicator=True,how='outer').query('_merge=="left_only"').drop('_merge',axis=1)
df_healthy.to_csv('/mnt/lustre/baumgartner/bkc035/data/BraTS2020/TrainingData/healthy_slices.csv',index=False)