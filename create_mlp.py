import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ion()
import pdb
import scipy.io as scipy_io
import cPickle as pkl
import numpy.matlib
import sys
import os
import shutil

import numpy as np


if __name__ =='__main__':
  file = h5py.File('train_mlp.h5','r')
  index = np.arange(file['data'].shape[0])
  np.random.shuffle(index)
  file_adj = h5py.File('train_mlp_adj.h5')
   
  
  file_adj.create_dataset('label',(file['label'].shape[0],1),dtype='f8')
  file_adj.create_dataset('data',(file['label'].shape[0],8000),dtype='f8')
  for i in range(file['label'].shape[0]):
    file_adj['label'][i] = file['label'][index[i]].argmax()
    print i
    file_adj['data'][i] = file['data'][index[i]]
  file_adj.close()
  sys.exit()
  #with open('/scratch/ys1297/ecog/source/mlp_ecog/train.h5list','a') as txt_file:
  #  for i in range(17):
  #    file_name ='/scratch/ys1297/ecog/source/mlp_ecog/'+'train_'+str(i)+'.h5'
  #    txt_file.write(file_name)
  #    file_sub = h5py.File(file_name)
  #    file_sub['data'] = file['data'][i*100000:(i+1)*100000]
  #    file_sub['label'] = file['label'][i*100000:(i+1)*100000]
  #    file_sub.close()
  #  print i
  #sys.exit()
 
  
  file1 = h5py.File('error_file_0','r')
  file2 = h5py.File('error_file_1','r')
  file3 = h5py.File('error_file_2','r')
  file4 = h5py.File('error_file_3','r')
  file5 = h5py.File('error_file_4','r')
  file6 = h5py.File('error_file_5','r')
  file7 = h5py.File('error_file_6','r')
  file8 = h5py.File('error_file_7','r')
  file_list=[]
  file_list.append(file1)
  file_list.append(file2)
  file_list.append(file3)
  file_list.append(file4)
  file_list.append(file5)
  file_list.append(file6)
  file_list.append(file7)
  file_list.append(file8)
  
  file_training = h5py.File('test_mlp.h5','w')
  file_training.create_dataset('label',(file1['feature'].shape[0],8),dtype='f8')
  file_training.create_dataset('data',(file1['feature'].shape[0],8000),dtype='f8')
  for i in range(file1['prediction'].shape[0]):
    feature = np.concatenate([file1['feature'][i],file2['feature'][i],file3['feature'][i],file4['feature'][i],file5['feature'][i],file6['feature'][i],file7['feature'][i],file8['feature'][i]],axis=1)
    file_training['data'][i]= feature
    tmp = np.zeros(8)
    for j in range(8):
      file = file_list[j] 
      tmp[j] = np.sum((np.asarray(file['prediction'][i])-np.asarray(file['truth'][i][20:,:]))*(np.asarray(file['prediction'][i])-np.asarray(file['truth'][i][20:,:]))).sum() 
    label = np.zeros(8)
    label[np.argmin(tmp)]= 1
    file_training['label'][i]= label
    #print np.argmin(tmp)
    if i % 10000==0:
      print "current:"+str(i) 
  file_training.close()
