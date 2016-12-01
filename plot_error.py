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
def DisplayData_Ecog(data,ret,fut,output_file):

    data = data.reshape(-1, 20, 18)
    data = np.swapaxes(data,1,2)

    # get reconstruction and future sequences if they exist
    fut = fut.reshape(-1, 20,18)
    fut = np.swapaxes(fut,1,2)
    
    ret = ret.reshape(-1,20,18)
    ret = np.swapaxes(ret,1,2)

    fut = np.concatenate((ret,fut),axis = 0)

    """ new plot method """
    num_rows = 1
    seq_length_= data.shape[0]
    plt.figure(2, figsize=(seq_length_, 3))
    plt.clf()
    #plt.figure(1)
    for i in xrange(seq_length_):
      plt.subplot(num_rows, data.shape[0], i+1)
      plt.imshow(-data[i, :, :], cmap=plt.cm.jet, interpolation="nearest")
      plt.clim(-0.8,-0.3)
      plt.axis('off')
      #plt.subplot(num_rows, seq_length_, i + 1+seq_length_)
      #plt.imshow(-fut[i, :, :], cmap=plt.cm.jet,  interpolation="nearest")
      #plt.clim(-0.8,-0.3)
      #plt.axis('off')
      #
      #plt.subplot(num_rows, seq_length_, i + 1+seq_length_*2)
      #plt.imshow(-data[i, :, :]+fut[i , :, :], cmap=plt.cm.jet,  interpolation="nearest")
      #plt.clim(-0.4,0.4)
      #plt.axis('off')
    plt.draw()
    plt.savefig(output_file, bbox_inches='tight')
    
                           
if __name__ =='__main__':
  #min_index = scipy_io.loadmat('min_index_fut.mat')['min_index'].T
  #min_value =  scipy_io.loadmat('min_index_fut.mat')['min_value'].T
  #for ensemble in range(8):
  #  folder_name = './ensemble'+str(ensemble)+'_min'
  #  try:
  #    shutil.rmtree(folder_name)
  #    os.mkdir(folder_name)
  #  except:
  #    pass  
  #  
  #  read_file_name = 'error_file_train' +str(ensemble)   
  #  file1 =  h5py.File(read_file_name,'r')
  #  truth = file1['truth']
  #  prediction = file1['prediction']
  #  reconstruction = file1['reconstruction']
  #  min_ensemble = np.where(min_index==ensemble)[0]
  #  temp = np.argsort(min_value[min_ensemble].T)
  #  min_ensemble = min_ensemble[temp].flatten()
  #  for i in range(0,min(min_ensemble.size,200),4):
  #      file_name = folder_name+'/'+ str(min_ensemble[i])+'_error'+str(min_value[min_ensemble[i]]) +'.pdf'
  #      #all_recon = np.vstack([reconstruction[min_ensemble[i]],prediction[min_ensemble[i]]])
  #      #DisplayData_Ecog(truth[min_ensemble[i]][0:20,:],prediction[min_ensemble[i]][0:20,:],file_name)
  #    
  #      DisplayData_Ecog(truth[min_ensemble[i]],reconstruction[min_ensemble[i]],prediction[min_ensemble[i]],file_name)
  #      print i
  #
  #sys.exit() 
 #
  """error plot against time """
  #file1 = h5py.File('error_file_4041_ensemble1','r')
  #file2 = h5py.File('error_file_4041_ensemble2','r')
  #file3 = h5py.File('error_file_4041_ensemble3','r')
  #file4 = h5py.File('error_file_4041_ensemble4','r')

  file1 = h5py.File('error_file_train0','r')
  file2 = h5py.File('error_file_train1','r')
  file3 = h5py.File('error_file_train2','r')
  file4 = h5py.File('error_file_train3','r')
  
  file5 = h5py.File('error_file_train4','r')
  file6 = h5py.File('error_file_train5','r')
  file7 = h5py.File('error_file_train6','r')
  file8 = h5py.File('error_file_train7','r')
 
  
  #plot_start_seg = 
  #plot_end_seg=  
  #prediction = (np.asarray(file1['prediction'])+np.asarray(file2['prediction'])+np.asarray(file3['prediction'])+np.asarray(file4['prediction']))/4
  prediction = file1['prediction']
  truth = file1['truth']
  mask = file1['mask']
  reconstruction = file1['reconstruction']
  """generate plots """
  #for i in range(100):
  #id = 41038
  #output_image_name = 'test_img'+str(id)+'.pdf'
  #DisplayData_Ecog(truth[id][0::4,:],prediction[1000],reconstruction[1000],output_image_name)
    #abc = np.asarray(truth[40700+20*i])
    #output_mat_name = 'for_delay'+str(40700+20*i)+'.mat'
    #scipy_io.savemat(output_mat_name,{'data':abc})
    #print i

  #plt.plot(truth[:,0,0].squeeze())
  #plt.draw()
  #plt.savefig('interval.pdf')
  tmp1 = np.sum((np.asarray(prediction)-np.asarray(truth[:,20:,:]))*(np.asarray(prediction)-np.asarray(truth[:,20:,:])),axis=2)[:,0:10] 
  tmp1_r = np.sum((np.asarray(reconstruction) - np.asarray(truth[:,0:20,:]))*(np.asarray(reconstruction) - np.asarray(truth[:,0:20,:])),axis=2)
  abc=  np.sum(tmp1,axis=0)
  scipy_io.savemat('error_single.mat',{'single_error':abc})

  plt.figure()
  plt.clf()
  plt.plot(10*np.log10(1/(abc/prediction.shape[0]/prediction.shape[2])),'rs-')
  plt.draw()
  plt.ylim([15,60])
  plt.savefig('error_0.pdf')
  
  prediction = file2['prediction']
  truth = file2['truth']
  mask = file2['mask']
  reconstruction = file2['reconstruction']
  #plt.plot(truth[:,0,0].squeeze())
  #plt.draw()
  #plt.savefig('interval.pdf')
  tmp2 = np.sum((np.asarray(prediction)-np.asarray(truth[:,20:,:]))*(np.asarray(prediction)-np.asarray(truth[:,20:,:])),axis=2)[:,0:10]    
  tmp2_r = np.sum((np.asarray(reconstruction) - np.asarray(truth[:,0:20,:]))*(np.asarray(reconstruction) - np.asarray(truth[:,0:20,:])),axis=2)
  abc=  np.sum(tmp2,axis=0)
  plt.figure()
  plt.clf()
  plt.plot(10*np.log10(1/(abc/prediction.shape[0]/prediction.shape[2])),'gs-')
  plt.draw()
  plt.ylim([15,60])
  plt.savefig('error_1.pdf')
  
  
  
  prediction = file3['prediction']
  truth = file3['truth']
  mask = file3['mask']
  reconstruction = file3['reconstruction']
  #plt.plot(truth[:,0,0].squeeze())
  #plt.draw()
  #plt.savefig('interval.pdf')
  tmp3 = np.sum((np.asarray(prediction)-np.asarray(truth[:,20:,:]))*(np.asarray(prediction)-np.asarray(truth[:,20:,:])),axis=2)[:,0:10] 
  tmp3_r = np.sum((np.asarray(reconstruction) - np.asarray(truth[:,0:20,:]))*(np.asarray(reconstruction) - np.asarray(truth[:,0:20,:])),axis=2)
  abc=  np.sum(tmp3,axis=0)
  plt.figure()
  plt.clf()
  plt.plot(10*np.log10(1/(abc/prediction.shape[0]/prediction.shape[2])),'gs-')
  plt.draw()
  plt.ylim([15,60])
  plt.savefig('error_2.pdf')
  
  
  prediction = file4['prediction']
  truth = file4['truth']
  mask = file4['mask']
  reconstruction = file4['reconstruction']
  #plt.plot(truth[:,0,0].squeeze())
  #plt.draw()
  #plt.savefig('interval.pdf')
  tmp4 = np.sum((np.asarray(prediction)-np.asarray(truth[:,20:,:]))*(np.asarray(prediction)-np.asarray(truth[:,20:,:])),axis=2)[:,0:10] 
  tmp4_r = np.sum((np.asarray(reconstruction) - np.asarray(truth[:,0:20,:]))*(np.asarray(reconstruction) - np.asarray(truth[:,0:20,:])),axis=2)
  abc=  np.sum(tmp4,axis=0)
  plt.figure()
  plt.clf()
  plt.plot(10*np.log10(1/(abc/prediction.shape[0]/prediction.shape[2])),'gs-')
  plt.draw()
  plt.ylim([15,60])
  plt.savefig('error_3.pdf')
  
  prediction = file5['prediction']
  truth = file5['truth']
  mask = file5['mask']
  reconstruction = file5['reconstruction']
  #plt.plot(truth[:,0,0].squeeze())
  #plt.draw()
  #plt.savefig('interval.pdf')
  tmp5 = np.sum((np.asarray(prediction)-np.asarray(truth[:,20:,:]))*(np.asarray(prediction)-np.asarray(truth[:,20:,:])),axis=2)[:,0:10] 
  tmp5_r = np.sum((np.asarray(reconstruction) - np.asarray(truth[:,0:20,:]))*(np.asarray(reconstruction) - np.asarray(truth[:,0:20,:])),axis=2)
  abc=  np.sum(tmp5,axis=0)
  plt.figure()
  plt.clf()
  plt.plot(10*np.log10(1/(abc/prediction.shape[0]/prediction.shape[2])),'rs-')
  plt.draw()
  plt.ylim([15,60])
  plt.savefig('error_4.pdf')
  
  
  prediction = file6['prediction']
  truth = file6['truth']
  mask = file6['mask']
  reconstruction = file6['reconstruction']
  #plt.plot(truth[:,0,0].squeeze())
  #plt.draw()
  #plt.savefig('interval.pdf')
  tmp6 = np.sum((np.asarray(prediction)-np.asarray(truth[:,20:,:]))*(np.asarray(prediction)-np.asarray(truth[:,20:,:])),axis=2)[:,0:10] 
  tmp6_r = np.sum((np.asarray(reconstruction) - np.asarray(truth[:,0:20,:]))*(np.asarray(reconstruction) - np.asarray(truth[:,0:20,:])),axis=2)
  abc=  np.sum(tmp6,axis=0)
  plt.figure()
  plt.clf()
  plt.plot(10*np.log10(1/(abc/prediction.shape[0]/prediction.shape[2])),'gs-')
  plt.draw()
  plt.ylim([15,60])
  plt.savefig('error_5.pdf')
  
  
  
  prediction = file7['prediction']
  truth = file7['truth']
  mask = file7['mask']
  reconstruction = file7['reconstruction']
  #plt.plot(truth[:,0,0].squeeze())
  #plt.draw()
  #plt.savefig('interval.pdf')
  tmp7 = np.sum((np.asarray(prediction)-np.asarray(truth[:,20:,:]))*(np.asarray(prediction)-np.asarray(truth[:,20:,:])),axis=2)[:,0:10] 
  tmp7_r = np.sum((np.asarray(reconstruction) - np.asarray(truth[:,0:20,:]))*(np.asarray(reconstruction) - np.asarray(truth[:,0:20,:])),axis=2)
  abc=  np.sum(tmp7,axis=0)
  plt.figure()
  plt.clf()
  plt.plot(10*np.log10(1/(abc/prediction.shape[0]/prediction.shape[2])),'gs-')
  plt.draw()
  plt.ylim([15,60])
  plt.savefig('error_6.pdf')
  
  
  prediction = file8['prediction']
  truth = file8['truth']
  mask = file8['mask']
  #plt.plot(truth[:,0,0].squeeze())
  #plt.draw()
  #plt.savefig('interval.pdf')
  tmp8 = np.sum((np.asarray(prediction)-np.asarray(truth[:,20:,:]))*(np.asarray(prediction)-np.asarray(truth[:,20:,:])),axis=2)[:,0:10]
  tmp8_r = np.sum((np.asarray(reconstruction) - np.asarray(truth[:,0:20,:]))*(np.asarray(reconstruction) - np.asarray(truth[:,0:20,:])),axis=2)
  abc=  np.sum(tmp8,axis=0)
  plt.figure()
  plt.clf()
  plt.plot(10*np.log10(1/(abc/prediction.shape[0]/prediction.shape[2])),'gs-')
  plt.draw()
  plt.ylim([15,60])
  plt.savefig('error_7.pdf')
  
  
  
  min_index = np.argmin(np.vstack([tmp1.sum(axis=1),tmp2.sum(axis=1),tmp3.sum(axis=1),tmp4.sum(axis=1),tmp5.sum(axis=1),tmp6.sum(axis=1),tmp7.sum(axis=1),tmp8.sum(axis=1)]),axis=0)
  min_value = np.min(np.vstack([tmp1.sum(axis=1),tmp2.sum(axis=1),tmp3.sum(axis=1),tmp4.sum(axis=1),tmp5.sum(axis=1),tmp6.sum(axis=1),tmp7.sum(axis=1),tmp8.sum(axis=1)]),axis=0)
  all_min = np.zeros(tmp4.shape)
  scipy_io.savemat('error_index_train.mat',{'min_index':min_index,'min_value':min_value}) 
  min_index_r = np.argmin(np.vstack([tmp1_r.sum(axis=1),tmp2_r.sum(axis=1),tmp3_r.sum(axis=1),tmp4_r.sum(axis=1),tmp5_r.sum(axis=1),tmp6_r.sum(axis=1),tmp7_r.sum(axis=1),tmp8_r.sum(axis=1)]),axis=0)
 
  for i in range(all_min.shape[0]):
      if min_index[i] ==0:
          all_min[i,:] = tmp1[i,:]
      elif min_index[i] ==1:
          all_min[i,:] = tmp2[i,:]
      elif min_index[i] ==2:
          all_min[i,:] = tmp3[i,:]
      elif min_index[i] ==3 :
          all_min[i,:] = tmp4[i,:]
      elif min_index[i] ==4 :
          all_min[i,:] = tmp5[i,:]
      elif min_index[i] ==5:
          all_min[i,:] = tmp6[i,:]
      elif min_index[i] ==6:
          all_min[i,:] = tmp7[i,:]
      else :
          all_min[i,:] = tmp8[i,:]
      
  gene_error=  np.sum(all_min,axis=0)
  

  predict_index = scipy_io.loadmat('predict.mat')['predict']
  min_index_r = predict_index[0]
  for i in range(all_min.shape[0]):
      if min_index_r[i] ==0:
          all_min[i,:] = tmp1[i,:]
      elif min_index_r[i] ==1:
          all_min[i,:] = tmp2[i,:]
      elif min_index_r[i] ==2:
          all_min[i,:] = tmp3[i,:]
      elif min_index_r[i] ==3 :
          all_min[i,:] = tmp4[i,:]
      elif min_index_r[i] ==4 :
          all_min[i,:] = tmp5[i,:]
      elif min_index_r[i] ==5:
          all_min[i,:] = tmp6[i,:]
      elif min_index_r[i] ==6:
          all_min[i,:] = tmp7[i,:]
      else :
          all_min[i,:] = tmp8[i,:]
  naive_error = np.sum(all_min,axis=0)

  scipy_io.savemat('error_index_train.mat',{'min_index':min_index,'min_value':min_value})
  scipy_io.savemat('error_compare.mat',{'gene_error':gene_error,'naive_error':naive_error}) 
  abc = np.sum(min_value,axis=0)
  plt.figure()
  plt.clf()
  plt.plot(10*np.log10(1/(abc/prediction.shape[0]/prediction.shape[2])),'gs-')
  plt.draw()
  plt.ylim([15,60])
  plt.savefig('error_all.pdf')
 
  
  file_dataset = h5py.File('test_classifer.h5','w')
  file_dataset.create_dataset('label',(file1['feature'].shape[0],8),dtype='f8') 
  
  file_dataset['data'] = np.concatenate([file1['feature'],file2['feature'],file3['feature'],file4['feature'],file5['feature'],file6['feature'],file7['feature'],file8['feature']],axis=1) 
  for i in range(file1['feature'].shape[0]):
      index = np.zeros(8)
      index[min_index[i]]=1
      file_dataset['label'][i] = index

  file_dataset.close() 

  scipy_io.savemat('min_index_fut.mat',{'min_index':min_index,'min_value':min_value,'min_index_r':min_index_r})
  sys.exit()
  
  prediction = file4['prediction']
  truth = file4['truth']
  mask = file4['mask']
  tmp4 = np.sum((np.asarray(prediction)-np.asarray(truth))*(np.asarray(prediction)-np.asarray(truth)),axis=2)
  
  mask = np.matlib.repmat(np.asarray((tmp3.sum(axis=1) < tmp4.sum(axis=1))),40,1).T
  abc=  np.sum(tmp3*mask+tmp4*(1-mask),axis=0)
  plt.plot(10*np.log10(1/(abc/prediction.shape[0]/prediction.shape[2])),'k.-')
  plt.draw()
  plt.ylim([15,60])
  
  
  prediction = file5['prediction']
  truth = file5['truth']
  mask = file5['mask']
  #plt.plot(truth[:,0,0].squeeze())
  #plt.draw()
  #plt.savefig('interval.pdf')
  tmp5 = np.sum((np.asarray(prediction)-np.asarray(truth))*(np.asarray(prediction)-np.asarray(truth)),axis=2)
  
  #plt.plot(10*np.log10(1/(abc/prediction.shape[0]/prediction.shape[2])),'b.-')
  #plt.draw()
  #plt.ylim([15,60])
  
  prediction = file6['prediction']
  truth = file6['truth']
  mask = file6['mask']
  tmp6 = np.sum((np.asarray(prediction)-np.asarray(truth))*(np.asarray(prediction)-np.asarray(truth)),axis=2)
  
  mask = np.matlib.repmat(np.asarray((tmp5.sum(axis=1) < tmp6.sum(axis=1))),40,1).T
  abc=  np.sum(tmp5*mask+tmp6*(1-mask),axis=0)
  plt.plot(10*np.log10(1/(abc/prediction.shape[0]/prediction.shape[2])),'b.-')
  plt.draw()
  plt.ylim([15,60])
  
  
  
  plt.legend(['trained','inter','original'])
  
  plt.savefig('error_2.pdf')
  #
  #
