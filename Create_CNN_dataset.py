import numpy as np
import h5py
import pdb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ion()
# train feature is for dataset 4041 with half overlapping window, window size is 20 for LSTM auto-encoder
file = h5py.File('train_feature_4041','r')
data = np.asarray(file['data'])
label = np.asarray(file['label'])
data_mean = np.mean(data,axis = 0)
print data.shape
pdb.set_trace()
data = data - data_mean

num_sample = 4189-1000
count_file_num = 0
new_file_flag = True
with open('test_list.h5list','w') as txt_file:
    for i in range(data.shape[0] -1000):     
    #for i in range(2000):
        if new_file_flag == True:
            save_file_name = './dataset_4041_cnn/ecog_valid' +str(count_file_num) 
            save_file = h5py.File(save_file_name,'w')
            save_file.create_dataset('data',(num_sample,1000,1000),dtype='f8')
            save_file.create_dataset('label',(num_sample,2),dtype='f4')
            txt_file.write(save_file_name+'\n')
	    count = 0
	    new_file_flag = False
        save_file['data'][count] = data[i:i+1000,:]
        save_file['label'][count] = label[i+1000,:]
	count = count+1

	#if count >=999:
	#    save_file.close()
	#    count_file_num = count_file_num+1
	#    new_file_flag = True
	#    print count_file_num

