import numpy as np
import h5py
from sklearn.svm import SVC
import pdb
from sklearn.cross_validation import train_test_split
#file = h5py.File('train.h5')
#X_train = np.array(file['data'])
#y_train = np.array(file['label']).argmax(axis=1)
#temp_data = np.zeros(X_train.shape)
#for i in range(100):
#    temp_data+= np.roll(X_train,i,axis=0)
#    print i
#
#new_file = h5py.File('train_circular_100.h5','w')
#new_file.create_dataset('data',data= temp_data)
#new_file.create_dataset('label',data = y_train)
#new_file.close()


file = h5py.File('train_circular_100.h5')
X = np.array(file['data'])
y = np.array(file['label'])
#pdb.set_trace()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1)
clf= SVC(kernel='linear')
clf.fit(X_train,y_train)

print clf.score(X_test,y_test)
