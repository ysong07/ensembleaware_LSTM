from data_handler import *
import lstm
#from mpi4py import MPI
import pdb
import scipy.io as scipy_io
from mpi4py import MPI


class LSTMCombo(object):
  def __init__(self, model):
    self.model_ = model
    
    self.lstm_stack_enc_ = lstm.LSTMStack()
    self.lstm_stack_dec_ = lstm.LSTMStack()
    self.lstm_stack_fut_ = lstm.LSTMStack()
    
    self.decoder_copy_init_state_ = model.decoder_copy_init_state
    self.future_copy_init_state_  = model.future_copy_init_state
    
    # add LSTM blocks for encoder, decoder and future predictor
    for l in model.lstm:
      self.lstm_stack_enc_.Add(lstm.LSTM(l))
    if model.dec_seq_length > 0:
      for l in model.lstm_dec:
        self.lstm_stack_dec_.Add(lstm.LSTM(l))
    if model.future_seq_length > 0:
      for l in model.lstm_future:
        self.lstm_stack_fut_.Add(lstm.LSTM(l))
    
    # do other initialization stuff
    assert model.dec_seq_length > 0 or model.future_seq_length > 0
    self.is_conditional_dec_ = model.dec_conditional
    self.is_conditional_fut_ = model.future_conditional
   
    if self.is_conditional_dec_ and model.dec_seq_length > 0:
      assert self.lstm_stack_dec_.HasInputs()
    if self.is_conditional_fut_ and model.future_seq_length > 0:
      assert self.lstm_stack_fut_.HasInputs()
   
    self.squash_relu_ = model.squash_relu
    self.binary_data_ = model.binary_data or model.squash_relu
    self.squash_relu_lambda_ = model.squash_relu_lambda
    self.relu_data_ = model.relu_data
    
    # load model if available
    if len(model.timestamp) > 0:
      old_st = model.timestamp[-1]
      ckpt = os.path.join(model.checkpoint_dir, '%s_%s.h5' % (model.name, old_st))
      f = h5py.File(ckpt)
      self.lstm_stack_enc_.Load(f)
      self.lstm_stack_dec_.Load(f)
      self.lstm_stack_fut_.Load(f)
      f.close()

  def Fprop(self, train=False):
    if self.squash_relu_:
      self.v_.apply_relu_squash(lambdaa=self.squash_relu_lambda_)
    self.lstm_stack_enc_.Reset()
    self.lstm_stack_dec_.Reset()
    self.lstm_stack_fut_.Reset()
    # Fprop through encoder.
    self.feature_ = []
    for t in xrange(self.enc_seq_length_):
      self.lstm_stack_enc_.Fprop(input_frame=self.v_.col_slice(t * self.num_dims_, (t+1) * self.num_dims_))
    
      init_state = self.lstm_stack_enc_.GetAllCurrentStates()
      temp = init_state[0].asarray()
      self.feature_.append(temp[temp.shape[0]-1,0:1000])
      #self.feature_.append(np.concatenate((init_state[0].asarray(),init_state[1].asarray()),axis=1))
    # Fprop through decoder.
    for t in xrange(self.dec_seq_length_):
      this_init_state = init_state if t == 0 else []
      if self.is_conditional_dec_ and t > 0:
        t2 = self.enc_seq_length_ - t
        input_frame=self.v_.col_slice(t2 * self.num_dims_, (t2+1) * self.num_dims_)
      else:
        input_frame = None
      self.lstm_stack_dec_.Fprop(input_frame=input_frame, init_state=this_init_state,
                                 output_frame=self.v_dec_.col_slice(t * self.num_dims_, (t+1) * self.num_dims_), copy_init_state=self.decoder_copy_init_state_)

    # Fprop through future predictor.
    for t in xrange(self.future_seq_length_):
      this_init_state = init_state if t == 0 else []
      if self.is_conditional_fut_ and t > 0:
        if train:
            t2 = self.enc_seq_length_ + t - 1
            input_frame=self.v_.col_slice(t2 * self.num_dims_, (t2+1) * self.num_dims_)
        else:
          # Instead of conditioning on true frame, condition on the generated frame at the test time
            t2 = t - 1
            input_frame=self.v_fut_.col_slice(t2 * self.num_dims_, (t2+1) * self.num_dims_)
            if self.binary_data_:
              input_frame.apply_sigmoid()
            elif self.relu_data_:
              input_frame.lower_bound(0)
      else:
        input_frame = None
      self.lstm_stack_fut_.Fprop(input_frame=input_frame, init_state=this_init_state,
                                 output_frame=self.v_fut_.col_slice(t * self.num_dims_, (t+1) * self.num_dims_), copy_init_state=self.future_copy_init_state_)
      init_state = self.lstm_stack_fut_.GetAllCurrentStates()
      #self.feature_.append(np.concatenate((init_state[0].asarray(),init_state[1].asarray()),axis=1))


    if self.binary_data_:
      if self.dec_seq_length_ > 0:
        self.v_dec_.apply_sigmoid()
      if self.future_seq_length_ > 0:
        self.v_fut_.apply_sigmoid()
    elif self.relu_data_:
      if self.dec_seq_length_ > 0:
        self.v_dec_.lower_bound(0)
      if self.future_seq_length_ > 0:
        self.v_fut_.lower_bound(0)

  def BpropAndOutp(self):
    if self.binary_data_:
      pass
    elif self.relu_data_:
      if self.dec_seq_length_ > 0:
        self.v_dec_deriv_.apply_rectified_linear_deriv(self.v_dec_)
      if self.future_seq_length_ > 0:
        self.v_fut_deriv_.apply_rectified_linear_deriv(self.v_fut_)

    init_state = self.lstm_stack_enc_.GetAllCurrentStates()
    init_deriv = self.lstm_stack_enc_.GetAllCurrentDerivs()

    # Backprop through decoder.
    for t in xrange(self.dec_seq_length_-1, -1, -1):
      this_init_state = init_state if t == 0 else []
      this_init_deriv = init_deriv if t == 0 else []
      if self.is_conditional_dec_ and t > 0:
        t2 = self.enc_seq_length_ - t
        input_frame=self.v_.col_slice(t2 * self.num_dims_, (t2+1) * self.num_dims_)
      else:
        input_frame = None
      self.lstm_stack_dec_.BpropAndOutp(input_frame=input_frame,
                                        init_state=this_init_state,
                                        init_deriv=this_init_deriv,
                                        output_deriv=self.v_dec_deriv_.col_slice(t * self.num_dims_, (t+1) * self.num_dims_), copy_init_state=self.decoder_copy_init_state_)

    # Backprop through future predictor.
    for t in xrange(self.future_seq_length_-1, -1, -1):
      this_init_state = init_state if t == 0 else []
      this_init_deriv = init_deriv if t == 0 else []
      if self.is_conditional_fut_ and t > 0:
        t2 = self.enc_seq_length_ + t - 1
        input_frame=self.v_.col_slice(t2 * self.num_dims_, (t2+1) * self.num_dims_)
      else:
        input_frame = None
      self.lstm_stack_fut_.BpropAndOutp(input_frame=input_frame,
                                        init_state=this_init_state,
                                        init_deriv=this_init_deriv,
                                        output_deriv=self.v_fut_deriv_.col_slice(t * self.num_dims_, (t+1) * self.num_dims_), copy_init_state=self.future_copy_init_state_)

    # Backprop thorough encoder.
    for t in xrange(self.enc_seq_length_-1, -1, -1):
      self.lstm_stack_enc_.BpropAndOutp(input_frame=self.v_.col_slice(t * self.num_dims_, (t+1) * self.num_dims_))

  def Update(self):
    self.lstm_stack_enc_.Update()
    self.lstm_stack_dec_.Update()
    self.lstm_stack_fut_.Update()

  def ComputeDeriv(self):
    for t in xrange(self.dec_seq_length_):
      t2 = self.enc_seq_length_ - t - 1
      dec = self.v_dec_.col_slice(t * self.num_dims_, (t+1) * self.num_dims_)
      v = self.v_.col_slice(t2 * self.num_dims_, (t2+1) * self.num_dims_)
      deriv = self.v_dec_deriv_.col_slice(t * self.num_dims_, (t+1) * self.num_dims_)
      dec.subtract(v, target=deriv)

    for t in xrange(self.future_seq_length_):
      t2 = t + self.enc_seq_length_
      f = self.v_fut_.col_slice(t * self.num_dims_, (t+1) * self.num_dims_)
      v = self.v_.col_slice(t2 * self.num_dims_, (t2+1) * self.num_dims_)
      deriv = self.v_fut_deriv_.col_slice(t * self.num_dims_, (t+1) * self.num_dims_)
      f.subtract(v, target=deriv)

  def GetLoss(self):
    for t in xrange(self.dec_seq_length_):
      t2 = self.enc_seq_length_ - t - 1
      dec = self.v_dec_.col_slice(t * self.num_dims_, (t+1) * self.num_dims_)
      v = self.v_.col_slice(t2 * self.num_dims_, (t2+1) * self.num_dims_)
      deriv = self.v_dec_deriv_.col_slice(t * self.num_dims_, (t+1) * self.num_dims_)
      
      if self.binary_data_:
        cm.cross_entropy_bernoulli(v, dec, target=deriv)
      else:
        dec.subtract(v, target=deriv)

    for t in xrange(self.future_seq_length_):
      t2 = t + self.enc_seq_length_
      f = self.v_fut_.col_slice(t * self.num_dims_, (t+1) * self.num_dims_)
      v = self.v_.col_slice(t2 * self.num_dims_, (t2+1) * self.num_dims_)
      deriv = self.v_fut_deriv_.col_slice(t * self.num_dims_, (t+1) * self.num_dims_)
      
      if self.binary_data_:
        cm.cross_entropy_bernoulli(v, f, target=deriv)
      else:
        f.subtract(v, target=deriv)

    loss_fut = 0
    loss_dec = 0
    
    if self.binary_data_:
      if self.dec_seq_length_ > 0:
        loss_dec = self.v_dec_deriv_.sum()
      if self.future_seq_length_ > 0:
        loss_fut = self.v_fut_deriv_.sum()
    else:
      if self.dec_seq_length_ > 0:
        loss_dec = 0.5 * (self.v_dec_deriv_.euclid_norm()**2)
      if self.future_seq_length_ > 0:
        loss_fut = 0.5 * (self.v_fut_deriv_.euclid_norm()**2)
    return loss_dec, loss_fut

  def Validate(self, data):
    data.Reset()
    dataset_size = data.GetDatasetSize()
    batch_size = data.GetBatchSize()
    num_batches = dataset_size / batch_size
    loss_dec = 0
    loss_fut = 0
    for ii in xrange(num_batches):
      v_cpu, _ = data.GetBatch()
      self.v_.overwrite(v_cpu)
      self.Fprop()
      this_loss_dec, this_loss_fut = self.GetLoss()
      if self.dec_seq_length_ > 0:
        loss_dec += this_loss_dec / (batch_size * self.dec_seq_length_)
      if self.future_seq_length_ > 0:
        loss_fut += this_loss_fut / (batch_size * self.future_seq_length_)

    loss_dec = loss_dec / num_batches
    loss_fut = loss_fut / num_batches
    return loss_dec, loss_fut

  def SetBatchSize(self, train_data):
   
    self.num_dims_ = train_data.GetDims()
    batch_size = train_data.GetBatchSize()
    seq_length = train_data.GetSeqLength()
    dec_seq_length    = self.model_.dec_seq_length
    future_seq_length = self.model_.future_seq_length
    assert seq_length == dec_seq_length + future_seq_length

    self.batch_size_ = batch_size
    self.enc_seq_length_    = seq_length - future_seq_length
    self.dec_seq_length_    = dec_seq_length
    self.future_seq_length_ = future_seq_length
    self.lstm_stack_enc_.SetBatchSize(batch_size, self.enc_seq_length_)
    self.v_ = cm.empty((batch_size, seq_length * self.num_dims_))
    if dec_seq_length > 0:
      self.lstm_stack_dec_.SetBatchSize(batch_size, dec_seq_length)
      self.v_dec_ = cm.empty((batch_size, dec_seq_length * self.num_dims_))
      self.v_dec_deriv_ = cm.empty((batch_size, dec_seq_length * self.num_dims_))

    if future_seq_length > 0:
      self.lstm_stack_fut_.SetBatchSize(batch_size, future_seq_length)
      self.v_fut_ = cm.empty((batch_size, future_seq_length * self.num_dims_))
      self.v_fut_deriv_ = cm.empty((batch_size, future_seq_length * self.num_dims_))

  def Save(self, model_file):
    sys.stdout.write(' Writing model to %s' % model_file)
    f = h5py.File(model_file, 'w')
    self.lstm_stack_enc_.Save(f)
    self.lstm_stack_dec_.Save(f)
    self.lstm_stack_fut_.Save(f)
    f.close()

  def Display(self, ii, fname):
    plt.figure(1)
    plt.clf()
    plt.subplot(2, 1, 1)
    plt.imshow(self.v_.asarray()[:, :1000], interpolation="nearest")
    plt.subplot(2, 1, 2)
    plt.imshow(self.v_dec_.asarray()[:, :1000], interpolation="nearest")
    plt.title('Reconstruction %d' % ii)
    plt.draw()
    #plt.pause(0.1)
    plt.savefig(fname)
  def Show_fix_range(self,data,batch_id,output_dir=None):
    self.SetBatchSize(data)
    data.Reset()
    data.Set_id(batch_id)
    try: 
        v_cpu, label,_ = data.Get_ordered_Batch()
    except:
        print "batch not valid"
    self.v_.overwrite(v_cpu)
    self.Fprop()
    rec = self.v_dec_.asarray()
    fut = self.v_fut_.asarray()
    #scipy_io.savemat('batch_data4',{'rec':rec,'fut':fut,'ori':v_cpu}) 
    file1 = scipy_io.loadmat('batch_data.mat')
    file2 = scipy_io.loadmat('batch_data2.mat')
    file3 = scipy_io.loadmat('batch_data3.mat')
    file4 = scipy_io.loadmat('batch_data4.mat')
    rec = (file1['rec']+file2['rec']+file3['rec']+file4['rec'])/4
    fut = (file1['fut']+file2['fut']+file3['fut']+file4['fut'])/4
    
    if output_dir is None:
      output_file = None
    else:
      output_file = os.path.join(output_dir)
    error_sum = 0 
    for rand_index in range(v_cpu.shape[0]):
      output_file = './imgs/Ecog_{}.pdf'.format(rand_index)
      error = data.DisplayData_Ecog(v_cpu, rec=rec, fut=fut, case_id=rand_index, output_file=output_file)
      print "error: " + str(error)
      error_sum += error
    print "sum of error: " + str(error_sum)
 
  def Show(self, data, output_dir=None):
    # get random batch from the data and displays the results
    self.SetBatchSize(data)
    data.Reset()
    v_cpu, _ = data.GetBatch()
    #rand_index = randint(0, v_cpu.shape[0] - 1)

    self.v_.overwrite(v_cpu)
    self.Fprop()
    rec = self.v_dec_.asarray()
    fut = self.v_fut_.asarray()
    # save or display the reconstructed/future predicted data
    if output_dir is None:
      output_file = None
    else:
      output_file = os.path.join(output_dir)

    #data.DisplayData(v_cpu, rec=rec, fut=fut, case_id=rand_index, output_file=output_file)
    for rand_index in range(v_cpu.shape[0]):
      output_file = './imgs/Ecog_{}.pdf'.format(rand_index)
      data.DisplayData_Ecog(v_cpu, rec=rec, fut=fut, case_id=rand_index, output_file=output_file)
  def Get_Prediction_error(self,data,save_file_name,sample_num = 2000):
    self.SetBatchSize(data)
    data.Reset()
    save_file = h5py.File(save_file_name,'w')
    #sample number: training set :15760 ,testing_set:2160
    sample_num = sample_num
   # save_file.create_dataset('prediction',(sample_num,40,360),dtype='f8')
   # save_file.create_dataset('truth',(sample_num,40,360),dtype='f8')
   # save_file.create_dataset('label',(sample_num,40,2),dtype='f4')
   # save_file.create_dataset('mask',(sample_num,40,2),dtype = 'f4')
    count = 0
    while True:
      try:
        v_cpu, label,mask = data.Get_ordered_Batch()
      except:
	break
      self.v_.overwrite(v_cpu)
      self.Fprop()
      fut = self.v_fut_.asarray()
      v_cpu = v_cpu.reshape(v_cpu.shape[0],-1,360)
      fut = fut.reshape(fut.shape[0],-1,360)
      for i in range(fut.shape[0]):
        save_file['prediction'][count] = fut[i,:,:]
        save_file['truth'][count] = v_cpu[i,v_cpu.shape[1]-fut.shape[1]:v_cpu.shape[1],:]
        test_label = np.zeros(2)
        test_label[int(label[i])] = 1
	test_mask = np.zeros(2)
	test_mask[int(mask[i])] = 1
        save_file['label'][count] = test_label
	save_file['mask'][count] = test_mask
	count = count+1
        if count >=sample_num:
          return None  
        print count
    save_file.close()

  def Extract_feature(self,data,save_file_name):
    self.SetBatchSize(data)
    data.Reset()
    list_data = []
    list_label =[]
    label =1
    save_file = h5py.File(save_file_name,'w')
    #sample number: training set :15760 ,testing_set:2160
    save_file.create_dataset('data',(8366,1000),dtype='f8')
    save_file.create_dataset('label',(8366,2),dtype='f4')
   
    count =0
    while label is not None:
      try: 
        v_cpu, label,mask = data.Get_ordered_Batch()
      except:
        break
      #print label.sum()
      self.v_.overwrite(v_cpu)
      self.Fprop()
      #list_data.append(self.feature_[:,11000:12000])
      #list_label.append(label)
      if count+self.batch_size_ >=8366:
	break
      for i in range(label.size):
        save_file['data'][count] = self.feature_[i]
	test_label = np.zeros(2)
	test_label[label[i]] = 1
	save_file['label'][count] = test_label
	count = count+1      
	print count 
        
    save_file.close()

   # output_data = list_data.pop(0)
   # output_label = list_label.pop(0)
   # 
   # for data,label in zip(list_data, list_label):
   #   if label is not None:
   #     output_data = np.concatenate([output_data,data],axis=0)
   #     output_label= np.concatenate([output_label,label],axis=0)  
   # import scipy.io as scipy_io
   # scipy_io.savemat(save_file_name,{'data':output_data,'label':output_label})	 
   #
  def RunAndShow(self, data, output_dir=None, max_dataset_size=0):
    self.SetBatchSize(data)
    data.Reset()
    dataset_size = data.GetDatasetSize()
    if max_dataset_size > 0 and dataset_size > max_dataset_size:
      dataset_size = max_dataset_size
    batch_size = data.GetBatchSize()
    num_batches = dataset_size / batch_size
    end = False
    for ii in xrange(num_batches):
      v_cpu, _ = data.GetBatch()
      self.v_.overwrite(v_cpu)
      self.Fprop()
      rec = self.v_dec_.asarray()
      fut = self.v_fut_.asarray()
      for j in xrange(batch_size):
        if j + ii * batch_size >= dataset_size:
          end = True
          break
        if output_dir is None:
          output_file = None
        else:
          output_file = os.path.join(output_dir, "%.6d.pdf" % (j + ii * batch_size))
        data.DisplayData(v_cpu, rec=rec, fut=fut, case_id=j, output_file=output_file)
      if end:
        break
  
  def Train(self, train_data, valid_data=None,rank=0):
    # Timestamp the model that we are training.
    st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d%H%M%S')
    model_file = os.path.join(self.model_.checkpoint_dir, '%s_%s' % (self.model_.name, st))
    self.model_.timestamp.append(st)
    print 'Model saved at %s.pbtxt' % model_file
    WritePbtxt(self.model_, '%s.pbtxt' % model_file)
    self.SetBatchSize(train_data)

    loss_dec = 0
    loss_fut = 0
    print_after = self.model_.print_after
    validate_after = self.model_.validate_after
    validate = validate_after > 0 and valid_data is not None
    save_after = self.model_.save_after
    save = save_after > 0
    display_after = self.model_.display_after
    display = display_after > 0

    for ii in xrange(1, self.model_.max_iters + 1):
      newline = False
      sys.stdout.write('\rStep %d' % ii)
      sys.stdout.flush()
      v_cpu, _ = train_data.GetBatch()
      self.v_.overwrite(v_cpu)
      self.Fprop(train=True)

      # Compute Performance.
      this_loss_dec, this_loss_fut = self.GetLoss()
      if self.dec_seq_length_ > 0:
        loss_dec += this_loss_dec / (self.dec_seq_length_ * self.batch_size_)
      if self.future_seq_length_ > 0:
        loss_fut += this_loss_fut / (self.future_seq_length_ * self.batch_size_)
      if self.binary_data_:
        self.ComputeDeriv()
      else:
        pass # Computing loss requires computing deriv, so ComputeDeriv is already done.
      if ii % print_after == 0:
        loss_dec /= print_after
        loss_fut /= print_after
        sys.stdout.write(' Dec %.5f Fut %.5f rank%f' % (loss_dec, loss_fut,rank))
        loss_dec = 0
        loss_fut = 0
        newline = True

      self.BpropAndOutp()
      self.Update()

      if display and ii % display_after == 0:
        #self.Display(ii, '%s_reconstruction.png' % model_file)
        fut = self.v_fut_.asarray() if self.future_seq_length_ > 0 else None
        rec = self.v_dec_.asarray() if self.dec_seq_length_ > 0 else None
        train_data.DisplayData(v_cpu, rec=rec, fut=fut)
        #self.lstm_stack_enc_.Display()
        #self.lstm_stack_dec_.Display()

      if validate and ii % validate_after == 0:
        valid_loss_dec, valid_loss_fut = self.Validate(valid_data)
        sys.stdout.write(' VDec %.5f VFut %.5f' % (valid_loss_dec, valid_loss_fut))
        newline = True

      if save and ii % save_after == 0:
        self.Save('%s.h5' % model_file)
      if newline:
        sys.stdout.write('\n')

    sys.stdout.write('\n')
  def ensemble_init(self,train_data,valid_data= None):
    self.train_data = train_data
    st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d%H%M%S')
    model_file = os.path.join(self.model_.checkpoint_dir, '%s_%s' % (self.model_.name, st))
    self.model_.timestamp.append(st)
    print 'Model saved at %s.pbtxt' % model_file
    WritePbtxt(self.model_, '%s.pbtxt' % model_file)
    self.SetBatchSize(self.train_data)
   
    loss_dec = 0
    loss_fut = 0
    print_after = self.model_.print_after
    validate_after = self.model_.validate_after
    validate = validate_after > 0 and valid_data is not None
    save_after = self.model_.save_after
    save = save_after > 0
    display_after = self.model_.display_after
    display = display_after > 0
  def ensemble_forward(self,ii,rank):
    newline = False
    #sys.stdout.write('\rStep %d' % ii)
    #sys.stdout.flush()
    v_cpu, _ = self.train_data.GetBatch()
    self.v_.overwrite(v_cpu)
    print np.array(v_cpu).sum()
    self.Fprop(train=True)
    
    # Compute Performance.
    this_loss_dec, this_loss_fut = self.GetLoss()
    #sys.stdout.write(' Dec %.5f Fut %.5f rank%f\n' % (this_loss_dec, this_loss_fut,rank))
    return this_loss_dec, this_loss_fut
def main():
  model = ReadModelProto(sys.argv[1])
  lstm_autoencoder = LSTMCombo(model)
  train_data = ChooseDataHandler(ReadDataProto(sys.argv[2]))
  valid_data = ChooseDataHandler(ReadDataProto(sys.argv[3]))
  lstm_autoencoder.Train(train_data, valid_data)

if __name__ == '__main__':
  # Set the board
  
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank() 
  board_id = 0
  # loss of sum of encoder and decoder
  loss = np.zeros(1)
  min_loss = np.zeros(1)
  min_loss_0 = np.zeros(1)
  min_loss_1 = np.zeros(1)
  min_loss_2 = np.zeros(1)
  min_loss_3 = np.zeros(1)
  sum_loss = np.zeros(1)
    
  board = LockGPU(board=0)
  model = ReadModelProto('models_33/lstm_combo_2layer_ecog_4_deploy.pbtxt')
  cm.CUDAMatrix.init_random(100)
  np.random.seed(100)
  lstm_autoencoder_0 = LSTMCombo(model)
  train_data_0 = ChooseDataHandler(ReadDataProto('datasets_33/ecog.pbtxt'))
    
  model = ReadModelProto('models_33/lstm_combo_2layer_ecog_5_deploy.pbtxt')
  cm.CUDAMatrix.init_random(100)
  np.random.seed(100)
  lstm_autoencoder_1 = LSTMCombo(model)
  train_data_1 = ChooseDataHandler(ReadDataProto('datasets_33/ecog.pbtxt'))

  model = ReadModelProto('models_33/lstm_combo_2layer_ecog_6_deploy.pbtxt')
  cm.CUDAMatrix.init_random(100)
  np.random.seed(100)
  lstm_autoencoder_2 = LSTMCombo(model)
  train_data_2 = ChooseDataHandler(ReadDataProto('datasets_33/ecog.pbtxt'))

  model = ReadModelProto('models_33/lstm_combo_2layer_ecog_7_deploy.pbtxt')
  cm.CUDAMatrix.init_random(100)
  np.random.seed(100)
  lstm_autoencoder_3 = LSTMCombo(model)
  train_data_3 = ChooseDataHandler(ReadDataProto('datasets_33/ecog.pbtxt'))
  

  lstm_autoencoder_0.ensemble_init(train_data_0)
  
  lstm_autoencoder_1.ensemble_init(train_data_1)
  lstm_autoencoder_2.ensemble_init(train_data_2)
  lstm_autoencoder_3.ensemble_init(train_data_3)
  
  comm.Barrier() 
  for ii in xrange(1, 10000000): 
    ls_dec_0, ls_fut_0 = lstm_autoencoder_0.ensemble_forward(ii,rank)
    loss_0 = np.array(ls_dec_0+ls_fut_0) 
    print "error_ensemble" +str(rank) +": " + str(loss_0)
    ls_dec_1, ls_fut_1 = lstm_autoencoder_1.ensemble_forward(ii,rank)
    loss_1 = np.array(ls_dec_1+ls_fut_1)
    print "error_ensemble" +str(rank+1) +": " + str(loss_1)
    
    ls_dec_2, ls_fut_2 = lstm_autoencoder_2.ensemble_forward(ii,rank)
    loss_2 = np.array(ls_dec_2+ls_fut_2)
    print "error_ensemble" +str(rank+2) +": " + str(loss_2)
    ls_dec_3, ls_fut_3 = lstm_autoencoder_3.ensemble_forward(ii,rank)
    loss_3 = np.array(ls_dec_3+ls_fut_3)
    print "error_ensemble" +str(rank+3) +": " + str(loss_1)


    #if ls_dec_0+ls_fut_0 < ls_dec_1 + ls_fut_1:
    #    loss = np.array(ls_dec_0+ls_fut_0)
    #    loss_id= 0
    #else:
    #    loss = np.array(ls_dec_1+ls_fut_1)
    #    loss_id= 1 
    #loss = np.min(loss_1,loss_2)
    #loss_id = np.argmin(loss_1,loss_2)
    comm.Barrier()
    comm.Allreduce(loss_0, min_loss_0, op = MPI.MIN)
    comm.Allreduce(loss_1, min_loss_1, op = MPI.MIN)
    comm.Allreduce(loss_2, min_loss_2, op = MPI.MIN)
    comm.Allreduce(loss_3, min_loss_3, op = MPI.MIN)
    min_loss = np.min([min_loss_0[0],min_loss_1[0],min_loss_2[0],min_loss_3[0]])

    #comm.Allreduce(loss,sum_loss,op=MPI.SUM)

    
    if (loss_0 == min_loss or np.random.ranf()*2<= min_loss/loss_0)  and loss_0<=2: 
      lstm_autoencoder_0.BpropAndOutp()
      lstm_autoencoder_0.Update()
      print "update " +str(rank) +'\n'
  
    if (loss_1 == min_loss or np.random.ranf()*2<= min_loss/loss_1)  and loss_1<=2:
      lstm_autoencoder_1.BpropAndOutp()
      lstm_autoencoder_1.Update()
      print "update " +str(rank+1) +'\n'
 	
    if (loss_2 == min_loss or np.random.ranf()*2<= min_loss/loss_2)  and loss_2<=2:
      lstm_autoencoder_2.BpropAndOutp()
      lstm_autoencoder_2.Update()
      print "update " +str(rank+2) +'\n' 

    if (loss_3 == min_loss or np.random.ranf()*2<= min_loss/loss_3)  and loss_3<=2:
      lstm_autoencoder_3.BpropAndOutp()
      lstm_autoencoder_3.Update()
      print "update " +str(rank+3) +'\n'  
      #if loss_id ==0:
      #  lstm_autoencoder_0.BpropAndOutp()
      #  lstm_autoencoder_0.Update()
      #  print "update " +str(rank)
      #else :
      #  lstm_autoencoder_1.BpropAndOutp()
      #  lstm_autoencoder_1.Update()
      #  print "update " +str(rank)
      #print "iteration: " + str(ii) + "loss: " + str(loss)
    comm.Barrier() 
    if ii %20000 ==5:
	if rank == 0:
	    lstm_autoencoder_0.Save('models_33/train_4/ensemble_MPI_%s' % ii)
 	    lstm_autoencoder_1.Save('models_33/train_5/ensemble_MPI_%s' % ii)
	    lstm_autoencoder_2.Save('models_33/train_6/ensemble_MPI_%s' % ii)
            lstm_autoencoder_3.Save('models_33/train_7/ensemble_MPI_%s' % ii) 
        #elif rank==1 :
        #    lstm_autoencoder_0.Save('models_joint_4041_ensemble/train_1/ensemble_MPI_%s' % ii)
	#    lstm_autoencoder_1.Save('models_joint_4041_ensemble/train_5/ensemble_MPI_%s' % ii)
        #elif rank==2 :
        #    lstm_autoencoder_0.Save('models_joint_4041_ensemble/train_2/ensemble_MPI_%s' % ii)
        #    lstm_autoencoder_1.Save('models_joint_4041_ensemble/train_6/ensemble_MPI_%s' % ii)
	#elif rank==3 :
	#    lstm_autoencoder_0.Save('models_joint_4041_ensemble/train_3/ensemble_MPI_%s' % ii)
	#    lstm_autoencoder_1.Save('models_joint_4041_ensemble/train_7/ensemble_MPI_%s' % ii)
    comm.Barrier()  
#ls_dec_2, ls_fut_2 = lstm_autoencoder_2.ensemble_forward()
  #ls_dec_3, ls_fut_3 = lstm_autoencoder_3.ensemble_forward()
  
    
  
#  print 'Using board', board

  #cm.CUDAMatrix.init_random(42)
  #np.random.seed(42)
  #cm.CUDAMatrix.init_random(1)
  #np.random.seed(1)
  #cm.CUDAMatrix.init_random(10)
  #np.random.seed(10)
  #cm.CUDAMatrix.init_random(100)
  #np.random.seed(100)
  #cm.CUDAMatrix.init_random(200)
  #np.random.seed(200)
