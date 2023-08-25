import math 
import numpy as np
import tensorflow as tf

class BaseGenerator(tf.keras.utils.Sequence): 
  """
  A sequence 
  """  
  
  def __init__(self, batch_size, training, n_items):
  
    self.n_items = n_items
    self.n_batches = int(math.ceil(n_items/batch_size))
    self.batch = 0
    self.epoch = 0
    self._iterator = None
    self.batch_size = batch_size
    self.training = training
    self.idx = 0
    self.n_items = n_items
    
    outputs = next(self._iter_func())
    
    self.output_shapes = []
    self.output_signature = []
    
    for x in outputs:
      if isinstance(x, (tuple, list)):
        self.output_shapes.append(tuple([y.shape for y in x]))
        self.output_signature.append(tuple([tf.TensorSpec(shape=y.shape, dtype=getattr(tf, str(y.dtype))) for y in x]))
      else:
        self.output_shapes.append(x.shape)
        self.output_signature.append(tf.TensorSpec(shape=x.shape, dtype=getattr(tf, str(x.dtype))))
    
    self.output_signature = tuple(self.output_signature)
    self.output_shapes = tuple(self.output_shapes)
    
    
  def dataset(self, cntx=None):
    
    dataset = tf.data.Dataset.from_generator(self, output_signature=self.output_signature)
    
    return dataset
    
    
  def partitioned_dataset(self, strategy):
  
     #return strategy.experimental_distribute_dataset(self.dataset())
     
     return strategy.distribute_datasets_from_function(self.dataset)
     
          
  def _iter_func(self):
    """
    Overwrite in subclass
    """
    pass
  
    
  def __len__(self):
    
    return self.n_batches
  
    
  def __getitem__(self, idx):
    
    #print('__getitem__', self.idx, self.n_batches)
   
    if self._iterator is None:
      self._iterator = self._iter_func()
      
      
    data_items = next(self._iterator)    
    self.idx += 1
    
    return data_items #x_inputs, y_true, weights    
        
              
  def on_epoch_end(self):
    
    self.idx = 0
    self.batch = 0
    self.epoch += 1
    self._iterator = None


  def __call__(self):
    
    for i in range(self.n_batches):
      yield self.__getitem__(self.idx)
    
 
