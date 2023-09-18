import os, math, time, sys
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from inspect import signature
from collections import defaultdict

def time_str(secs):
 
  if secs < 500:
    return f'{secs:5.1f}s'

  else:
    mins, secs = divmod(int(secs), 60)
 
    if mins > 60:
      hours, mins = divmod(mins, 60)
      
      if hours > 48:
        days, hours =  divmod(hours, 24)
        return f'{days}d{hours}h{mins}m{secs}s'
      
      else:
        return f'{hours}h{mins}m{secs}s'
        
    else:
      return f'{mins}m{secs}s'


class ModelManager(object):
  """
  Main Model Manager class
  
  """
  
  _prev_print_width = 0
  
  def __init__(self, n_gpu, report_interval=1.0):
    
    self.report_interval = report_interval
    self.loss_metrics = None
    self.acc_metrics = None
    self.report_line1 = None
    self.report_line2 = None
    self.strategy = None
    self.num_replicas = None
    self.global_batch_size = None
    self.n_batches = None
    self.epoch = 0
    self.n_epoces = None
    
    self.data_generator = self.get_generator()
    self.data_generator_test = self.get_generator(training=False)
    
    self.init_strategy(n_gpu)
    
    with self.strategy.scope():
      self.init_metrics()
  
  
  def _print(self, msg):
    
    if msg.endswith('\r'):
      msg = msg[:-1].ljust(self._prev_print_width) + '\r' # Extend to cover prev
      self._prev_print_width = len(msg)    
      sys.stdout.write(msg)
      sys.stdout.flush()
 
    else:
      print(msg)
      self._prev_print_width = 0
 
 
  def info(self, msg):

    _print('INFO: ' + msg)
 
 
  def warn(self, msg):

    _print('WARN: ' + msg)


  def critical(self, msg):

    _print('STOP')
    _print('EXIT: ' + msg)
    sys.exit(0)
  
      
  def plot_training_history(self, *histories, file_path=None):
    
    from matplotlib import pyplot as plt
    from matplotlib import cm
    
    def _get_line(label, vals):
      texts = [label] + ['%.3e' % x for x in vals]
      return '\t'.join(texts) + '\n'
 
    cmap = cm.get_cmap('rainbow')
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(16,8)
    plot_options = {'linewidth':2, 'alpha':0.5} # for all charts
 
    color_dict = {'accuracy':'#FF2000','recall':'#BBBB00','precision':'#0080FF'}
 
    if file_path:
      table_file = os.path.splitext(file_path)[0] + '.tsv'
    else:
      table_file = 'training_history.tsv'
 
    with open(table_file, 'w') as out_file_obj:
      write = out_file_obj.write
 
      for i, history in enumerate(histories):
        if isinstance(history, dict):
          hd = history
        else:
          hd = history.history
 
        n = len(hd['loss'])
        epochs = np.arange(n) + 1
        m = len(hd)
 
        plot_options['color'] = cmap(float(i % 10)/10)
 
        ax1.plot(epochs, hd['loss'], label='Train %d' % i,
                 linestyle='--', **plot_options)
        ax1.plot(epochs, hd['val_loss'], label='Test %d' % i,
                 **plot_options)
        ax1.set_title('Loss')
        ax1.set_xlabel('Iteration')
 
        write(_get_line('loss', hd['loss']))
        write(_get_line('val_loss', hd['val_loss']))
 
 
        for j, metric in enumerate(hd):
 
          if 'loss' in metric:
            continue
 
          if 'val_' in metric:
            linestyle = '-'
            set_type = 'Test'
            met_name = metric[4:]
          else:
            linestyle = '--'
            set_type = 'Train'
            met_name = metric
 
          if i > 0:
            label='%s %s %d' % (set_type, met_name, i)
          else:
            label='%s %s' % (set_type, met_name)
 
          plot_options['color'] = cmap(float(j % m)/m)  # color_dict.get(met_name)
          ax2.plot(epochs, hd[metric], label=label,  linestyle=linestyle, **plot_options)
          write(_get_line(metric, hd[metric]))
 
        ax2.set_title('Accuracy etc.')
        ax2.set_xlabel('Iteration')
        ax2.set_yticks(np.arange(0, 1.01, 0.05))
        ax2.set_xticks(np.arange(0, n, 10))

      ax1.legend()
      ax2.legend()
 
      ax1.grid(True, linewidth=0.5, alpha=0.5)
      ax2.grid(True, linewidth=0.5, alpha=0.5)
 
      if file_path:
        plt.savefig(file_path, dpi=300)
      else:
        plt.show()

  def checkpoint(self, model, model_path, epoch):
    """Overwrite in subclass"""
    
    if epoch and (epoch % 10 == 0):
      file_root, file_ext = os.path.splitext(model_path)
      save_path = f'{file_root}_EP{epoch+1}{file_ext}'
      model.save_weights(save_path)
      self._print(f'Checkpoint {save_path}')
  
  
  def get_generator(self, training=False, data_source=None):
    """Overwrite in subclass"""
    
    return None    
    
    
  def get_model(self):
    """Overwrite in subclass"""
    pass    
  
  
  def get_optimizer(self):
    """Overwrite in subclass"""
    pass    
    
    
  def get_loss_func(self):
    """Overwrite in subclass"""
    pass 
  
  
  def get_acc_func(self):
    """Overwrite in subclass"""
    pass 
  
  
  def get_acc_metric_classes(self):
    """Overwrite in subclass if get_acc_processor_func() is set"""
    
    return [keras.metrics.Mean for x in self.get_acc_names()]

  
  def get_acc_processor_func(self):
    """Overwrite in subclass"""
    return None
  
  
  def get_acc_names(self):
    """Overwrite in subclass"""
    pass   


  def get_loss_names(self):
    """Overwrite in subclass"""
    
    return ('loss',) 
    
     
  def positional_encoding(self, batch_size, seq_width, depth):
 
    pos_encoding = np.empty((batch_size, seq_width, depth), np.float32)
 
    positions = np.arange(0, seq_width)[:,None]
 
    angle_rates = np.arange(0.0, depth) * -(math.log(10000.0) / depth)
 
    pos_terms = positions * np.exp(angle_rates)

    # apply sin to even indices in the array; 2i
    pos_encoding[:, :, 0::2] = np.sin(pos_terms[:,0::2])

    # apply cos to odd indices in the array; 2i+1
    pos_encoding[:, :, 1::2] = np.cos(pos_terms[:,1::2])

    return tf.constant(pos_encoding, dtype=tf.float32)


  def normal_noise(self, x, stddev=0.05):
 
    is_not_mask = tf.cast(x != 0.0, dtype=x.dtype)
 
    # Noise is zero where blank
    noise = is_not_mask * tf.random.normal(tf.shape(x), 0.0, stddev, dtype=x.dtype)
 
    return x + noise


  def do_nothing(self, x):
 
    return x


  def attention_block(self, query, key, attn_heads, attn_embed_dim, attn_ff_depth, name='A', dropout_frac=0.1,
                      attention_mask=None, dens_params={'activation':'gelu', 'kernel_initializer': 'he_normal'}):

    atten1 = layers.MultiHeadAttention(attn_heads, attn_embed_dim, dropout=dropout_frac, name=f'att_{name}')
    att_norm1a = layers.LayerNormalization(name=f'att_norm1{name}')
    att_norm1b = layers.LayerNormalization(name=f'att_norm2{name}')
    att_dens1a = layers.Dense(attn_ff_depth, name=f'att_ff1{name}', **dens_params)
    att_dens1b = layers.Dense(attn_embed_dim, name=f'att_ff2{name}') # Simple, linear
 
    x = atten1(query, key, attention_mask=attention_mask, return_attention_scores=False)
    #x, scores = atten1(query, key, attention_mask=attention_mask, return_attention_scores=True)
 
    x = layers.Add()([x, query])
    att_out = att_norm1a(x)
 
    x = att_norm1b(layers.Add()([att_dens1b(att_dens1a(att_out)), att_out]))

    return x 

 
  def init_strategy(self, n_gpu=None):

    gpus = tf.config.list_physical_devices('GPU')
    n_avail = len(gpus)
 
    if not n_gpu:
      n_gpu = n_avail
 
    if n_gpu > 1:
      gpus = [x.name for x in gpus]
      self.strategy = tf.distribute.MirroredStrategy() # devices=gpus[:n_gpu])
      self.num_replicas = self.strategy.num_replicas_in_sync
      self.global_batch_size = self.data_generator.batch_size * self.num_replicas
 
    else:
      self.strategy = tf.distribute.get_strategy()
      self.num_replicas = 1
      self.global_batch_size = self.data_generator.batch_size
 
    self.n_batches = int(math.ceil(self.data_generator.n_batches // self.num_replicas))
 
    self._print(f'Num devices/GPUs used: {self.num_replicas}')
    self._print(f'Batches: {self.n_batches:,} of global size {self.global_batch_size:,} from {self.data_generator.n_items:,} items')
  
  
  def init_metrics(self):
    
    loss_names = self.get_loss_names()
    acc_names = self.get_acc_names()
    
    loss_str = ' '.join(['%s:{:7.5f}' % name for name in loss_names])
    acc_str = ' '.join(['%s:{:5.3f}' % name for name in acc_names])
 
    self.report_line1 = 'EP:{:3d}/{:3d} B:{:3d}/{:3d} T:{}/{} ' + loss_str + ' ' + acc_str
    self.report_line2 = self.report_line1 + ' VAL: ' + loss_str + ' ' + acc_str + ' dT:{:5.1f}ms'
    
    loss_met = keras.metrics.Mean
    
    self.loss_metrics = [(loss_met(name=mn), loss_met(name='val_'+mn)) for mn in loss_names]
    
    metric_classes = self.get_acc_metric_classes()
     
    self.acc_metrics  = [(mc(name=f'am{i}'), mc(name=f'val_am{i}')) for i, mc in enumerate(metric_classes)]
    
  
  def _report(self, epoch, batch, t_taken, disp_time, mean_dt=None):
     
     acc_processor = self.get_acc_processor_func()
     acc_results = [m[0].result() for m in self.acc_metrics]
     
     if acc_processor:
       acc_results = list(acc_processor(*acc_results))
     
     batch_info = [epoch+1, self.n_epochs, batch+1, self.n_batches, time_str(t_taken), time_str(disp_time)]
     batch_info += [m[0].result() for m in self.loss_metrics] + acc_results
     
     if mean_dt: # Test/validation
       acc_results = [m[1].result() for m in self.acc_metrics]
       
       if acc_processor:
         acc_results = list(acc_processor(*acc_results))
          
       batch_info += [m[1].result() for m in self.loss_metrics] + acc_results + [mean_dt]
       self._print(self.report_line2.format(*batch_info))
     
     else:
       self._print(self.report_line1.format(*batch_info) + '\r')
  
  
  def transfer_compatible_weights(self, load_path, save_path):
    
    model = self.get_model()
    model.load_weights(load_path, by_name=True, skip_mismatch=True)
    model.save_weights(save_path)
   
    self._print(f'Transferred compatible weights from {load_path} to {save_path}')
       
  
  def infer(self, data_source, model_path):
    
    data_generator = self.get_generator(training=False, data_source=data_source)
    n = data_generator.n_items
    batch_size = data_generator.batch_size
 
    self._print(f'Making inference for {n:,} items\n')
    model = self.get_model()
    model.load_weights(model_path)
    pred_out = None
    true_out = None
    multi_pred = None
    multi_true = None
    
    i = 0
    for batch, (x_in, y_true, weights) in enumerate(data_generator):
      self._print(f' .. {i:,}\r')
      j = min(n, i+batch_size)
      y_pred = model(x_in, training=False)
      
      if batch == 0:
        if isinstance(y_pred, (tuple, list)): # Can have more outputs than true comparisons
          pred_out = [np.zeros((n,) + y.shape[1:]) for y in y_pred]
          multi_pred = True
        else:
          pred_out = np.zeros((n,) + y_pred.shape[1:])
          multi_pred = False
      
        if isinstance(y_true, (tuple, list)):
          true_out = [np.zeros((n,) + y.shape[1:]) for y in y_true]
          multi_true = True
        else:
          true_out = np.zeros((n,) + y_true.shape[1:])
          multi_true = False
       
      if multi_true:
        for k, y in enumerate(y_true):
          pred_out[k][i:j] = y[:j-i]
      
      else:
        true_out[i:j] = y_true[:j-i]
      
      if multi_pred:
        for k, y in enumerate(y_pred):
          pred_out[k][i:j] = y[:j-i]
      
      else:
        pred_out[i:j] = y_pred[:j-i]

      i = j
    
    self._print(f'.. done {i}\n')
    return pred_out, true_out
   
   
  def train(self, n_epochs, model_path):
    
    self.n_epochs = n_epochs
    
    acc_func      = self.get_acc_func()
    acc_metrics  = self.acc_metrics
    acc_n_params = len(signature(acc_func).parameters)
    loss_func    = self.get_loss_func()
    loss_metric  = self.loss_metrics[0]
    n_batches    = self.n_batches
    num_replicas = self.num_replicas
    strategy     = self.strategy
    
    with strategy.scope():
      def global_loss_func(y_true, y_pred, weights):
        part_loss = loss_func(y_true, y_pred, weights)
        loss = tf.nn.compute_average_loss(part_loss, global_batch_size=self.global_batch_size, sample_weight=weights)
        #loss += tf.nn.scale_regularization_loss(tf.add_n([weights]))
 
        return loss
        
      calc_loss = global_loss_func if num_replicas > 1 else loss_func
      
      model = self.get_model()
 
      if os.path.exists(model_path):
        self._print(f'Loading {model_path}')
        model.load_weights(model_path)
 
      optimizer = self.get_optimizer()
      train_dataset = self.data_generator.partitioned_dataset(strategy)
      test_dataset  = self.data_generator_test.partitioned_dataset(strategy)
 
    all_metrics = self.loss_metrics + self.acc_metrics
 
    @tf.function
    def test_train_step(x_in, y_true, weights, training=True):
      """Per replica"""
      
      if training:
        j = 0
        with tf.GradientTape() as tape:
          y_pred = model(x_in, training=training)
          loss = calc_loss(y_true, y_pred, weights)

        tvs = model.trainable_variables
        grads = tape.gradient(loss, tvs)
        optimizer.apply_gradients(zip(grads, tvs))
      
      else:
        j = 1
        y_pred = model(x_in, training=training)
        loss = calc_loss(y_true, y_pred, weights)
 
      loss /= num_replicas
      loss_metric[j](loss)
 
      if acc_n_params == 4:
        vals = acc_func(x_in, y_true, y_pred, weights)
      else:
        vals = acc_func(y_true, y_pred, weights)
 
      for v, acc_metric in enumerate(acc_metrics):
        acc_metric[j](vals[v])

      return loss

    @tf.function
    def distrib_test_train_step(x_in, y_true, weights, training=True):
      """Global : combined replicas"""
      losses = strategy.run(test_train_step, args=(x_in, y_true, weights, training))
      return strategy.reduce(tf.distribute.ReduceOp.SUM, losses, axis=None)
 
    mean_dt = 0
    history = defaultdict(list)

    #self.data_generator.on_epoch_end()
    #self.data_generator_test.on_epoch_end()
    interval = self.report_interval
    
    for epoch in range(n_epochs):
 
      for m1, m2 in all_metrics:
        m1.reset_states()
        m2.reset_states()
 
      n_steps = 0.0
      t_taken = 0.0
      t_prev  = 0.0
      t_first = 0.0
 
      for batch, (x_in, y_true, weights) in enumerate(train_dataset):
        if batch == 0:
          epoch_start = time.time() # After generator initialisation
          
        loss = distrib_test_train_step(x_in, y_true, weights)        
        
        # Report times and stats
        n_steps += 1.0
        batch_end = time.time()
 
        if batch_end > (t_prev+interval):
          t_taken = batch_end-epoch_start
 
          if t_first:
            batch_time = (t_taken-t_first)/(n_steps-1.0)
            disp_time = t_first + (n_batches-1) * (batch_time)
          else:
            t_first = t_taken
            batch_time = t_taken / n_steps
            disp_time = n_batches * batch_time # Est total
          
          t_prev = batch_end
 
        self._report(epoch, batch, t_taken, disp_time)

      t_taken = time.time()-epoch_start
      mean_dt = int(1e3 * t_taken/n_steps) # Miliseconds
      self.data_generator.on_epoch_end()

      # Test
      v_time = 0.0
      test_loss = 0.0
      for batch, (x_in, y_true, weights) in enumerate(test_dataset):
        loss = distrib_test_train_step(x_in, y_true, weights, training=False)
        
        v_time = time.time()-epoch_start
        v_time -= t_taken
        #self._report(epoch, batch, v_time, t_taken, mean_dt)
        
      self.checkpoint(model, model_path, epoch)
      self._report(epoch, batch, v_time, t_taken, mean_dt)
      self.data_generator_test.on_epoch_end()
 
      for m1, m2 in all_metrics:
        history[m1.name].append(m1.result())
        history[m2.name].append(m2.result())
        m1.reset_states()
        m2.reset_states()

    self._print('Finalizing')

    model.save_weights(model_path)
 
    model_path_root = os.path.splitext(model_path)[0]
 
    history_path = model_path_root + '_training.png'
 
    self.plot_training_history(history, file_path=history_path)


   

