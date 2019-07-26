# 迁移原生TensorFlow应用到ModelArts平台的horovod



### 将数据下载入容器

- 在数据加载前，使用mox.file.copy_parallel接口将obs上的数据下载到容器/cache目录
- 训练过程中，将输出的日志文件、模型文件等保存在容器的/cache目录下
- 训练结束后，将保存在/cache目录下的日志文件、模型文件通过mox.file.copy_parallel接口拷贝回到obs目录

```
copy_parallel(src_url, dst_url, file_list=None, threads=16,
                  is_processing=True, use_queue=True):
  """
  Copy all files in src_url to dst_url. Same usage as `shutil.copytree`.
  Note that this method can only copy a directory. If you want to copy a single file,
  please use `mox.file.copy`

  Example::

    copy_parallel(src_url='/tmp', dst_url='s3://bucket_name/my_data')

  :param src_url: Source path or s3 url
  :param dst_url: Destination path or s3 url
  :param file_list: A list of relative path to `src_url` of files need to be copied.
  :param threads: Number of threads or processings in Pool.
  :param is_processing: If True, multiprocessing is used. If False, multithreading is used.
  :param use_queue: Whether use queue to manage downloading list.
  :return: None
  """
```

### 训练过程中直接访问OBS

通过moxing.file.read接口从OBS读取文件流，再做下一步处理（cv2.imdecode等）

```
"""
  Read all data from file in OBS or local.
  Same usage as open(url, 'r').read()

  Example::

    import moxing as mox
    image_buf = mox.file.read('/home/username/x.jpg', binary=True)

    # check file exist or not while using r or r+ mode. No checking for default.
    image_buf = mox.file.read('/home/username/x.jpg', binary=True, exist_check=True)

  :param url: Path or s3 url to the file.
  :param client_id: String id of specified obs client.
  :param binary: Whether to read the file as binary mode.
  :return: data string.
  """
```

### Tensorflow原生应用

这块代码为TensorFlow原生应用，使用的是ResNet-50 的图像分类模型，其中CIFAR数据集下载地址为：[CIFAR数据集](http://www.cs.toronto.edu/~kriz/cifar.html)，选择python版本即可。

```python
import numpy as np
import os
import tensorflow as tf
import random
random.seed(20190725)
from tensorflow.contrib.slim import nets
slim = tf.contrib.slim
tf.logging.set_verbosity(tf.logging.INFO)
class cifar_generator( object):
    def __init__(self, path, num_sample=50000):
        self.num_sample = num_sample
        data = self.read_data(path)
        self.y_train = data['y_train']
        self.X_test = data['X_test'].transpose(0, 2, 3, 1)
        self.y_test = data['y_test']
        self.X_train = data['X_train'].transpose(0, 2, 3, 1)
        self.idx = 0

    def load_CIFAR_batch(self, filename):
        import pickle
        with open(filename, 'rb') as f:
            datadict = pickle.load(f, encoding='bytes')
            X = datadict[b'data']
            Y = datadict[b'labels']
            X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
            Y = np.array(Y)
        return X, Y

    def load_CIFAR10(self, ROOT):
        xs = []
        ys = []
        for b in range(1, 6):
            f = os.path.join(ROOT, 'data_batch_%d' % (b,))
            X, Y = self.load_CIFAR_batch(f)
            xs.append(X)
            ys.append(Y)
        Xtr = np.concatenate(xs)
        Ytr = np.concatenate(ys)
        del X, Y
        Xte, Yte = self.load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
        return Xtr, Ytr, Xte, Yte

    def read_data(self, cifar10_dir, num_test=1000):
        X_train, y_train, X_test, y_test = self.load_CIFAR10(cifar10_dir)
        # Subsample the data
        mask = list(range(self.num_sample))
        random.shuffle(mask)
        X_train = X_train[mask]
        y_train = y_train[mask]
        mask = range(num_test)
        X_test = X_test[mask]
        y_test = y_test[mask]

        mean_image = np.mean(X_train, axis=0)
        X_train -= mean_image
        X_test -= mean_image

        X_train = X_train.transpose(0, 3, 1, 2).copy()
        X_test = X_test.transpose(0, 3, 1, 2).copy()

        return {
            'X_train': X_train, 'y_train': y_train,
            'X_test': X_test, 'y_test': y_test,
        }

    def reset(self):
        mask = list(range(self.num_sample))
        random.shuffle(mask)
        self.X_train = self.X_train[mask]
        self.y_train = self.y_train[mask]
        self.idx = 0

    def get_next_batch(self, batch_size=128):
        if self.idx * batch_size >= self.num_sample:
            self.reset()
        batch_images = self.X_train[self.idx * batch_size:self.idx * batch_size + batch_size]
        batch_labels = self.y_train[self.idx * batch_size:self.idx * batch_size + batch_size]
        self.idx += 1
        return batch_images, batch_labels

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    batch_size = 128
    num_classes = 10
    num_steps = 10000
    model_save_path = './checkpoints3'
    cifar_path = '/data/cifar-10-batches-py'
    data_generator = cifar_generator(cifar_path)

    inputs = tf.placeholder(tf.float32, shape=[None, 32, 32, 3], name='inputs')
    labels = tf.placeholder(tf.int32, shape=[None], name='labels')
    is_training = tf.placeholder(tf.bool, name='is_training')
    with slim.arg_scope(nets.resnet_v1.resnet_arg_scope()):
        net, endpoints = nets.resnet_v1.resnet_v1_50(inputs, num_classes=None,
                                                     is_training=is_training)
    with tf.variable_scope('Logits'):
        net = tf.squeeze(net, axis=[1, 2])
        net = slim.dropout(net, keep_prob=0.5, scope='scope')
        logits = slim.fully_connected(net, num_outputs=num_classes,
                                      activation_fn=None, scope='fc')

    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits)
    loss = tf.reduce_mean(losses)

    logits = tf.nn.softmax(logits)
    classes = tf.argmax(logits, axis=1, name='classes')
    accuracy = tf.reduce_mean(tf.cast(
        tf.equal(tf.cast(classes, dtype=tf.int32), labels), dtype=tf.float32))
    lr = 0.0002
    optimizer = tf.train.AdamOptimizer(lr, beta1=0.5)
    global_step = tf.train.get_or_create_global_step()
    train_step = optimizer.minimize(loss, global_step=global_step)

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.95
    hooks = [
        tf.train.StopAtStepHook(last_step=num_steps),
        tf.train.LoggingTensorHook(tensors={'step': global_step,
                                            'loss': loss,
                                            'acc': accuracy},
                                   every_n_iter=1)
    ]
    with tf.train.MonitoredTrainingSession(checkpoint_dir=model_save_path,
                                           hooks=hooks,
                                           config=config) as mon_sess:
        while not mon_sess.should_stop():
            image_, label_ = data_generator.get_next_batch()
            mon_sess.run(train_step, feed_dict={inputs: image_,
                                                labels: label_,
                                                is_training: True})

```



### Tensorflow原生应用向Horovod迁移

迁移到Horovod需要修改的地方如下：

- 在ModelArts上训练时需要在第一个进程下载数据，完成后所有进程同步运行：

  ```python
  local_data_url = '/home/tcd/train_dir/data/'
  if hvd.local_rank() == 0:
  	mox.file.copy_parallel(flags.data_url, local_data_url)
  hvd._allreduce([0], name="Barrier")
  ```

- **切分数据集**并分配给每个进程，每个进程使用不同的数据块，示例如下：

  ```python
  block_sample = num_sample // hvd.size()
  start_point = hvd.local_rank() * block_sample
  end_point = hvd.local_rank() * block_sample + block_sample
  train_mask = list(range(start_point, end_point))
  random.shuffle(train_mask)
  X_train = X_train[train_mask]
  y_train = y_train[train_mask]
  ```

- 由于数据的切分，**训练总的step数量**需要除以进程数：**num_steps // hvd.size()**；

- 分布式多卡训练时，**学习率的大小**需要随着GPU数量的变化而变化：**lr = 0.0002 * hvd.size()**；

- 由于多进程同步训练模型，需要设置**全局step**：

  ```python
  global_step = tf.train.get_or_create_global_step()
  train_step = optimizer.minimize(loss, global_step=global_step)
  ```

- 为使得多个进程的模型参数能同步更新需要增加 **hvd.BroadcastGlobalVariables**设置：

  ```python
  hooks = [
          hvd.BroadcastGlobalVariablesHook(0),
          tf.train.StopAtStepHook(last_step=num_steps // hvd.size()),
          tf.train.LoggingTensorHook(tensors={'step': global_step,
                                              'loss': loss,
                                              'acc': accuracy},
                                     every_n_iter=10),
  ]
  ```

- 模型训练完成后，需要从训练环境拷贝到OBS路径：

  ```python
  if hvd.local_rank() == 0:
      mox.file.copy_parallel(model_save_path, flags.train_url)
  ```



  ##### Horovod迁移成功的代码，可以直接支持分布式训练：

```python
import numpy as np
import os
import tensorflow as tf
import random
import horovod.tensorflow as hvd
import moxing.tensorflow as mox
from tensorflow.contrib.slim import nets
slim = tf.contrib.slim

tf.flags.DEFINE_string('data_url', '', 'Directory for storing input data')
tf.flags.DEFINE_string('train_url', '', 'Train Url')
flags = tf.flags.FLAGS

class cifar_generator( object):
    def __init__(self, path, num_sample=50000, local_rank=0, hvd_size=1):
        self.num_sample = num_sample
        self.local_rank = local_rank
        self.block_sample = num_sample // hvd_size
        start_point = self.local_rank * self.block_sample
        end_point = self.local_rank * self.block_sample + self.block_sample
        self.train_mask = list(range(start_point, end_point))
        data = self.read_data(path)
        self.y_train = data['y_train']
        self.X_test = data['X_test'].transpose(0, 2, 3, 1)
        self.y_test = data['y_test']
        self.X_train = data['X_train'].transpose(0, 2, 3, 1)
        self.idx = 0

    def load_CIFAR_batch(self, filename):
        import pickle
        with open(filename, 'rb') as f:
            datadict = pickle.load(f, encoding='bytes')
            X = datadict[b'data']
            Y = datadict[b'labels']
            X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
            Y = np.array(Y)
        return X, Y

    def load_CIFAR10(self, ROOT):
        xs = []
        ys = []
        import time
        time.sleep(self.local_rank * 1)
        for b in range(1, 6):
            f = os.path.join(ROOT, 'data_batch_%d' % (b,))
            X, Y = self.load_CIFAR_batch(f)
            xs.append(X)
            ys.append(Y)
        Xtr = np.concatenate(xs)
        Ytr = np.concatenate(ys)
        del X, Y
        Xte, Yte = self.load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
        return Xtr, Ytr, Xte, Yte

    def read_data(self, cifar10_dir, num_test=1000):
        X_train, y_train, X_test, y_test = self.load_CIFAR10(cifar10_dir)
        # Subsample the data
        random.shuffle(self.train_mask)
        X_train = X_train[self.train_mask]
        y_train = y_train[self.train_mask]
        mask = range(num_test)
        X_test = X_test[mask]
        y_test = y_test[mask]

        mean_image = np.mean(X_train, axis=0)
        X_train -= mean_image
        X_test -= mean_image

        X_train = X_train.transpose(0, 3, 1, 2).copy()
        X_test = X_test.transpose(0, 3, 1, 2).copy()

        return {
            'X_train': X_train, 'y_train': y_train,
            'X_test': X_test, 'y_test': y_test,
        }

    def reset(self):
        self.train_mask = list(range(self.block_sample))
        random.shuffle(self.train_mask)
        self.X_train = self.X_train[self.train_mask]
        self.y_train = self.y_train[self.train_mask]
        self.idx = 0

    def get_next_batch(self, batch_size=128):
        if self.idx * batch_size >= self.block_sample:
            self.reset()
            epoch_end = True
        batch_images = self.X_train[self.idx * batch_size:self.idx * batch_size + batch_size]
        batch_labels = self.y_train[self.idx * batch_size:self.idx * batch_size + batch_size]
        self.idx += 1
        return batch_images, batch_labels

if __name__ == '__main__':
    hvd.init()
    if hvd.local_rank() == 0:
        mox.file.copy_parallel(flags.data_url, '/cache/data/')
    hvd._allreduce([0], name="Barrier")
    batch_size = 128
    num_classes = 10
    num_steps = 500
    model_save_path = './checkpoint'
    data_generator = cifar_generator('/cache/data/',
                                     local_rank=hvd.local_rank(),
                                     hvd_size=hvd.size())
    inputs = tf.placeholder(tf.float32, shape=[None, 32, 32, 3], name='inputs')
    labels = tf.placeholder(tf.int32, shape=[None], name='labels')
    is_training = tf.placeholder(tf.bool, name='is_training')
    with slim.arg_scope(nets.resnet_v1.resnet_arg_scope()):
        net, endpoints = nets.resnet_v1.resnet_v1_50(inputs, num_classes=None,
                                                     is_training=is_training)
    with tf.variable_scope('Logits'):
        net = tf.squeeze(net, axis=[1, 2])
        net = slim.dropout(net, keep_prob=0.5, scope='scope')
        logits = slim.fully_connected(net, num_outputs=num_classes,
                                      activation_fn=None, scope='fc')

    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits)
    loss = tf.reduce_mean(losses)
    logits = tf.nn.softmax(logits)
    classes = tf.argmax(logits, axis=1, name='classes')
    accuracy = tf.reduce_mean(tf.cast(
        tf.equal(tf.cast(classes, dtype=tf.int32), labels), dtype=tf.float32))

    lr = 0.0002 * hvd.size()
    optimizer = tf.train.AdamOptimizer(lr, beta1=0.5)
    optimizer = hvd.DistributedOptimizer(optimizer)
    global_step = tf.train.get_or_create_global_step()
    train_step = optimizer.minimize(loss, global_step=global_step)

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(hvd.local_rank())
    hooks = [
        hvd.BroadcastGlobalVariablesHook(0),
        tf.train.StopAtStepHook(last_step=num_steps // hvd.size()),
        tf.train.LoggingTensorHook(tensors={'step': global_step,
                                            'loss': loss,
                                            'acc': accuracy},
                                   every_n_iter=10),
    ]
    checkpoint_dir = model_save_path if hvd.rank() == 0 else None
    with tf.train.MonitoredTrainingSession(checkpoint_dir=checkpoint_dir,
                                           hooks=hooks,
                                           config=config) as mon_sess:
        while not mon_sess.should_stop():
            image_, label_ = data_generator.get_next_batch()
            mon_sess.run(train_step, feed_dict={inputs: image_,
                                                labels: label_,
                                                is_training: True})

    if hvd.local_rank() == 0:
        mox.file.copy_parallel(model_save_path, flags.train_url)

```






