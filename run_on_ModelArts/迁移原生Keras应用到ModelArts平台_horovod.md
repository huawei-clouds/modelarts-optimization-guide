# keras迁移modelarts指导

## 数据访问

### 将数据下载入容器

* 在数据加载前，使用mox.file.copy_parallel接口将obs上的数据下载到容器/cache目录
* 训练过程中，将输出的日志文件、模型文件等保存在容器的/cache目录下
* 训练结束后，将保存在/cache目录下的日志文件、模型文件通过mox.file.copy_parallel接口拷贝回到obs目录

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

##  keras.utils.multi_gpu_model实现单机多卡

keras单机多卡只需要定义了Model之后使用 keras.utils.multi_gpu_model即可

#Set up standard ResNet-50 model.

base_model = keras.applications.resnet50.ResNet50(weights=None, include_top=False, pooling='avg')
predictions = keras.layers.Dense(102, activation='softmax')(base_model.output)
model = keras.models.Model(inputs=base_model.input, outputs=predictions)

**model = keras.utils.multi_gpu_model(model, gpus=args.num_gpus)**
opt = keras.optimizers.SGD(lr=args.base_lr, momentum=args.momentum)

model.compile(loss=keras.losses.categorical_crossentropy,
​                            optimizer=opt,
​                            metrics=['accuracy', 'top_k_categorical_accuracy'])

其中num_gpus是机器中使用的gpu数量，下发作业时会以num_gpus参数传入。

## horovod实现单机多卡

```diff
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from keras.preprocessing import image
import keras
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import cv2
import os
+ # set used gpu, here is 2 gpus
+ os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
import sys
+ from keras import backend as K
+ import tensorflow as tf
+ import horovod.keras as hvd
+ import random
import moxing as mox
+ hvd.init()
+ config = tf.ConfigProto()
+ config.gpu_options.allow_growth = True
+ config.gpu_options.visible_device_list = str(hvd.local_rank())
+ rank: ' + str(hvd.local_rank()))
+ K.set_session(tf.Session(config=config))

parser = argparse.ArgumentParser(description='Keras ImageNet Example',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_url', default=os.path.expanduser('~/imagenet/train'),
                    help='path to training data')
parser.add_argument('--num_gpus', type=int, default=1,
                    help='num of gpus to use')
parser.add_argument('--log-dir', default='~/train_url/',
                    help='tensorboard log directory')
parser.add_argument('--checkpoint_format', default='/cache/checkpoint-{epoch}.h5',
                    help='checkpoint file format')
parser.add_argument('--train_url', default='/cache/weight.h5',
                    help='weight file')
# Default settings from https://arxiv.org/abs/1706.02677.
parser.add_argument('--batch-size', type=int, default=32,
                    help='input batch size for training')
parser.add_argument('--val-batch-size', type=int, default=32,
                    help='input batch size for validation')
parser.add_argument('--epochs', type=int, default=10,
                    help='number of epochs to train')
parser.add_argument('--base-lr', type=float, default=0.01,
                    help='learning rate for a single GPU')
parser.add_argument('--warmup-epochs', type=float, default=5,
                    help='number of warmup epochs')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--wd', type=float, default=0.0001,
                    help='weight decay')

args = parser.parse_args()

- mox.file.copy_parallel(args.data_url, '/cache/data/')
- args.data_url = '/cache/data/'
+ hvd.init()
+ if hvd.local_rank() == 0:
+     mox.file.copy_parallel(args.data_url, '/cache/')
+ args.data_url = '/cache/'
+ hvd.allreduce([0], name="Barrier")

resume_from_epoch = 0
for try_epoch in range(args.epochs, 0, -1):
    if os.path.exists(args.checkpoint_format.format(epoch=try_epoch)):
        resume_from_epoch = try_epoch
        break

+ resume_from_epoch = hvd.broadcast(resume_from_epoch, 0, name='resume_from_epoch')

# Training data iterator.
train_gen = image.ImageDataGenerator(
    width_shift_range=0.33, height_shift_range=0.33, zoom_range=0.5, horizontal_flip=True,
    preprocessing_function=keras.applications.resnet50.preprocess_input)
train_iter = train_gen.flow_from_directory(args.data_url,
                                           batch_size=args.batch_size,
                                           target_size=(224, 224))


# Set up standard ResNet-50 model.
base_model = keras.applications.resnet50.ResNet50(weights=None, include_top=False, pooling='avg')
predictions = keras.layers.Dense(102, activation='softmax')(base_model.output)
model = keras.models.Model(inputs=base_model.input, outputs=predictions)

model = keras.utils.multi_gpu_model(model, gpus=args.num_gpus)
opt = keras.optimizers.SGD(lr=args.base_lr, momentum=args.momentum)
+ opt = hvd.DistributedOptimizer(opt)

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=opt,
              metrics=['accuracy', 'top_k_categorical_accuracy'])

- callbacks = [keras.callbacks.ModelCheckpoint(args.checkpoint_format),keras.callbacks.TensorBoard(args.log_dir)]
+ callbacks = [
+     # Horovod: broadcast initial variable states from rank 0 to all other processes.
+     # This is necessary to ensure consistent initialization of all workers when
+     # training is started with random weights or restored from a checkpoint.
+     hvd.callbacks.BroadcastGlobalVariablesCallback(0),

+     # Horovod: average metrics among workers at the end of every epoch.
+     #
+     # Note: This callback must be in the list before the ReduceLROnPlateau,
+     # TensorBoard, or other metrics-based callbacks.
+     hvd.callbacks.MetricAverageCallback()]

+ if hvd.local_rank() == 0:
+     callbacks.append(keras.callbacks.ModelCheckpoint(args.checkpoint_format))
+     callbacks.append(keras.callbacks.TensorBoard(args.log_dir))

model.fit_generator(train_iter,
              steps_per_epoch=len(train_iter),
              callbacks=callbacks,
              epochs=args.epochs,
              workers=4,
              initial_epoch=resume_from_epoch)

- local_checkpoint_url = '/cache/checkpoint/'
- if not mox.file.exists(local_checkpoint_url):
-     mox.file.mk_dir(local_checkpoint_url)
- model.save_weights(os.path.join(local_checkpoint_url, 'weight.h5'))
- mox.file.copy_parallel(local_checkpoint_url, args.train_url)
+ if hvd.local_rank() == 0:
+     local_checkpoint_url = '/cache/checkpoint/'
+     if not mox.file.exists(local_checkpoint_url):
+         mox.file.mk_dir(local_checkpoint_url)
+     model.save_weights(os.path.join(local_checkpoint_url, 'weight.h5'))
+     mox.file.copy_parallel(local_checkpoint_url, args.train_url)
```



## keras horovod读数据注意事项

由于每个rank在取一个batch的数据时，是从所有数据中取自己训练一部分，会有两个影响：

1.不同的rank可能在同一个batch中取到的数据会有相同的部分，对于训练精度有影响。

2.最终训练的epoch相当于每张卡都训练了一遍所有的数据集，即实际的epoch数是卡数*epoch数

所以推荐均分数据集让数据集的每一部分单独对自己的rank可见，如下：

sample_list = list(range(len(train_info)))
random.shuffle(sample_list)
part_samples = len(train_info) // hvd.size()
local_pos = part_samples * hvd.local_rank()
train_info = [train_info[item] for item in sample_list[local_pos:local_pos + part_samples]]
y_train = [y_train[item] for item in sample_list[local_pos:local_pos + part_samples]]

其中train_info是训练图片的list，y_train是训练图片标签的list。

## 多进程实现分布式或多卡注意事项

1. 通过进程号控制数据读取，尽量使得每个进程读取不同的数据块
2. OBS文件读写操作尽量使用某一进程负责，不要出现多进程同时操作同一文件的情况
3. 每个进程都会打印各自的日志，可以根据进程编号自行控制
4. 验证集如果切分到多个进程并行执行，结果打印需要进程间同步或自行控制
5. 模型参数初始化后，需要从某一进程广播到其它进程，使得所有进程模型的初始化参数保持一致，（imagenet实际测试，不一致的初始化参数会导致最终训练精度降低2%）
