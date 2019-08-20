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

左侧为线下单机单卡代码，右侧为线上单机多卡代码



| 1    | from keras.preprocessing.image import ImageDataGenerator     | =    | 1    | from keras.preprocessing.image import ImageDataGenerator     |
| ---- | ------------------------------------------------------------ | ---- | ---- | ------------------------------------------------------------ |
| 2    | from keras.optimizers import Adam                            |      | 2    | from keras.optimizers import Adam                            |
| 3    | from sklearn.model_selection import train_test_split         |      | 3    | from sklearn.model_selection import train_test_split         |
| 4    | from keras.preprocessing.image import img_to_array           |      | 4    | from keras.preprocessing.image import img_to_array           |
| 5    | from keras.utils import to_categorical                       |      | 5    | from keras.utils import to_categorical                       |
| 6    | from keras.preprocessing import image                        |      | 6    | from keras.preprocessing import image                        |
| 7    | import keras                                                 |      | 7    | import keras                                                 |
| 8    | import numpy as np                                           |      | 8    | import numpy as np                                           |
| 9    | import argparse                                              |      | 9    | import argparse                                              |
| 10   | import random                                                |      | 10   | import random                                                |
| 11   | import cv2                                                   |      | 11   | import cv2                                                   |
| 12   | import os                                                    |      | 12   | import os                                                    |
|      |                                                              | -+   | 13   | # set used gpu, here is 2 gpus                               |
|      |                                                              |      | 14   | os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'                   |
| 13   | import sys                                                   | =    | 15   | import sys                                                   |
|      |                                                              | -+   | 16   | from keras import backend as K                               |
|      |                                                              |      | 17   | import tensorflow as tf                                      |
|      |                                                              |      | 18   | import horovod.keras as hvd                                  |
|      |                                                              |      | 19   | import random                                                |
| 14   | import moxing as mox                                         | =    | 20   | import moxing as mox                                         |
|      |                                                              | -+   | 21   |                                                              |
|      |                                                              |      | 22   | # Horovod: initialize Horovod.                               |
|      |                                                              |      | 23   | hvd.init()                                                   |
|      |                                                              |      | 24   |                                                              |
|      |                                                              |      | 25   | # Horovod: pin GPU to be used to process local rank (one GPU per process) |
|      |                                                              |      | 26   | config = tf.ConfigProto()                                    |
|      |                                                              |      | 27   | config.gpu_options.allow_growth = True                       |
|      |                                                              |      | 28   | config.gpu_options.visible_device_list = str(hvd.local_rank()) |
|      |                                                              |      | 29   | print ('--------------------------- size: ' + str(hvd.size()) + ' -------------------- rank: ' + str(hvd.local_rank())) |
|      |                                                              |      | 30   | K.set_session(tf.Session(config=config))                     |
| 15   |                                                              | =    | 31   |                                                              |
| 16   | parser = argparse.ArgumentParser(description='Keras ImageNet Example', |      | 32   | parser = argparse.ArgumentParser(description='Keras ImageNet Example', |
| 17   | formatter_class=argparse.ArgumentDefaultsHelpFormatter)      |      | 33   | formatter_class=argparse.ArgumentDefaultsHelpFormatter)      |
| 18   | parser.add_argument('--data_url', default=os.path.expanduser('~/imagenet/train'), |      | 34   | parser.add_argument('--data_url', default=os.path.expanduser('~/imagenet/train'), |
| 19   | help='path to training data')                                |      | 35   | help='path to training data')                                |
| 20   | parser.add_argument('--num_gpus', type=int, default=1,       |      | 36   | parser.add_argument('--num_gpus', type=int, default=1,       |
| 21   | help='num of gpus to use')                                   |      | 37   | help='num of gpus to use')                                   |
| 22   | parser.add_argument('--log-dir', default='~/train_url/',     |      | 38   | parser.add_argument('--log-dir', default='~/train_url/',     |
| 23   | help='tensorboard log directory')                            |      | 39   | help='tensorboard log directory')                            |
| 24   | parser.add_argument('--checkpoint_format', default='/cache/checkpoint-{epoch}.h5', |      | 40   | parser.add_argument('--checkpoint_format', default='/cache/checkpoint-{epoch}.h5', |
| 25   | help='checkpoint file format')                               |      | 41   | help='checkpoint file format')                               |
| 26   | parser.add_argument('--train_url', default='/cache/weight.h5', |      | 42   | parser.add_argument('--train_url', default='/cache/weight.h5', |
| 27   | help='weight file')                                          |      | 43   | help='weight file')                                          |
| 28   | parser.add_argument('--batch_size', type=int, default=32,    |      | 44   | parser.add_argument('--batch_size', type=int, default=32,    |
| 29   | help='input batch size for training')                        |      | 45   | help='input batch size for training')                        |
| 30   | parser.add_argument('--val_batch_size', type=int, default=32, |      | 46   | parser.add_argument('--val_batch_size', type=int, default=32, |
| 31   | help='input batch size for validation')                      |      | 47   | help='input batch size for validation')                      |
| 32   | parser.add_argument('--epochs', type=int, default=10,        |      | 48   | parser.add_argument('--epochs', type=int, default=10,        |
| 33   | help='number of epochs to train')                            |      | 49   | help='number of epochs to train')                            |
| 34   | parser.add_argument('--base-lr', type=float, default=0.01,   |      | 50   | parser.add_argument('--base-lr', type=float, default=0.01,   |
| 35   | help='learning rate for a single GPU')                       |      | 51   | help='learning rate for a single GPU')                       |
| 36   | parser.add_argument('--warmup-epochs', type=float, default=5, |      | 52   | parser.add_argument('--warmup-epochs', type=float, default=5, |
| 37   | help='number of warmup epochs')                              |      | 53   | help='number of warmup epochs')                              |
| 38   | parser.add_argument('--momentum', type=float, default=0.9,   |      | 54   | parser.add_argument('--momentum', type=float, default=0.9,   |
| 39   | help='SGD momentum')                                         |      | 55   | help='SGD momentum')                                         |
| 40   | parser.add_argument('--wd', type=float, default=0.0001,      |      | 56   | parser.add_argument('--wd', type=float, default=0.0001,      |
| 41   | help='weight decay')                                         |      | 57   | help='weight decay')                                         |
| 42   |                                                              |      | 58   |                                                              |
| 43   | args = parser.parse_args()                                   |      | 59   | args = parser.parse_args()                                   |
|      |                                                              | <>   | 60   |                                                              |
|      |                                                              |      | 61   | # set rank0 to download data to local cache url              |
|      |                                                              |      | 62   | if hvd.local_rank() == 0:                                    |
| 44   | mox.file.copy_parallel(args.data_url, '/cache/data/')        |      | 63   | mox.file.copy_parallel(args.data_url, '/cache/')             |
| 45   | args.data_url = '/cache/data/'                               |      | 64   | args.data_url = '/cache/'                                    |
|      |                                                              |      | 65   | # broadcast to other rank that rank0 has finished copy       |
|      |                                                              |      | 66   | hvd.allreduce([0], name="Barrier")                           |
| 46   |                                                              | =    | 67   |                                                              |
| 47   | resume_from_epoch = 0                                        |      | 68   | resume_from_epoch = 0                                        |
| 48   | for try_epoch in range(args.epochs, 0, -1):                  |      | 69   | for try_epoch in range(args.epochs, 0, -1):                  |
| 49   | if os.path.exists(args.checkpoint_format.format(epoch=try_epoch)): |      | 70   | if os.path.exists(args.checkpoint_format.format(epoch=try_epoch)): |
| 50   | resume_from_epoch = try_epoch                                |      | 71   | resume_from_epoch = try_epoch                                |
| 51   | break                                                        |      | 72   | break                                                        |
|      |                                                              | -+   | 73   |                                                              |
|      |                                                              |      | 74   | # Horovod: broadcast resume_from_epoch from rank 0 (which will have |
|      |                                                              |      | 75   | # checkpoints) to other ranks.                               |
|      |                                                              |      | 76   | resume_from_epoch = hvd.broadcast(resume_from_epoch, 0, name='resume_from_epoch') |
| 52   |                                                              | =    | 77   |                                                              |
| 53   | # Training data iterator.                                    |      | 78   | # Training data iterator.                                    |
| 54   | train_gen = image.ImageDataGenerator(                        |      | 79   | train_gen = image.ImageDataGenerator(                        |
| 55   | width_shift_range=0.33, height_shift_range=0.33, zoom_range=0.5, horizontal_flip=True, |      | 80   | width_shift_range=0.33, height_shift_range=0.33, zoom_range=0.5, horizontal_flip=True, |
| 56   | preprocessing_function=keras.applications.resnet50.preprocess_input) |      | 81   | preprocessing_function=keras.applications.resnet50.preprocess_input) |
| 57   | train_iter = train_gen.flow_from_directory(args.data_url,    |      | 82   | train_iter = train_gen.flow_from_directory(args.data_url,    |
| 58   | batch_size=args.batch_size,                                  | <>   | 83   | batch_size=args.batch_size,                                  |
| 59   | target_size=(224, 224))                                      |      | 84   | target_size=(224, 224))                                      |
| 60   |                                                              | =    | 85   |                                                              |
| 61   |                                                              |      | 86   |                                                              |
| 62   | # Set up standard ResNet-50 model.                           |      | 87   | # Set up standard ResNet-50 model.                           |
| 63   | base_model = keras.applications.resnet50.ResNet50(weights=None, include_top=False, pooling='avg') |      | 88   | base_model = keras.applications.resnet50.ResNet50(weights=None, include_top=False, pooling='avg') |
| 64   | predictions = keras.layers.Dense(102, activation='softmax')(base_model.output) |      | 89   | predictions = keras.layers.Dense(102, activation='softmax')(base_model.output) |
| 65   | model = keras.models.Model(inputs=base_model.input, outputs=predictions) |      | 90   | model = keras.models.Model(inputs=base_model.input, outputs=predictions) |
| 66   |                                                              |      | 91   |                                                              |
| 67   | model = keras.utils.multi_gpu_model(model, gpus=args.num_gpus) | +-   |      |                                                              |
| 68   | opt = keras.optimizers.SGD(lr=args.base_lr, momentum=args.momentum) | =    | 92   | opt = keras.optimizers.SGD(lr=args.base_lr, momentum=args.momentum) |
|      |                                                              | -+   | 93   |                                                              |
|      |                                                              |      | 94   | # Horovod: add Horovod Distributed Optimizer.                |
|      |                                                              |      | 95   | opt = hvd.DistributedOptimizer(opt)                          |
| 69   |                                                              | =    | 96   |                                                              |
| 70   | model.compile(loss=keras.losses.categorical_crossentropy,    |      | 97   | model.compile(loss=keras.losses.categorical_crossentropy,    |
| 71   | optimizer=opt,                                               |      | 98   | optimizer=opt,                                               |
| 72   | metrics=['accuracy', 'top_k_categorical_accuracy'])          |      | 99   | metrics=['accuracy', 'top_k_categorical_accuracy'])          |
| 73   |                                                              |      | 100  |                                                              |
|      |                                                              | <>   | 101  | callbacks = [                                                |
|      |                                                              |      | 102  | # Horovod: broadcast initial variable states from rank 0 to all other processes. |
|      |                                                              |      | 103  | # This is necessary to ensure consistent initialization of all workers when |
|      |                                                              |      | 104  | # training is started with random weights or restored from a checkpoint. |
|      |                                                              |      | 105  | hvd.callbacks.BroadcastGlobalVariablesCallback(0),           |
|      |                                                              |      | 106  |                                                              |
|      |                                                              |      | 107  | # Horovod: average metrics among workers at the end of every epoch. |
|      |                                                              |      | 108  | #                                                            |
|      |                                                              |      | 109  | # Note: This callback must be in the list before the ReduceLROnPlateau, |
|      |                                                              |      | 110  | # TensorBoard, or other metrics-based callbacks.             |
|      |                                                              |      | 111  | hvd.callbacks.MetricAverageCallback()]                       |
|      |                                                              |      | 112  |                                                              |
|      |                                                              |      | 113  | # set rank0 to save ModelCheckpoint to cache                 |
|      |                                                              |      | 114  | if hvd.local_rank() == 0:                                    |
| 74   | callbacks = [keras.callbacks.ModelCheckpoint(args.checkpoint_format), |      | 115  | callbacks.append(keras.callbacks.ModelCheckpoint(args.checkpoint_format)) |
| 75   | keras.callbacks.TensorBoard(args.log_dir)]                   |      |      |                                                              |
| 76   |                                                              |      | 116  | callbacks.append(keras.callbacks.TensorBoard(args.log_dir))  |
| 77   |                                                              | =    | 117  |                                                              |
| 78   | model.fit_generator(train_iter,                              |      | 118  | model.fit_generator(train_iter,                              |
| 79   | steps_per_epoch=len(train_iter),                             |      | 119  | steps_per_epoch=len(train_iter),                             |
| 80   | callbacks=callbacks,                                         |      | 120  | callbacks=callbacks,                                         |
| 81   | epochs=args.epochs,                                          |      | 121  | epochs=args.epochs,                                          |
| 82   | workers=4,                                                   |      | 122  | workers=4,                                                   |
| 83   | initial_epoch=resume_from_epoch)                             |      | 123  | initial_epoch=resume_from_epoch)                             |
|      |                                                              | <>   | 124  |                                                              |
|      |                                                              |      | 125  | # set rank0 to save weight and move ckpt from cache to s3 url |
| 84   |                                                              |      | 126  | if hvd.local_rank() == 0:                                    |
| 85   | local_checkpoint_url = '/cache/checkpoint/'                  |      | 127  | local_checkpoint_url = '/cache/checkpoint/'                  |
| 86   | if not mox.file.exists(local_checkpoint_url):                |      | 128  | if not mox.file.exists(local_checkpoint_url):                |
| 87   | mox.file.mk_dir(local_checkpoint_url)                        |      | 129  | mox.file.mk_dir(local_checkpoint_url)                        |
| 88   | model.save_weights(os.path.join(local_checkpoint_url, 'weight.h5')) |      | 130  | model.save_weights(os.path.join(local_checkpoint_url, 'weight.h5')) |
| 89   | mox.file.copy_parallel(local_checkpoint_url, args.train_url) |      | 131  | mox.file.copy_parallel(local_checkpoint_url, args.train_url) |

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