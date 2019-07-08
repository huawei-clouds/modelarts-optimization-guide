# Modifying keras to MoXing using the mnist example  #

本文档以mnist数据集为例，将keras的训练代码修改为moxing版本。

改MoXing注意事项：

  1. keras的所有模型接口必须转化为tensorflow.python.keras的接口。

  2. 使用keras bn层时，tensorflow.python.keras的bn层的training参数训练时必需指定为True，默认为False。
  
  3. keras日志打印的精度以及损失值均为累计值，tf为每个batch的值。
  

注： keras mnist案例的数据集为mnist.npz格式， MoXing的数据集为ubyte.gz格式。

数据集下载地址：keras:[https://s3.amazonaws.com/img-datasets/mnist.npz](https://s3.amazonaws.com/img-datasets/mnist.npz "keras mnist")  

moxing:[https://storage.googleapis.com/cvdf-datasets/mnist/](https://storage.googleapis.com/cvdf-datasets/mnist/ "MoXing mnist")

示例代码如下：
  
**keras:**

    from __future__ import absolute_import
	from __future__ import division
	from __future__ import print_function
	
	import keras
	from keras.datasets import mnist
	from keras.models import Sequential
	from keras.layers import Dense, Dropout, Flatten
	from keras.layers import Conv2D, MaxPooling2D
	from keras import backend as K
	
	batch_size = 64
	num_classes = 10
	epochs = 12
	
	# input image dimensions
	img_rows, img_cols = 28, 28
	
	(x_train, y_train), (_, _) = mnist.load_data(path='/home/tmp/data/mnist.npz')  # dataset
	
	if K.image_data_format() == 'channels_first':                                  # data format
	    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
	    input_shape = (1, img_rows, img_cols)
	else:
	    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
	    input_shape = (img_rows, img_cols, 1)
	
	x_train = x_train.astype('float32')
	x_train /= 255
	
	# convert class vectors to binary class matrices
	y_train = keras.utils.to_categorical(y_train, num_classes)                     # one-hot label
	
	model = Sequential()                                                           # build model
	model.add(Conv2D(32, kernel_size=(3, 3),                                  
	                 activation='relu',
	                 input_shape=input_shape))
	
	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(num_classes, activation='softmax'))
	
	model.compile(loss=keras.losses.categorical_crossentropy,                      # loss
	              optimizer=keras.optimizers.Adadelta(),                           # optimizer
	              metrics=['accuracy'])                                            # accuracy
	
	model.fit(x_train, y_train,                                                    # model training
	          batch_size=batch_size,
	          epochs=epochs,
	          verbose=1)


**MoXing:**
    
	from __future__ import absolute_import
	from __future__ import division
	from __future__ import print_function
	
	import tensorflow as tf
	from tensorflow.contrib import slim
	from tensorflow.examples.tutorials.mnist import input_data
	from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
	from tensorflow.python.keras.layers import Dense, Dropout, Flatten
	from tensorflow.python.keras.losses import categorical_crossentropy
	
	import moxing.tensorflow as mox
	
	
	tf.flags.DEFINE_string('data_url', '/home/jnn/nfs/mnist', 'Dir of dataset')
	tf.flags.DEFINE_string('train_url', None, 'Train Url')
	flags = tf.flags.FLAGS
	
	num_classes = 10
	
	# input image dimensions
	img_rows, img_cols = 28, 28
	
	def main(*args):
	
	  mox.set_flag('variable_update', 'parameter_server')
	
	  def input_fn(mode):                                                    # dataset, build a dataset generator
	
	    mnist = input_data.read_data_sets(flags.data_url, reshape=False, one_hot=False)
	
	    def gen():
	      while True:
	        yield mnist.train.next_batch(64)
	
	    ds = tf.data.Dataset.from_generator(
	        gen, output_types=(tf.float32, tf.int64),
	        output_shapes=(tf.TensorShape([None, 28, 28, 1]), tf.TensorShape([None])))
	
	    return ds.make_one_shot_iterator().get_next()
	
	  def model_fn(inputs, mode):                                           # build model
	    images, labels = inputs
	    # convert class vectors to binary class matrices
	    labels_one_hot = slim.one_hot_encoding(labels, num_classes)
	
	    def my_model(x):
	
	      x = Conv2D(32, kernel_size=(3, 3), activation='relu')(x)
	      x = Conv2D(64, (3, 3), activation='relu')(x)
	      x = MaxPooling2D(pool_size=(2, 2))(x)
	      x = Dropout(0.25)(x)
	      x = Flatten()(x)
	      x = Dense(128, activation='relu')(x)
	      x = Dropout(0.5)(x)
	      x = Dense(num_classes, activation='softmax')(x)
	
	      return x
	
	    logits = my_model(images)
	    loss = tf.reduce_mean(categorical_crossentropy(logits, labels_one_hot))    # loss
	    log_info = {'loss': loss}
	
	    accuracy_top_1 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits, labels, 1), tf.float32))   # accuracy
	
	    log_info['accuracy'] = accuracy_top_1
	    export_spec = mox.ExportSpec(inputs_dict={'images': images},               # for export model
	                                 outputs_dict={'logits': labels},
	                                 version='model')
	
	    return mox.ModelSpec(loss=loss,
	                         log_info=log_info,
	                         export_spec=export_spec)
	
	  def optimizer_fn():                                                          # optimizer
	    opt = mox.get_optimizer_fn('adadelta', learning_rate=1.0)()
	    return opt
	
	  mox.run(input_fn=input_fn,                                                   # model tarining
	          model_fn=model_fn,
	          optimizer_fn=optimizer_fn,
	          run_mode=mox.ModeKeys.TRAIN,
	          max_number_of_steps=3000,
	          log_dir=flags.train_url,
	          export_model=mox.ExportKeys.TF_SERVING)
	
	if __name__ == "__main__":
	  tf.app.run(main=main)
    


