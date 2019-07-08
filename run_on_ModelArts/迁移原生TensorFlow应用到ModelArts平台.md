# 迁移原生TensorFlow应用到modelarts平台

主要修改逻辑包括如下：

- 对于支持s3的数据读取接口，直接配置obs桶路径即可，在modelarts平台选择的数据集路径，会以--data_url=s3://xxx/yyy形式传给应用，程序中直接配置data_url并引用即可
- 对于不支持s3的数据读取接口，需要使用mox.file.copy_parallel(src, dst)拷贝到容器/cache/目录
- 将数据读取逻辑封装为tf的张量形式，封装成input_fn
- 将模型定义封装为model_fn，返回模型定义mode_spec
- 使用mox.run启动训练，对应的参数配置参考moxing使用手册

下面左侧为TensorFlow原生应用，右侧为使用moxing api改造的应用，改造后可直接支持分布式多卡训练。

<table class="fc" cellspacing="0" cellpadding="0">
<tr class="SectionBegin">
<td class="TextItemNum AlignRight">1</td>
<td class="TextItemSame">from __future__ import absolute_import</td>
<td class="AlignCenter">=</td>
<td class="TextItemNum AlignRight">1</td>
<td class="TextItemSame">from __future__ import absolute_import</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">2</td>
<td class="TextItemSame">from __future__ import division</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">2</td>
<td class="TextItemSame">from __future__ import division</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">3</td>
<td class="TextItemSame">from __future__ import print_function</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">3</td>
<td class="TextItemSame">from __future__ import print_function</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">4</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">4</td>
<td class="TextItemSame">&nbsp;</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">5</td>
<td class="TextItemSame">from tensorflow.examples.tutorials.mnist import input_data</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">5</td>
<td class="TextItemSame">from tensorflow.examples.tutorials.mnist import input_data</td>
</tr>
<tr class="SectionEnd">
<td class="TextItemNum AlignRight">6</td>
<td class="TextItemSame">import tensorflow as tf</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">6</td>
<td class="TextItemSame">import tensorflow as tf</td>
</tr>
<tr class="SectionBegin">
<td class="TextItemNum AlignRight">&nbsp;</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">-+</td>
<td class="TextItemNum AlignRight">7</td>
<td class="TextItemSigMod"><span class="TextSegSigDiff">import</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">moxing.tensorflow</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">as</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">mox</span></td>
</tr>
<tr class="SectionEnd">
<td class="TextItemNum AlignRight">&nbsp;</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">8</td>
<td class="TextItemSigMod"><span class="TextSegSigDiff">import</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">os</span></td>
</tr>
<tr class="SectionBegin">
<td class="TextItemNum AlignRight">7</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">=</td>
<td class="TextItemNum AlignRight">9</td>
<td class="TextItemSame">&nbsp;</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">8</td>
<td class="TextItemSame">tf.flags.DEFINE_string('data_url', None, 'Directory for storing input data')</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">10</td>
<td class="TextItemSame">tf.flags.DEFINE_string('data_url', None, 'Directory for storing input data')</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">9</td>
<td class="TextItemSame">tf.flags.DEFINE_string('train_url', None, 'Train Url')</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">11</td>
<td class="TextItemSame">tf.flags.DEFINE_string('train_url', None, 'Train Url')</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">10</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">12</td>
<td class="TextItemSame">&nbsp;</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">11</td>
<td class="TextItemSame">flags = tf.flags.FLAGS</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">13</td>
<td class="TextItemSame">flags = tf.flags.FLAGS</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">12</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">14</td>
<td class="TextItemSame">&nbsp;</td>
</tr>
<tr class="SectionEnd">
<td class="TextItemNum AlignRight">13</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">15</td>
<td class="TextItemSame">&nbsp;</td>
</tr>
<tr class="SectionAll">
<td class="TextItemNum AlignRight">14</td>
<td class="TextItemInsigMod">def main(*args, **kwargs):</td>
<td class="AlignCenter">&lt;&gt;</td>
<td class="TextItemNum AlignRight">16</td>
<td class="TextItemInsigMod">def main(*args, **kwargs):<span class="TextSegInsigDiff"> &nbsp;</span></td>
</tr>
<tr class="SectionBegin">
<td class="TextItemNum AlignRight">15</td>
<td class="TextItemSame">###### Import data</td>
<td class="AlignCenter">=</td>
<td class="TextItemNum AlignRight">17</td>
<td class="TextItemSame">###### Import data</td>
</tr>
<tr class="SectionEnd">
<td class="TextItemNum AlignRight">16</td>
<td class="TextItemSame">&nbsp; mnist = input_data.read_data_sets(flags.data_url, one_hot=True)</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">18</td>
<td class="TextItemSame">&nbsp; mnist = input_data.read_data_sets(flags.data_url, one_hot=True)</td>
</tr>
<tr class="SectionBegin">
<td class="TextItemNum AlignRight">17</td>
<td class="TextItemSigMod"><span class="TextSegSigDiff">######</span></td>
<td class="AlignCenter">&lt;&gt;</td>
<td class="TextItemNum AlignRight">19</td>
<td class="TextItemSigMod"><span class="TextSegInsigDiff">&nbsp; </span><span class="TextSegSigDiff">def</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">input_fn(run_mode,</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">**kwargs):</span></td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">&nbsp;</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">20</td>
<td class="TextItemSigMod"><span class="TextSegInsigDiff">&nbsp;&nbsp;&nbsp; </span><span class="TextSegSigDiff">def</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">gen():</span></td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">&nbsp;</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">21</td>
<td class="TextItemSigMod"><span class="TextSegInsigDiff">&nbsp;&nbsp; &nbsp;&nbsp; </span><span class="TextSegSigDiff">while</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">True:</span></td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">&nbsp;</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">22</td>
<td class="TextItemSigMod"><span class="TextSegInsigDiff">&nbsp;&nbsp; &nbsp; &nbsp;&nbsp; </span><span class="TextSegSigDiff">if</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">run_mode</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">==</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">mox.ModeKeys.TRAIN:</span></td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">&nbsp;</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">23</td>
<td class="TextItemSigMod"><span class="TextSegInsigDiff">&nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; </span><span class="TextSegSigDiff">yield</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">mnist.train.next_batch(50)</span></td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">&nbsp;</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">24</td>
<td class="TextItemSigMod"><span class="TextSegInsigDiff">&nbsp;&nbsp; &nbsp; &nbsp;&nbsp; </span><span class="TextSegSigDiff">else:</span></td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">&nbsp;</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">25</td>
<td class="TextItemSigMod"><span class="TextSegInsigDiff">&nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; </span><span class="TextSegSigDiff">yield</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">mnist.test.next_batch(50)</span></td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">&nbsp;</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">26</td>
<td class="TextItemSigMod"><span class="TextSegInsigDiff">&nbsp;&nbsp;&nbsp; </span><span class="TextSegSigDiff">ds</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">=</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">tf.data.Dataset.from_generator(</span></td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">&nbsp;</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">27</td>
<td class="TextItemSigMod"><span class="TextSegInsigDiff">&nbsp;&nbsp; &nbsp; &nbsp;&nbsp; </span><span class="TextSegSigDiff">gen,</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">output_types=(tf.float32,</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">tf.int64),</span></td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">&nbsp;</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">28</td>
<td class="TextItemSigMod"><span class="TextSegInsigDiff">&nbsp;&nbsp; &nbsp; &nbsp;&nbsp; </span><span class="TextSegSigDiff">output_shapes=(tf.TensorShape([None,</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">784]),</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">tf.TensorShape([None,</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">10])))</span></td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">&nbsp;</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">29</td>
<td class="TextItemSigMod"><span class="TextSegInsigDiff">&nbsp;&nbsp;&nbsp; </span><span class="TextSegSigDiff">return</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">ds.make_one_shot_iterator().get_next()</span></td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">&nbsp;</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">30</td>
<td class="TextItemSigMod"><span class="TextSegSigDiff">#####</span></td>
</tr>
<tr class="SectionEnd">
<td class="TextItemNum AlignRight">&nbsp;</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">31</td>
<td class="TextItemInsigMod">&nbsp;</td>
</tr>
<tr class="SectionAll">
<td class="TextItemNum AlignRight">18</td>
<td class="TextItemSame">##### define the model for training or evaling.&nbsp;</td>
<td class="AlignCenter">=</td>
<td class="TextItemNum AlignRight">32</td>
<td class="TextItemSame">##### define the model for training or evaling.&nbsp;</td>
</tr>
<tr class="SectionBegin">
<td class="TextItemNum AlignRight">19</td>
<td class="TextItemSigMod">&nbsp; <span class="TextSegSigDiff">x</span> <span class="TextSegSigDiff">=</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">tf.placehol</span>de<span class="TextSegSigDiff">r</span><span class="TextSegSigDiff">(</span><span class="TextSegSigDiff">tf.float32</span>, <span class="TextSegSigDiff">[</span><span class="TextSegSigDiff">N</span><span class="TextSegSigDiff">on</span>e, <span class="TextSegSigDiff">784]</span>)</td>
<td class="AlignCenter">&lt;&gt;</td>
<td class="TextItemNum AlignRight">33</td>
<td class="TextItemSigMod">&nbsp; <span class="TextSegSigDiff">def</span> <span class="TextSegSigDiff">m</span><span class="TextSegSigDiff">o</span>de<span class="TextSegSigDiff">l_fn(inputs</span>, <span class="TextSegSigDiff">run_mod</span>e, <span class="TextSegSigDiff">**kwargs</span>)<span class="TextSegSigDiff">:</span></td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">20</td>
<td class="TextItemSigMod"><span class="TextSegInsigDiff">&nbsp; </span><span class="TextSegSigDiff">y_</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">=</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">tf.placeholder(tf.float32,</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">[None,</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">10])</span></td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">&nbsp;</td>
<td class="TextItemSame">&nbsp;</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">&nbsp;</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">34</td>
<td class="TextItemSigMod"><span class="TextSegInsigDiff">&nbsp;&nbsp;&nbsp; </span><span class="TextSegSigDiff">x,</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">y_</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">=</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">inputs</span></td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">21</td>
<td class="TextItemInsigMod">&nbsp; W = tf.get_variable(name='W', initializer=tf.zeros([784, 10]))</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">35</td>
<td class="TextItemInsigMod">&nbsp;&nbsp;<span class="TextSegInsigDiff">&nbsp; </span>W = tf.get_variable(name='W', initializer=tf.zeros([784, 10]))</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">22</td>
<td class="TextItemInsigMod">&nbsp; b = tf.get_variable(name='b', initializer=tf.zeros([10]))</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">36</td>
<td class="TextItemInsigMod">&nbsp;&nbsp;<span class="TextSegInsigDiff">&nbsp; </span>b = tf.get_variable(name='b', initializer=tf.zeros([10]))</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">23</td>
<td class="TextItemInsigMod">&nbsp; y = tf.matmul(x, W) + b</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">37</td>
<td class="TextItemInsigMod">&nbsp;&nbsp;<span class="TextSegInsigDiff">&nbsp; </span>y = tf.matmul(x, W) + b</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">24</td>
<td class="TextItemInsigMod">&nbsp; cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">38</td>
<td class="TextItemInsigMod">&nbsp;&nbsp;<span class="TextSegInsigDiff">&nbsp; </span>cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">25</td>
<td class="TextItemInsigMod">&nbsp; correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">39</td>
<td class="TextItemInsigMod">&nbsp;&nbsp;<span class="TextSegInsigDiff">&nbsp; </span>correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">26</td>
<td class="TextItemInsigMod">&nbsp; accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">40</td>
<td class="TextItemInsigMod">&nbsp;&nbsp;<span class="TextSegInsigDiff">&nbsp; </span>accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">27</td>
<td class="TextItemInsigMod">#####</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">41</td>
<td class="TextItemInsigMod">#####<span class="TextSegInsigDiff"> &nbsp;</span></td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">28</td>
<td class="TextItemInsigMod">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">&nbsp;</td>
<td class="TextItemSame">&nbsp;</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">29</td>
<td class="TextItemSigMod">##### use <span class="TextSegSigDiff">tf</span> api to run train and evaluate</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">42</td>
<td class="TextItemSigMod">##### use <span class="TextSegSigDiff">moxing</span> api to run train and evaluate</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">30</td>
<td class="TextItemSigMod">&nbsp; tr<span class="TextSegSigDiff">ain_step</span> = <span class="TextSegSigDiff">tf</span><span class="TextSegSigDiff">.tr</span><span class="TextSegSigDiff">ain.GradientDescentOptimizer(0.5).minimize(cross_entropy)</span></td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">43</td>
<td class="TextItemSigMod">&nbsp;&nbsp;<span class="TextSegInsigDiff">&nbsp; </span><span class="TextSegSigDiff">me</span>tr<span class="TextSegSigDiff">ic_hook</span> = <span class="TextSegSigDiff">mox.LogEvalu</span><span class="TextSegSigDiff">ationM</span><span class="TextSegSigDiff">e</span><span class="TextSegSigDiff">t</span><span class="TextSegSigDiff">r</span><span class="TextSegSigDiff">i</span><span class="TextSegSigDiff">cHo</span><span class="TextSegSigDiff">ok(</span></td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">31</td>
<td class="TextItemSigMod">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">44</td>
<td class="TextItemSigMod"><span class="TextSegInsigDiff">&nbsp;&nbsp; &nbsp;&nbsp; </span><span class="TextSegSigDiff">monitor_info={'loss':</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">cross_entropy,</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">'accuracy':</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">accuracy},</span></td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">32</td>
<td class="TextItemSigMod">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">45</td>
<td class="TextItemSigMod"><span class="TextSegInsigDiff">&nbsp;&nbsp; &nbsp;&nbsp; </span><span class="TextSegSigDiff">batch_size=50,</span></td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">33</td>
<td class="TextItemSigMod">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">46</td>
<td class="TextItemSigMod"><span class="TextSegInsigDiff">&nbsp;&nbsp; &nbsp;&nbsp; </span><span class="TextSegSigDiff">samples_in_train=60,000,</span></td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">34</td>
<td class="TextItemSigMod">&nbsp; ses<span class="TextSegSigDiff">s</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">=</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">tf.</span><span class="TextSegInsigDiff">I</span>n<span class="TextSegSigDiff">teractiveSession()</span></td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">47</td>
<td class="TextItemSigMod">&nbsp;&nbsp;<span class="TextSegInsigDiff">&nbsp; &nbsp; </span>s<span class="TextSegSigDiff">ampl</span>es<span class="TextSegSigDiff">_</span><span class="TextSegInsigDiff">i</span>n<span class="TextSegSigDiff">_eval=10,000,</span></td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">35</td>
<td class="TextItemSigMod">&nbsp; <span class="TextSegSigDiff">tf.g</span><span class="TextSegSigDiff">lobal</span><span class="TextSegSigDiff">_</span><span class="TextSegSigDiff">variables_initializer().run()</span></td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">48</td>
<td class="TextItemSigMod">&nbsp;&nbsp;<span class="TextSegInsigDiff">&nbsp; &nbsp; </span><span class="TextSegSigDiff">evalu</span><span class="TextSegSigDiff">ate_every</span><span class="TextSegSigDiff">_</span><span class="TextSegSigDiff">n_</span><span class="TextSegSigDiff">epochs=1,</span></td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">36</td>
<td class="TextItemSigMod">&nbsp; <span class="TextSegSigDiff">#</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">Tr</span><span class="TextSegSigDiff">a</span><span class="TextSegSigDiff">i</span><span class="TextSegInsigDiff">n</span></td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">49</td>
<td class="TextItemSigMod">&nbsp;&nbsp;<span class="TextSegInsigDiff">&nbsp; &nbsp; </span><span class="TextSegSigDiff">prefix='[VALIDATIO</span><span class="TextSegInsigDiff">N</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">METRICS]')</span></td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">37</td>
<td class="TextItemSigMod">&nbsp; <span class="TextSegSigDiff">f</span>or <span class="TextSegSigDiff">_</span> in<span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">r</span><span class="TextSegSigDiff">an</span>ge<span class="TextSegSigDiff">(1000</span><span class="TextSegSigDiff">):</span></td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">50</td>
<td class="TextItemSigMod">&nbsp;&nbsp;<span class="TextSegInsigDiff">&nbsp; </span><span class="TextSegSigDiff">exp</span>or<span class="TextSegSigDiff">t_spec</span> <span class="TextSegSigDiff">=</span> <span class="TextSegSigDiff">mox.ExportSpec(</span>in<span class="TextSegSigDiff">puts_dict={'ima</span>ge<span class="TextSegSigDiff">s':</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">x},</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">outputs_dict={'logits':</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">y})</span></td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">&nbsp;</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">51</td>
<td class="TextItemSigMod"><span class="TextSegInsigDiff">&nbsp;&nbsp;&nbsp; </span><span class="TextSegSigDiff">return</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">mox.ModelSpec(loss=cross_entropy,</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">log_info={'loss':</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">cross_entropy,</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">'accuracy':</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">accuracy},</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">hooks=[metric_hook],</span></td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">&nbsp;</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">52</td>
<td class="TextItemSigMod"><span class="TextSegInsigDiff">&nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; </span><span class="TextSegSigDiff">export_spec=export_spec)</span></td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">38</td>
<td class="TextItemSigMod"><span class="TextSegInsigDiff">&nbsp;&nbsp;&nbsp; </span><span class="TextSegSigDiff">batch_xs,</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">batch_ys</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">=</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">mnist.train.next_batch(50)</span></td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">53</td>
<td class="TextItemSigMod">&nbsp;</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">39</td>
<td class="TextItemSigMod"><span class="TextSegInsigDiff">&nbsp;&nbsp;&nbsp; </span><span class="TextSegSigDiff">sess.run(train_step,</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">feed_dict={x:</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">batch_xs,</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">y_:</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">batch_ys})</span></td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">54</td>
<td class="TextItemSigMod">&nbsp;</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">&nbsp;</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">55</td>
<td class="TextItemSigMod"><span class="TextSegInsigDiff">&nbsp; </span><span class="TextSegSigDiff">mox.run(input_fn=input_fn,</span></td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">&nbsp;</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">56</td>
<td class="TextItemSigMod"><span class="TextSegInsigDiff">&nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; </span><span class="TextSegSigDiff">model_fn=model_fn,</span></td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">40</td>
<td class="TextItemSigMod">&nbsp; <span class="TextSegSigDiff">p</span><span class="TextSegSigDiff">ri</span><span class="TextSegSigDiff">n</span><span class="TextSegSigDiff">t(s</span><span class="TextSegSigDiff">ess.ru</span>n(<span class="TextSegSigDiff">accuracy</span>, <span class="TextSegSigDiff">feed_dict={x:</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">mnist.</span>te<span class="TextSegSigDiff">st.images</span>,</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">57</td>
<td class="TextItemSigMod">&nbsp;&nbsp;<span class="TextSegInsigDiff">&nbsp; &nbsp; &nbsp; &nbsp; </span><span class="TextSegSigDiff">optimizer_fn=mox.get_optimize</span><span class="TextSegSigDiff">r_f</span>n(<span class="TextSegSigDiff">'sgd'</span>, <span class="TextSegSigDiff">learn</span><span class="TextSegSigDiff">i</span><span class="TextSegSigDiff">ng_ra</span>te<span class="TextSegSigDiff">=0.01)</span>,</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">41</td>
<td class="TextItemSigMod">&nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp;<span class="TextSegInsigDiff">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; </span><span class="TextSegSigDiff">y_:</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">mnist.t</span><span class="TextSegSigDiff">e</span><span class="TextSegSigDiff">s</span><span class="TextSegSigDiff">t.labels}))</span></td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">58</td>
<td class="TextItemSigMod">&nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; <span class="TextSegSigDiff">run</span><span class="TextSegSigDiff">_mode=mox</span><span class="TextSegSigDiff">.ModeKeys.TR</span><span class="TextSegSigDiff">AIN,</span></td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">42</td>
<td class="TextItemSigMod">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">59</td>
<td class="TextItemSigMod"><span class="TextSegInsigDiff">&nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; </span><span class="TextSegSigDiff">inter_mode=mox.ModeKeys.EVAL,</span></td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">&nbsp;</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">60</td>
<td class="TextItemSigMod"><span class="TextSegInsigDiff">&nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; </span><span class="TextSegSigDiff">batch_size=50,</span></td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">&nbsp;</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">61</td>
<td class="TextItemSigMod"><span class="TextSegInsigDiff">&nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; </span><span class="TextSegSigDiff">auto_batch=False,</span></td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">&nbsp;</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">62</td>
<td class="TextItemSigMod"><span class="TextSegInsigDiff">&nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; </span><span class="TextSegSigDiff">log_dir=flags.train_url,</span></td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">&nbsp;</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">63</td>
<td class="TextItemSigMod"><span class="TextSegInsigDiff">&nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; </span><span class="TextSegSigDiff">max_number_of_steps=1000,</span></td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">&nbsp;</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">64</td>
<td class="TextItemSigMod"><span class="TextSegInsigDiff">&nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; </span><span class="TextSegSigDiff">log_every_n_steps=10,</span></td>
</tr>
<tr class="SectionEnd">
<td class="TextItemNum AlignRight">&nbsp;</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">65</td>
<td class="TextItemSigMod"><span class="TextSegInsigDiff">&nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; </span><span class="TextSegSigDiff">export_model=mox.ExportKeys.TF_SERVING)</span></td>
</tr>
<tr class="SectionAll">
<td class="TextItemNum AlignRight">43</td>
<td class="TextItemSame">######</td>
<td class="AlignCenter">=</td>
<td class="TextItemNum AlignRight">66</td>
<td class="TextItemSame">######</td>
</tr>
<tr class="SectionAll">
<td class="TextItemNum AlignRight">&nbsp;</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">-+</td>
<td class="TextItemNum AlignRight">67</td>
<td class="TextItemInsigMod">&nbsp;</td>
</tr>
<tr class="SectionBegin">
<td class="TextItemNum AlignRight">44</td>
<td class="TextItemSame">if __name__ == '__main__':</td>
<td class="AlignCenter">=</td>
<td class="TextItemNum AlignRight">68</td>
<td class="TextItemSame">if __name__ == '__main__':</td>
</tr>
<tr class="SectionEnd">
<td class="TextItemNum AlignRight">45</td>
<td class="TextItemSame">&nbsp; tf.app.run(main=main)</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">69</td>
<td class="TextItemSame">&nbsp; tf.app.run(main=main)</td>
</tr>
</table>







