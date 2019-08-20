# pytorch迁移modelarts指导

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

## torch.nn.parallel.DistributedDataParallel实现单机多卡

左侧为线下单机单卡代码，右侧为线上单机多卡代码

<table class="fc" cellspacing="0" cellpadding="0">
<tr class="SectionBegin">
<td class="TextItemNum AlignRight">1</td>
<td class="TextItemSame">import argparse</td>
<td class="AlignCenter">=</td>
<td class="TextItemNum AlignRight">1</td>
<td class="TextItemSame">import argparse</td>
</tr>
<tr class="SectionEnd">
<td class="TextItemNum AlignRight">2</td>
<td class="TextItemSame">import os</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">2</td>
<td class="TextItemSame">import os</td>
</tr>
<tr class="SectionAll">
<td class="TextItemNum AlignRight">&nbsp;</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">-+</td>
<td class="TextItemNum AlignRight">3</td>
<td class="TextItemSigRightMod"><span class="TextSegSigDiff">os.environ[&quot;CUDA_VISIBLE_DEVICES&quot;]</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">=</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">'0,</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">1'</span></td>
</tr>
<tr class="SectionBegin">
<td class="TextItemNum AlignRight">3</td>
<td class="TextItemSame">import shutil</td>
<td class="AlignCenter">=</td>
<td class="TextItemNum AlignRight">4</td>
<td class="TextItemSame">import shutil</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">4</td>
<td class="TextItemSame">import time</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">5</td>
<td class="TextItemSame">import time</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">5</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">6</td>
<td class="TextItemSame">&nbsp;</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">6</td>
<td class="TextItemSame">import torch</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">7</td>
<td class="TextItemSame">import torch</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">7</td>
<td class="TextItemSame">import torch.nn as nn</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">8</td>
<td class="TextItemSame">import torch.nn as nn</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">8</td>
<td class="TextItemSame">import torch.backends.cudnn as cudnn</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">9</td>
<td class="TextItemSame">import torch.backends.cudnn as cudnn</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">9</td>
<td class="TextItemSame">import torch.optim</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">10</td>
<td class="TextItemSame">import torch.optim</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">10</td>
<td class="TextItemSame">import torch.utils.data</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">11</td>
<td class="TextItemSame">import torch.utils.data</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">11</td>
<td class="TextItemSame">import torchvision.transforms as transforms</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">12</td>
<td class="TextItemSame">import torchvision.transforms as transforms</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">12</td>
<td class="TextItemSame">import torchvision.datasets as datasets</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">13</td>
<td class="TextItemSame">import torchvision.datasets as datasets</td>
</tr>
<tr class="SectionEnd">
<td class="TextItemNum AlignRight">13</td>
<td class="TextItemSame">import torchvision.models as models</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">14</td>
<td class="TextItemSame">import torchvision.models as models</td>
</tr>
<tr class="SectionBegin">
<td class="TextItemNum AlignRight">&nbsp;</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">-+</td>
<td class="TextItemNum AlignRight">15</td>
<td class="TextItemInsigRightMod">&nbsp;</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">&nbsp;</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">16</td>
<td class="TextItemSigRightMod"><span class="TextSegSigDiff">import</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">torch.distributed</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">as</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">dist</span></td>
</tr>
<tr class="SectionEnd">
<td class="TextItemNum AlignRight">&nbsp;</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">17</td>
<td class="TextItemSigRightMod"><span class="TextSegSigDiff">import</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">torch.multiprocessing</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">as</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">mp</span></td>
</tr>
<tr class="SectionBegin">
<td class="TextItemNum AlignRight">14</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">=</td>
<td class="TextItemNum AlignRight">18</td>
<td class="TextItemSame">&nbsp;</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">15</td>
<td class="TextItemSame">model_names = sorted(name for name in models.__dict__</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">19</td>
<td class="TextItemSame">model_names = sorted(name for name in models.__dict__</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">16</td>
<td class="TextItemSame">&nbsp; &nbsp; if name.islower() and not name.startswith(&quot;__&quot;)</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">20</td>
<td class="TextItemSame">&nbsp; &nbsp; if name.islower() and not name.startswith(&quot;__&quot;)</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">17</td>
<td class="TextItemSame">&nbsp; &nbsp; and callable(models.__dict__[name]))</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">21</td>
<td class="TextItemSame">&nbsp; &nbsp; and callable(models.__dict__[name]))</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">18</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">22</td>
<td class="TextItemSame">&nbsp;</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">19</td>
<td class="TextItemSame">parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">23</td>
<td class="TextItemSame">parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">20</td>
<td class="TextItemSame">parser.add_argument('--data', help='path to dataset')</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">24</td>
<td class="TextItemSame">parser.add_argument('--data', help='path to dataset')</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">21</td>
<td class="TextItemSame">parser.add_argument('--arch', default='resnet18',</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">25</td>
<td class="TextItemSame">parser.add_argument('--arch', default='resnet18',</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">22</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; choices=model_names,</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">26</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; choices=model_names,</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">23</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; help='model architecture: ' +</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">27</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; help='model architecture: ' +</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">24</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; ' | '.join(model_names) +</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">28</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; ' | '.join(model_names) +</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">25</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; ' (default: resnet18)')</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">29</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; ' (default: resnet18)')</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">26</td>
<td class="TextItemSame">parser.add_argument('--workers', default=4, type=int, metavar='N',</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">30</td>
<td class="TextItemSame">parser.add_argument('--workers', default=4, type=int, metavar='N',</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">27</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; help='number of data loading workers (default: 4)')</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">31</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; help='number of data loading workers (default: 4)')</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">28</td>
<td class="TextItemSame">parser.add_argument('--print-freq', default=20, type=int,</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">32</td>
<td class="TextItemSame">parser.add_argument('--print-freq', default=20, type=int,</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">29</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; metavar='N', help='print frequency (default: 10)')</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">33</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; metavar='N', help='print frequency (default: 10)')</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">30</td>
<td class="TextItemSame">parser.add_argument('--epochs', default=10, type=int, metavar='N',</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">34</td>
<td class="TextItemSame">parser.add_argument('--epochs', default=10, type=int, metavar='N',</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">31</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; help='number of total epochs to run')</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">35</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; help='number of total epochs to run')</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">32</td>
<td class="TextItemSame">parser.add_argument('--batch_size', default=256, type=int,</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">36</td>
<td class="TextItemSame">parser.add_argument('--batch_size', default=256, type=int,</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">33</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; metavar='N',</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">37</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; metavar='N',</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">34</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; help='mini-batch size (default: 256), this is the total '</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">38</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; help='mini-batch size (default: 256), this is the total '</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">35</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; 'batch size of all GPUs on the current node when '</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">39</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; 'batch size of all GPUs on the current node when '</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">36</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; 'using Data Parallel or Distributed Data Parallel')</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">40</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; 'using Data Parallel or Distributed Data Parallel')</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">37</td>
<td class="TextItemSame">parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">41</td>
<td class="TextItemSame">parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">38</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; metavar='LR', help='initial learning rate', dest='lr')</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">42</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; metavar='LR', help='initial learning rate', dest='lr')</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">39</td>
<td class="TextItemSame">parser.add_argument('--momentum', default=0.9, type=float, metavar='M',</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">43</td>
<td class="TextItemSame">parser.add_argument('--momentum', default=0.9, type=float, metavar='M',</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">40</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; help='momentum')</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">44</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; help='momentum')</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">41</td>
<td class="TextItemSame">parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">45</td>
<td class="TextItemSame">parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">42</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; metavar='W', help='weight decay (default: 1e-4)',</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">46</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; metavar='W', help='weight decay (default: 1e-4)',</td>
</tr>
<tr class="SectionEnd">
<td class="TextItemNum AlignRight">43</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; dest='weight_decay')</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">47</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; dest='weight_decay')</td>
</tr>
<tr class="SectionAll">
<td class="TextItemNum AlignRight">44</td>
<td class="TextItemSigDiffMod">parser.add_argument('--gpu', default=<span class="TextSegSigDiff">0</span>, type=<span class="TextSegSigDiff">int</span>, help='GPU id to use.')</td>
<td class="AlignCenter">&lt;&gt;</td>
<td class="TextItemNum AlignRight">48</td>
<td class="TextItemSigDiffMod">parser.add_argument('--gpu', default=<span class="TextSegSigDiff">'0,1,2,3'</span>, type=<span class="TextSegSigDiff">s</span><span class="TextSegSigDiff">tr</span>, help='GPU id to use.')</td>
</tr>
<tr class="SectionBegin">
<td class="TextItemNum AlignRight">45</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">=</td>
<td class="TextItemNum AlignRight">49</td>
<td class="TextItemSame">&nbsp;</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">46</td>
<td class="TextItemSame">best_acc1 = 0</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">50</td>
<td class="TextItemSame">best_acc1 = 0</td>
</tr>
<tr class="SectionEnd">
<td class="TextItemNum AlignRight">47</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">51</td>
<td class="TextItemSame">&nbsp;</td>
</tr>
<tr class="SectionAll">
<td class="TextItemNum AlignRight">48</td>
<td class="TextItemInsigLeftMod">&nbsp;</td>
<td class="AlignCenter">+-</td>
<td class="TextItemNum AlignRight">&nbsp;</td>
<td class="TextItemSame">&nbsp;</td>
</tr>
<tr class="SectionAll">
<td class="TextItemNum AlignRight">49</td>
<td class="TextItemSame">def main():</td>
<td class="AlignCenter">=</td>
<td class="TextItemNum AlignRight">52</td>
<td class="TextItemSame">def main():</td>
</tr>
<tr class="SectionBegin">
<td class="TextItemNum AlignRight">50</td>
<td class="TextItemSigDiffMod">&nbsp; &nbsp; args = parser.parse_args()</td>
<td class="AlignCenter">&lt;&gt;</td>
<td class="TextItemNum AlignRight">53</td>
<td class="TextItemSigDiffMod">&nbsp; &nbsp; args<span class="TextSegSigDiff">,</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">_</span> = parser.parse<span class="TextSegSigDiff">_known</span>_args()</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">&nbsp;</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">54</td>
<td class="TextItemSigRightMod"><span class="TextSegInsigDiff">&nbsp; &nbsp; </span><span class="TextSegSigDiff">start</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">=</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">time.time()</span></td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">&nbsp;</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">55</td>
<td class="TextItemSigRightMod"><span class="TextSegInsigDiff">&nbsp; &nbsp; </span><span class="TextSegSigDiff">print('---------Copy</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">DATA</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">to</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">cache---------')</span></td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">&nbsp;</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">56</td>
<td class="TextItemSigRightMod"><span class="TextSegInsigDiff">&nbsp; &nbsp; </span><span class="TextSegSigDiff">import</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">mox_patch_0603_v4</span></td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">&nbsp;</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">57</td>
<td class="TextItemSigRightMod"><span class="TextSegInsigDiff">&nbsp; &nbsp; </span><span class="TextSegSigDiff">import</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">moxing</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">as</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">mox</span></td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">&nbsp;</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">58</td>
<td class="TextItemSigRightMod"><span class="TextSegInsigDiff">&nbsp; &nbsp; </span><span class="TextSegSigDiff">mox.file.set_auth(negotiation=False)</span></td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">&nbsp;</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">59</td>
<td class="TextItemSigRightMod"><span class="TextSegInsigDiff">&nbsp; &nbsp; </span><span class="TextSegSigDiff">mox.file.make_dirs('/cache/fruit/')</span></td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">&nbsp;</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">60</td>
<td class="TextItemSigRightMod"><span class="TextSegInsigDiff">&nbsp; &nbsp; </span><span class="TextSegSigDiff">mox.file.make_dirs('/cache/results')</span></td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">&nbsp;</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">61</td>
<td class="TextItemSigRightMod"><span class="TextSegInsigDiff">&nbsp; &nbsp; </span><span class="TextSegSigDiff">mox.file.copy_parallel(args.data,</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">'/cache/fruit/')</span></td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">&nbsp;</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">62</td>
<td class="TextItemSigRightMod"><span class="TextSegInsigDiff">&nbsp; &nbsp; </span><span class="TextSegSigDiff">args.data</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">=</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">'/cache/fruit/'</span></td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">&nbsp;</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">63</td>
<td class="TextItemSigRightMod"><span class="TextSegInsigDiff">&nbsp; &nbsp; </span><span class="TextSegSigDiff">print('---------Copy</span><span class="TextSegInsigDiff">&nbsp; </span><span class="TextSegSigDiff">Finished---------</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">'</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">+</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">str(time.time()</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">-</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">start)</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">+</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">'</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">s')</span></td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">&nbsp;</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">64</td>
<td class="TextItemSigRightMod"><span class="TextSegInsigDiff">&nbsp; &nbsp; </span><span class="TextSegSigDiff">ngpus_per_node</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">=</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">len(args.gpu.split(','))</span></td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">&nbsp;</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">65</td>
<td class="TextItemSigRightMod"><span class="TextSegInsigDiff">&nbsp; &nbsp; </span><span class="TextSegSigDiff">mp.spawn(main_worker,</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">nprocs=ngpus_per_node,</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">args=(ngpus_per_node,</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">args))</span></td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">&nbsp;</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">66</td>
<td class="TextItemSigRightMod"><span class="TextSegInsigDiff">&nbsp; &nbsp; </span><span class="TextSegSigDiff">save_dir</span><span class="TextSegInsigDiff">&nbsp; </span><span class="TextSegSigDiff">=</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">'/cache/results'</span></td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">&nbsp;</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">67</td>
<td class="TextItemSigRightMod"><span class="TextSegInsigDiff">&nbsp; &nbsp; </span><span class="TextSegSigDiff">s3_results_path</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">=</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">'s3://modelarts-no1/ckpt/pytorch_example_results/'</span></td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">&nbsp;</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">68</td>
<td class="TextItemSigRightMod"><span class="TextSegInsigDiff">&nbsp; &nbsp; </span><span class="TextSegSigDiff">mox.file.copy_parallel(src_url=save_dir,</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">dst_url=s3_results_path)</span></td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">&nbsp;</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">69</td>
<td class="TextItemInsigRightMod">&nbsp;</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">&nbsp;</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">70</td>
<td class="TextItemSigRightMod"><span class="TextSegSigDiff">def</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">main_worker(gpu,</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">ngpus_per_node,</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">args):</span></td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">&nbsp;</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">71</td>
<td class="TextItemSigRightMod"><span class="TextSegInsigDiff">&nbsp; &nbsp; </span><span class="TextSegSigDiff">args.gpu</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">=</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">gpu</span></td>
</tr>
<tr class="SectionEnd">
<td class="TextItemNum AlignRight">&nbsp;</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">72</td>
<td class="TextItemSigRightMod"><span class="TextSegInsigDiff">&nbsp; &nbsp; </span><span class="TextSegSigDiff">dist.init_process_group(backend='nccl',</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">init_method='tcp://0.0.0.0:6666',</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">world_size=ngpus_per_node,</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">rank=args.gpu)</span></td>
</tr>
<tr class="SectionBegin">
<td class="TextItemNum AlignRight">51</td>
<td class="TextItemSame">&nbsp; &nbsp; global best_acc1</td>
<td class="AlignCenter">=</td>
<td class="TextItemNum AlignRight">73</td>
<td class="TextItemSame">&nbsp; &nbsp; global best_acc1</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">52</td>
<td class="TextItemSame">&nbsp; &nbsp; model = models.__dict__[args.arch]()</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">74</td>
<td class="TextItemSame">&nbsp; &nbsp; model = models.__dict__[args.arch]()</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">53</td>
<td class="TextItemSame">&nbsp; &nbsp; if args.gpu is not None:</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">75</td>
<td class="TextItemSame">&nbsp; &nbsp; if args.gpu is not None:</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">54</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; print(&quot;Use GPU: {} for training&quot;.format(args.gpu))</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">76</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; print(&quot;Use GPU: {} for training&quot;.format(args.gpu))</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">55</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; torch.cuda.set_device(args.gpu)</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">77</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; torch.cuda.set_device(args.gpu)</td>
</tr>
<tr class="SectionEnd">
<td class="TextItemNum AlignRight">56</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; model = model.cuda(args.gpu)</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">78</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; model = model.cuda(args.gpu)</td>
</tr>
<tr class="SectionBegin">
<td class="TextItemNum AlignRight">&nbsp;</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">-+</td>
<td class="TextItemNum AlignRight">79</td>
<td class="TextItemSigRightMod"><span class="TextSegInsigDiff">&nbsp; &nbsp; &nbsp; &nbsp; </span><span class="TextSegSigDiff">args.workers</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">=</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">int(args.workers</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">/</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">ngpus_per_node)</span></td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">&nbsp;</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">80</td>
<td class="TextItemSigRightMod"><span class="TextSegInsigDiff">&nbsp; &nbsp; &nbsp; &nbsp; </span><span class="TextSegSigDiff">args.lr</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">=</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">args.lr</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">*</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">ngpus_per_node</span></td>
</tr>
<tr class="SectionEnd">
<td class="TextItemNum AlignRight">&nbsp;</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">81</td>
<td class="TextItemSigRightMod"><span class="TextSegInsigDiff">&nbsp; &nbsp; &nbsp; &nbsp; </span><span class="TextSegSigDiff">model</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">=</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">torch.nn.parallel.DistributedDataParallel(model,</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">device_ids=[args.gpu])</span></td>
</tr>
<tr class="SectionBegin">
<td class="TextItemNum AlignRight">57</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">=</td>
<td class="TextItemNum AlignRight">82</td>
<td class="TextItemSame">&nbsp;</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">58</td>
<td class="TextItemSame">&nbsp; &nbsp; # define loss function (criterion) and optimizer</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">83</td>
<td class="TextItemSame">&nbsp; &nbsp; # define loss function (criterion) and optimizer</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">59</td>
<td class="TextItemSame">&nbsp; &nbsp; criterion = nn.CrossEntropyLoss().cuda(args.gpu)</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">84</td>
<td class="TextItemSame">&nbsp; &nbsp; criterion = nn.CrossEntropyLoss().cuda(args.gpu)</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">60</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">85</td>
<td class="TextItemSame">&nbsp;</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">61</td>
<td class="TextItemSame">&nbsp; &nbsp; optimizer = torch.optim.SGD(model.parameters(), args.lr,</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">86</td>
<td class="TextItemSame">&nbsp; &nbsp; optimizer = torch.optim.SGD(model.parameters(), args.lr,</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">62</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; momentum=args.momentum,</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">87</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; momentum=args.momentum,</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">63</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; weight_decay=args.weight_decay)</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">88</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; weight_decay=args.weight_decay)</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">64</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">89</td>
<td class="TextItemSame">&nbsp;</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">65</td>
<td class="TextItemSame">&nbsp; &nbsp; cudnn.benchmark = True</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">90</td>
<td class="TextItemSame">&nbsp; &nbsp; cudnn.benchmark = True</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">66</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">91</td>
<td class="TextItemSame">&nbsp;</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">67</td>
<td class="TextItemSame">&nbsp; &nbsp; # Data loading code</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">92</td>
<td class="TextItemSame">&nbsp; &nbsp; # Data loading code</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">68</td>
<td class="TextItemSame">&nbsp; &nbsp; traindir = os.path.join(args.data, 'train')</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">93</td>
<td class="TextItemSame">&nbsp; &nbsp; traindir = os.path.join(args.data, 'train')</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">69</td>
<td class="TextItemSame">&nbsp; &nbsp; valdir = os.path.join(args.data, 'eval')</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">94</td>
<td class="TextItemSame">&nbsp; &nbsp; valdir = os.path.join(args.data, 'eval')</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">70</td>
<td class="TextItemSame">&nbsp; &nbsp; normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">95</td>
<td class="TextItemSame">&nbsp; &nbsp; normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">71</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; std=[0.229, 0.224, 0.225])</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">96</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; std=[0.229, 0.224, 0.225])</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">72</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">97</td>
<td class="TextItemSame">&nbsp;</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">73</td>
<td class="TextItemSame">&nbsp; &nbsp; train_dataset = datasets.ImageFolder(</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">98</td>
<td class="TextItemSame">&nbsp; &nbsp; train_dataset = datasets.ImageFolder(</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">74</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; traindir,</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">99</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; traindir,</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">75</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; transforms.Compose([</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">100</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; transforms.Compose([</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">76</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; transforms.RandomResizedCrop(224),</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">101</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; transforms.RandomResizedCrop(224),</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">77</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; transforms.RandomHorizontalFlip(),</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">102</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; transforms.RandomHorizontalFlip(),</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">78</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; transforms.ToTensor(),</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">103</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; transforms.ToTensor(),</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">79</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; normalize,</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">104</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; normalize,</td>
</tr>
<tr class="SectionEnd">
<td class="TextItemNum AlignRight">80</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; ]))</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">105</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; ]))</td>
</tr>
<tr class="SectionAll">
<td class="TextItemNum AlignRight">81</td>
<td class="TextItemSigDiffMod">&nbsp;</td>
<td class="AlignCenter">&lt;&gt;</td>
<td class="TextItemNum AlignRight">106</td>
<td class="TextItemSigDiffMod"><span class="TextSegInsigDiff">&nbsp; &nbsp; </span><span class="TextSegSigDiff">train_sampler</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">=</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">torch.utils.data.distributed.DistributedSampler(train_dataset)</span></td>
</tr>
<tr class="SectionAll">
<td class="TextItemNum AlignRight">82</td>
<td class="TextItemSame">&nbsp; &nbsp; train_loader = torch.utils.data.DataLoader(</td>
<td class="AlignCenter">=</td>
<td class="TextItemNum AlignRight">107</td>
<td class="TextItemSame">&nbsp; &nbsp; train_loader = torch.utils.data.DataLoader(</td>
</tr>
<tr class="SectionBegin">
<td class="TextItemNum AlignRight">83</td>
<td class="TextItemSigDiffMod">&nbsp; &nbsp; &nbsp; &nbsp; train_dataset, batch_size=args.batch_size, shuffle=<span class="TextSegSigDiff">Tru</span>e,</td>
<td class="AlignCenter">&lt;&gt;</td>
<td class="TextItemNum AlignRight">108</td>
<td class="TextItemSigDiffMod">&nbsp; &nbsp; &nbsp; &nbsp; train_dataset, batch_size=args.batch_size, shuffle=<span class="TextSegSigDiff">Fals</span>e,</td>
</tr>
<tr class="SectionEnd">
<td class="TextItemNum AlignRight">84</td>
<td class="TextItemSigDiffMod">&nbsp; &nbsp; &nbsp; &nbsp; num_workers=args.workers, pin_memory=True, sampler=<span class="TextSegSigDiff">Non</span><span class="TextSegSigDiff">e</span>)</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">109</td>
<td class="TextItemSigDiffMod">&nbsp; &nbsp; &nbsp; &nbsp; num_workers=args.workers, pin_memory=True, sampler=<span class="TextSegSigDiff">train_sampler</span>)</td>
</tr>
<tr class="SectionBegin">
<td class="TextItemNum AlignRight">85</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">=</td>
<td class="TextItemNum AlignRight">110</td>
<td class="TextItemSame">&nbsp;</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">86</td>
<td class="TextItemSame">&nbsp; &nbsp; val_dataset = datasets.ImageFolder(</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">111</td>
<td class="TextItemSame">&nbsp; &nbsp; val_dataset = datasets.ImageFolder(</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">87</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; valdir,</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">112</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; valdir,</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">88</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; transforms.Compose([</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">113</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; transforms.Compose([</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">89</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; transforms.Resize(256),</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">114</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; transforms.Resize(256),</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">90</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; transforms.CenterCrop(224),</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">115</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; transforms.CenterCrop(224),</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">91</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; transforms.ToTensor(),</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">116</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; transforms.ToTensor(),</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">92</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; normalize,</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">117</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; normalize,</td>
</tr>
<tr class="SectionEnd">
<td class="TextItemNum AlignRight">93</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; ]))</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">118</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; ]))</td>
</tr>
<tr class="SectionAll">
<td class="TextItemNum AlignRight">&nbsp;</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">-+</td>
<td class="TextItemNum AlignRight">119</td>
<td class="TextItemSigRightMod"><span class="TextSegInsigDiff">&nbsp; &nbsp; </span><span class="TextSegSigDiff">val_sampler</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">=</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">torch.utils.data.distributed.DistributedSampler(val_dataset)</span></td>
</tr>
<tr class="SectionBegin">
<td class="TextItemNum AlignRight">94</td>
<td class="TextItemSame">&nbsp; &nbsp; val_loader = torch.utils.data.DataLoader(</td>
<td class="AlignCenter">=</td>
<td class="TextItemNum AlignRight">120</td>
<td class="TextItemSame">&nbsp; &nbsp; val_loader = torch.utils.data.DataLoader(</td>
</tr>
<tr class="SectionEnd">
<td class="TextItemNum AlignRight">95</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; val_dataset, batch_size=args.batch_size, shuffle=False,</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">121</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; val_dataset, batch_size=args.batch_size, shuffle=False,</td>
</tr>
<tr class="SectionAll">
<td class="TextItemNum AlignRight">96</td>
<td class="TextItemSigDiffMod">&nbsp; &nbsp; &nbsp; &nbsp; num_workers=args.workers, pin_memory=True, sampler=<span class="TextSegSigDiff">Non</span><span class="TextSegSigDiff">e</span>)</td>
<td class="AlignCenter">&lt;&gt;</td>
<td class="TextItemNum AlignRight">122</td>
<td class="TextItemSigDiffMod">&nbsp; &nbsp; &nbsp; &nbsp; num_workers=args.workers, pin_memory=True, sampler=<span class="TextSegSigDiff">val_sampler</span>)</td>
</tr>
<tr class="SectionBegin">
<td class="TextItemNum AlignRight">97</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">=</td>
<td class="TextItemNum AlignRight">123</td>
<td class="TextItemSame">&nbsp;</td>
</tr>
<tr class="SectionEnd">
<td class="TextItemNum AlignRight">98</td>
<td class="TextItemSame">&nbsp; &nbsp; for epoch in range(0, args.epochs):</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">124</td>
<td class="TextItemSame">&nbsp; &nbsp; for epoch in range(0, args.epochs):</td>
</tr>
<tr class="SectionBegin">
<td class="TextItemNum AlignRight">&nbsp;</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">-+</td>
<td class="TextItemNum AlignRight">125</td>
<td class="TextItemSigRightMod"><span class="TextSegInsigDiff">&nbsp; &nbsp; &nbsp; &nbsp; </span><span class="TextSegSigDiff">train_sampler.set_epoch(epoch)</span></td>
</tr>
<tr class="SectionEnd">
<td class="TextItemNum AlignRight">&nbsp;</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">126</td>
<td class="TextItemInsigRightMod">&nbsp;</td>
</tr>
<tr class="SectionBegin">
<td class="TextItemNum AlignRight">99</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; adjust_learning_rate(optimizer, epoch, args)</td>
<td class="AlignCenter">=</td>
<td class="TextItemNum AlignRight">127</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; adjust_learning_rate(optimizer, epoch, args)</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">100</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">128</td>
<td class="TextItemSame">&nbsp;</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">101</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; # train for one epoch</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">129</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; # train for one epoch</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">102</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; train(train_loader, model, criterion, optimizer, epoch, args)</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">130</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; train(train_loader, model, criterion, optimizer, epoch, args)</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">103</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">131</td>
<td class="TextItemSame">&nbsp;</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">104</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; # evaluate on validation set</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">132</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; # evaluate on validation set</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">105</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; acc1 = validate(val_loader, model, criterion, args)</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">133</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; acc1 = validate(val_loader, model, criterion, args)</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">106</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">134</td>
<td class="TextItemSame">&nbsp;</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">107</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; # remember best acc@1 and save checkpoint</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">135</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; # remember best acc@1 and save checkpoint</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">108</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; is_best = acc1 &gt; best_acc1</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">136</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; is_best = acc1 &gt; best_acc1</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">109</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; best_acc1 = max(acc1, best_acc1)</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">137</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; best_acc1 = max(acc1, best_acc1)</td>
</tr>
<tr class="SectionEnd">
<td class="TextItemNum AlignRight">110</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">138</td>
<td class="TextItemSame">&nbsp;</td>
</tr>
<tr class="SectionBegin">
<td class="TextItemNum AlignRight">&nbsp;</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&lt;&gt;</td>
<td class="TextItemNum AlignRight">139</td>
<td class="TextItemSigRightMod"><span class="TextSegInsigDiff">&nbsp; &nbsp; &nbsp; &nbsp; </span><span class="TextSegSigDiff">if</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">args.gpu</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">==</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">0:</span></td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">111</td>
<td class="TextItemInsigDiffMod">&nbsp; &nbsp; &nbsp; &nbsp; save_checkpoint({</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">140</td>
<td class="TextItemInsigDiffMod">&nbsp; &nbsp; &nbsp; &nbsp; <span class="TextSegInsigDiff">&nbsp; &nbsp; </span>save_checkpoint({</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">112</td>
<td class="TextItemInsigDiffMod">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 'epoch': epoch + 1,</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">141</td>
<td class="TextItemInsigDiffMod">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; <span class="TextSegInsigDiff">&nbsp; &nbsp; </span>'epoch': epoch + 1,</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">113</td>
<td class="TextItemInsigDiffMod">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 'arch': args.arch,</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">142</td>
<td class="TextItemInsigDiffMod">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; <span class="TextSegInsigDiff">&nbsp; &nbsp; </span>'arch': args.arch,</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">114</td>
<td class="TextItemInsigDiffMod">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 'state_dict': model.state_dict(),</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">143</td>
<td class="TextItemInsigDiffMod">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; <span class="TextSegInsigDiff">&nbsp; &nbsp; </span>'state_dict': model.state_dict(),</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">115</td>
<td class="TextItemInsigDiffMod">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 'best_acc1': best_acc1,</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">144</td>
<td class="TextItemInsigDiffMod">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; <span class="TextSegInsigDiff">&nbsp; &nbsp; </span>'best_acc1': best_acc1,</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">116</td>
<td class="TextItemInsigDiffMod">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 'optimizer' : optimizer.state_dict(),</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">145</td>
<td class="TextItemInsigDiffMod">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; <span class="TextSegInsigDiff">&nbsp; &nbsp; </span>'optimizer' : optimizer.state_dict(),</td>
</tr>
<tr class="SectionEnd">
<td class="TextItemNum AlignRight">117</td>
<td class="TextItemInsigDiffMod">&nbsp; &nbsp; &nbsp; &nbsp; }, is_best)</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">146</td>
<td class="TextItemInsigDiffMod">&nbsp; &nbsp; &nbsp; &nbsp; <span class="TextSegInsigDiff">&nbsp; &nbsp; </span>}, is_best)</td>
</tr>
<tr class="SectionBegin">
<td class="TextItemNum AlignRight">118</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">=</td>
<td class="TextItemNum AlignRight">147</td>
<td class="TextItemSame">&nbsp;</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">119</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">148</td>
<td class="TextItemSame">&nbsp;</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">120</td>
<td class="TextItemSame">def train(train_loader, model, criterion, optimizer, epoch, args):</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">149</td>
<td class="TextItemSame">def train(train_loader, model, criterion, optimizer, epoch, args):</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">121</td>
<td class="TextItemSame">&nbsp; &nbsp; batch_time = AverageMeter('Time', ':6.3f')</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">150</td>
<td class="TextItemSame">&nbsp; &nbsp; batch_time = AverageMeter('Time', ':6.3f')</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">122</td>
<td class="TextItemSame">&nbsp; &nbsp; data_time = AverageMeter('Data', ':6.3f')</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">151</td>
<td class="TextItemSame">&nbsp; &nbsp; data_time = AverageMeter('Data', ':6.3f')</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">123</td>
<td class="TextItemSame">&nbsp; &nbsp; losses = AverageMeter('Loss', ':.4e')</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">152</td>
<td class="TextItemSame">&nbsp; &nbsp; losses = AverageMeter('Loss', ':.4e')</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">124</td>
<td class="TextItemSame">&nbsp; &nbsp; top1 = AverageMeter('Acc@1', ':6.2f')</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">153</td>
<td class="TextItemSame">&nbsp; &nbsp; top1 = AverageMeter('Acc@1', ':6.2f')</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">125</td>
<td class="TextItemSame">&nbsp; &nbsp; top5 = AverageMeter('Acc@5', ':6.2f')</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">154</td>
<td class="TextItemSame">&nbsp; &nbsp; top5 = AverageMeter('Acc@5', ':6.2f')</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">126</td>
<td class="TextItemSame">&nbsp; &nbsp; progress = ProgressMeter(</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">155</td>
<td class="TextItemSame">&nbsp; &nbsp; progress = ProgressMeter(</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">127</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; len(train_loader),</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">156</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; len(train_loader),</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">128</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; [batch_time, data_time, losses, top1, top5],</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">157</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; [batch_time, data_time, losses, top1, top5],</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">129</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; prefix=&quot;Epoch: [{}]&quot;.format(epoch))</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">158</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; prefix=&quot;Epoch: [{}]&quot;.format(epoch))</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">130</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">159</td>
<td class="TextItemSame">&nbsp;</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">131</td>
<td class="TextItemSame">&nbsp; &nbsp; # switch to train mode</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">160</td>
<td class="TextItemSame">&nbsp; &nbsp; # switch to train mode</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">132</td>
<td class="TextItemSame">&nbsp; &nbsp; model.train()</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">161</td>
<td class="TextItemSame">&nbsp; &nbsp; model.train()</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">133</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">162</td>
<td class="TextItemSame">&nbsp;</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">134</td>
<td class="TextItemSame">&nbsp; &nbsp; end = time.time()</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">163</td>
<td class="TextItemSame">&nbsp; &nbsp; end = time.time()</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">135</td>
<td class="TextItemSame">&nbsp; &nbsp; for i, (images, target) in enumerate(train_loader):</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">164</td>
<td class="TextItemSame">&nbsp; &nbsp; for i, (images, target) in enumerate(train_loader):</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">136</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; # measure data loading time</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">165</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; # measure data loading time</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">137</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; data_time.update(time.time() - end)</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">166</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; data_time.update(time.time() - end)</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">138</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">167</td>
<td class="TextItemSame">&nbsp;</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">139</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; if args.gpu is not None:</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">168</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; if args.gpu is not None:</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">140</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; images = images.cuda(args.gpu, non_blocking=True)</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">169</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; images = images.cuda(args.gpu, non_blocking=True)</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">141</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; target = target.cuda(args.gpu, non_blocking=True)</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">170</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; target = target.cuda(args.gpu, non_blocking=True)</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">142</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">171</td>
<td class="TextItemSame">&nbsp;</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">143</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; # compute output</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">172</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; # compute output</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">144</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; output = model(images)</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">173</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; output = model(images)</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">145</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; loss = criterion(output, target)</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">174</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; loss = criterion(output, target)</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">146</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">175</td>
<td class="TextItemSame">&nbsp;</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">147</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; # measure accuracy and record loss</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">176</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; # measure accuracy and record loss</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">148</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; acc1, acc5 = accuracy(output, target, topk=(1, 5))</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">177</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; acc1, acc5 = accuracy(output, target, topk=(1, 5))</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">149</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; losses.update(loss.item(), images.size(0))</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">178</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; losses.update(loss.item(), images.size(0))</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">150</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; top1.update(acc1[0], images.size(0))</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">179</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; top1.update(acc1[0], images.size(0))</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">151</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; top5.update(acc5[0], images.size(0))</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">180</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; top5.update(acc5[0], images.size(0))</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">152</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">181</td>
<td class="TextItemSame">&nbsp;</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">153</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; # compute gradient and do SGD step</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">182</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; # compute gradient and do SGD step</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">154</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; optimizer.zero_grad()</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">183</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; optimizer.zero_grad()</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">155</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; loss.backward()</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">184</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; loss.backward()</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">156</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; optimizer.step()</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">185</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; optimizer.step()</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">157</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">186</td>
<td class="TextItemSame">&nbsp;</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">158</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; # measure elapsed time</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">187</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; # measure elapsed time</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">159</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; batch_time.update(time.time() - end)</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">188</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; batch_time.update(time.time() - end)</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">160</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; end = time.time()</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">189</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; end = time.time()</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">161</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">190</td>
<td class="TextItemSame">&nbsp;</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">162</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; if i % args.print_freq == 0:</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">191</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; if i % args.print_freq == 0:</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">163</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; progress.display(i)</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">192</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; progress.display(i)</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">164</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">193</td>
<td class="TextItemSame">&nbsp;</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">165</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">194</td>
<td class="TextItemSame">&nbsp;</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">166</td>
<td class="TextItemSame">def validate(val_loader, model, criterion, args):</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">195</td>
<td class="TextItemSame">def validate(val_loader, model, criterion, args):</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">167</td>
<td class="TextItemSame">&nbsp; &nbsp; batch_time = AverageMeter('Time', ':6.3f')</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">196</td>
<td class="TextItemSame">&nbsp; &nbsp; batch_time = AverageMeter('Time', ':6.3f')</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">168</td>
<td class="TextItemSame">&nbsp; &nbsp; losses = AverageMeter('Loss', ':.4e')</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">197</td>
<td class="TextItemSame">&nbsp; &nbsp; losses = AverageMeter('Loss', ':.4e')</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">169</td>
<td class="TextItemSame">&nbsp; &nbsp; top1 = AverageMeter('Acc@1', ':6.2f')</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">198</td>
<td class="TextItemSame">&nbsp; &nbsp; top1 = AverageMeter('Acc@1', ':6.2f')</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">170</td>
<td class="TextItemSame">&nbsp; &nbsp; top5 = AverageMeter('Acc@5', ':6.2f')</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">199</td>
<td class="TextItemSame">&nbsp; &nbsp; top5 = AverageMeter('Acc@5', ':6.2f')</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">171</td>
<td class="TextItemSame">&nbsp; &nbsp; progress = ProgressMeter(</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">200</td>
<td class="TextItemSame">&nbsp; &nbsp; progress = ProgressMeter(</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">172</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; len(val_loader),</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">201</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; len(val_loader),</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">173</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; [batch_time, losses, top1, top5],</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">202</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; [batch_time, losses, top1, top5],</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">174</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; prefix='Test: ')</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">203</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; prefix='Test: ')</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">175</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">204</td>
<td class="TextItemSame">&nbsp;</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">176</td>
<td class="TextItemSame">&nbsp; &nbsp; # switch to evaluate mode</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">205</td>
<td class="TextItemSame">&nbsp; &nbsp; # switch to evaluate mode</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">177</td>
<td class="TextItemSame">&nbsp; &nbsp; model.eval()</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">206</td>
<td class="TextItemSame">&nbsp; &nbsp; model.eval()</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">178</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">207</td>
<td class="TextItemSame">&nbsp;</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">179</td>
<td class="TextItemSame">&nbsp; &nbsp; with torch.no_grad():</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">208</td>
<td class="TextItemSame">&nbsp; &nbsp; with torch.no_grad():</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">180</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; end = time.time()</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">209</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; end = time.time()</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">181</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; for i, (images, target) in enumerate(val_loader):</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">210</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; for i, (images, target) in enumerate(val_loader):</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">182</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; if args.gpu is not None:</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">211</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; if args.gpu is not None:</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">183</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; images = images.cuda(args.gpu, non_blocking=True)</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">212</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; images = images.cuda(args.gpu, non_blocking=True)</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">184</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; target = target.cuda(args.gpu, non_blocking=True)</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">213</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; target = target.cuda(args.gpu, non_blocking=True)</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">185</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">214</td>
<td class="TextItemSame">&nbsp;</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">186</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; # compute output</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">215</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; # compute output</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">187</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; output = model(images)</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">216</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; output = model(images)</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">188</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; loss = criterion(output, target)</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">217</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; loss = criterion(output, target)</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">189</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">218</td>
<td class="TextItemSame">&nbsp;</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">190</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; # measure accuracy and record loss</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">219</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; # measure accuracy and record loss</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">191</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; acc1, acc5 = accuracy(output, target, topk=(1, 5))</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">220</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; acc1, acc5 = accuracy(output, target, topk=(1, 5))</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">192</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; losses.update(loss.item(), images.size(0))</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">221</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; losses.update(loss.item(), images.size(0))</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">193</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; top1.update(acc1[0], images.size(0))</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">222</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; top1.update(acc1[0], images.size(0))</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">194</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; top5.update(acc5[0], images.size(0))</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">223</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; top5.update(acc5[0], images.size(0))</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">195</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">224</td>
<td class="TextItemSame">&nbsp;</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">196</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; # measure elapsed time</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">225</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; # measure elapsed time</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">197</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; batch_time.update(time.time() - end)</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">226</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; batch_time.update(time.time() - end)</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">198</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; end = time.time()</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">227</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; end = time.time()</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">199</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">228</td>
<td class="TextItemSame">&nbsp;</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">200</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; if i % args.print_freq == 0:</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">229</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; if i % args.print_freq == 0:</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">201</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; progress.display(i)</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">230</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; progress.display(i)</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">202</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">231</td>
<td class="TextItemSame">&nbsp;</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">203</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; # TODO: this should also be done with the ProgressMeter</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">232</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; # TODO: this should also be done with the ProgressMeter</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">204</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">233</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">205</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; .format(top1=top1, top5=top5))</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">234</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; .format(top1=top1, top5=top5))</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">206</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">235</td>
<td class="TextItemSame">&nbsp;</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">207</td>
<td class="TextItemSame">&nbsp; &nbsp; return top1.avg</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">236</td>
<td class="TextItemSame">&nbsp; &nbsp; return top1.avg</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">208</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">237</td>
<td class="TextItemSame">&nbsp;</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">209</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">238</td>
<td class="TextItemSame">&nbsp;</td>
</tr>
<tr class="SectionEnd">
<td class="TextItemNum AlignRight">210</td>
<td class="TextItemSame">def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">239</td>
<td class="TextItemSame">def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):</td>
</tr>
<tr class="SectionAll">
<td class="TextItemNum AlignRight">&nbsp;</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">-+</td>
<td class="TextItemNum AlignRight">240</td>
<td class="TextItemSigRightMod"><span class="TextSegInsigDiff">&nbsp; &nbsp; </span><span class="TextSegSigDiff">filename</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">=</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">'/cache/results/checkpoint.pth.tar'</span></td>
</tr>
<tr class="SectionBegin">
<td class="TextItemNum AlignRight">211</td>
<td class="TextItemSame">&nbsp; &nbsp; torch.save(state, filename)</td>
<td class="AlignCenter">=</td>
<td class="TextItemNum AlignRight">241</td>
<td class="TextItemSame">&nbsp; &nbsp; torch.save(state, filename)</td>
</tr>
<tr class="SectionEnd">
<td class="TextItemNum AlignRight">212</td>
<td class="TextItemSame">&nbsp; &nbsp; if is_best:</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">242</td>
<td class="TextItemSame">&nbsp; &nbsp; if is_best:</td>
</tr>
<tr class="SectionAll">
<td class="TextItemNum AlignRight">213</td>
<td class="TextItemSigDiffMod">&nbsp; &nbsp; &nbsp; &nbsp; shutil.copyfile(filename, 'model_best.pth.tar')</td>
<td class="AlignCenter">&lt;&gt;</td>
<td class="TextItemNum AlignRight">243</td>
<td class="TextItemSigDiffMod">&nbsp; &nbsp; &nbsp; &nbsp; shutil.copyfile(filename, '<span class="TextSegSigDiff">/cache/results/</span>model_best.pth.tar')</td>
</tr>
<tr class="SectionBegin">
<td class="TextItemNum AlignRight">214</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">=</td>
<td class="TextItemNum AlignRight">244</td>
<td class="TextItemSame">&nbsp;</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">215</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">245</td>
<td class="TextItemSame">&nbsp;</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">216</td>
<td class="TextItemSame">class AverageMeter(object):</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">246</td>
<td class="TextItemSame">class AverageMeter(object):</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">217</td>
<td class="TextItemSame">&nbsp; &nbsp; &quot;&quot;&quot;Computes and stores the average and current value&quot;&quot;&quot;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">247</td>
<td class="TextItemSame">&nbsp; &nbsp; &quot;&quot;&quot;Computes and stores the average and current value&quot;&quot;&quot;</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">218</td>
<td class="TextItemSame">&nbsp; &nbsp; def __init__(self, name, fmt=':f'):</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">248</td>
<td class="TextItemSame">&nbsp; &nbsp; def __init__(self, name, fmt=':f'):</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">219</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; self.name = name</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">249</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; self.name = name</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">220</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; self.fmt = fmt</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">250</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; self.fmt = fmt</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">221</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; self.reset()</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">251</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; self.reset()</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">222</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">252</td>
<td class="TextItemSame">&nbsp;</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">223</td>
<td class="TextItemSame">&nbsp; &nbsp; def reset(self):</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">253</td>
<td class="TextItemSame">&nbsp; &nbsp; def reset(self):</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">224</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; self.val = 0</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">254</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; self.val = 0</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">225</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; self.avg = 0</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">255</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; self.avg = 0</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">226</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; self.sum = 0</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">256</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; self.sum = 0</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">227</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; self.count = 0</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">257</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; self.count = 0</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">228</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">258</td>
<td class="TextItemSame">&nbsp;</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">229</td>
<td class="TextItemSame">&nbsp; &nbsp; def update(self, val, n=1):</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">259</td>
<td class="TextItemSame">&nbsp; &nbsp; def update(self, val, n=1):</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">230</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; self.val = val</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">260</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; self.val = val</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">231</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; self.sum += val * n</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">261</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; self.sum += val * n</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">232</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; self.count += n</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">262</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; self.count += n</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">233</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; self.avg = self.sum / self.count</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">263</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; self.avg = self.sum / self.count</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">234</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">264</td>
<td class="TextItemSame">&nbsp;</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">235</td>
<td class="TextItemSame">&nbsp; &nbsp; def __str__(self):</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">265</td>
<td class="TextItemSame">&nbsp; &nbsp; def __str__(self):</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">236</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">266</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">237</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; return fmtstr.format(**self.__dict__)</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">267</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; return fmtstr.format(**self.__dict__)</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">238</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">268</td>
<td class="TextItemSame">&nbsp;</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">239</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">269</td>
<td class="TextItemSame">&nbsp;</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">240</td>
<td class="TextItemSame">class ProgressMeter(object):</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">270</td>
<td class="TextItemSame">class ProgressMeter(object):</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">241</td>
<td class="TextItemSame">&nbsp; &nbsp; def __init__(self, num_batches, meters, prefix=&quot;&quot;):</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">271</td>
<td class="TextItemSame">&nbsp; &nbsp; def __init__(self, num_batches, meters, prefix=&quot;&quot;):</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">242</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; self.batch_fmtstr = self._get_batch_fmtstr(num_batches)</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">272</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; self.batch_fmtstr = self._get_batch_fmtstr(num_batches)</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">243</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; self.meters = meters</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">273</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; self.meters = meters</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">244</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; self.prefix = prefix</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">274</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; self.prefix = prefix</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">245</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">275</td>
<td class="TextItemSame">&nbsp;</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">246</td>
<td class="TextItemSame">&nbsp; &nbsp; def display(self, batch):</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">276</td>
<td class="TextItemSame">&nbsp; &nbsp; def display(self, batch):</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">247</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; entries = [self.prefix + self.batch_fmtstr.format(batch)]</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">277</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; entries = [self.prefix + self.batch_fmtstr.format(batch)]</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">248</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; entries += [str(meter) for meter in self.meters]</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">278</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; entries += [str(meter) for meter in self.meters]</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">249</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; print('\t'.join(entries))</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">279</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; print('\t'.join(entries))</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">250</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">280</td>
<td class="TextItemSame">&nbsp;</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">251</td>
<td class="TextItemSame">&nbsp; &nbsp; def _get_batch_fmtstr(self, num_batches):</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">281</td>
<td class="TextItemSame">&nbsp; &nbsp; def _get_batch_fmtstr(self, num_batches):</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">252</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; num_digits = len(str(num_batches // 1))</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">282</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; num_digits = len(str(num_batches // 1))</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">253</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; fmt = '{:' + str(num_digits) + 'd}'</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">283</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; fmt = '{:' + str(num_digits) + 'd}'</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">254</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; return '[' + fmt + '/' + fmt.format(num_batches) + ']'</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">284</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; return '[' + fmt + '/' + fmt.format(num_batches) + ']'</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">255</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">285</td>
<td class="TextItemSame">&nbsp;</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">256</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">286</td>
<td class="TextItemSame">&nbsp;</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">257</td>
<td class="TextItemSame">def adjust_learning_rate(optimizer, epoch, args):</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">287</td>
<td class="TextItemSame">def adjust_learning_rate(optimizer, epoch, args):</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">258</td>
<td class="TextItemSame">&nbsp; &nbsp; &quot;&quot;&quot;Sets the learning rate to the initial LR decayed by 10 every 30 epochs&quot;&quot;&quot;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">288</td>
<td class="TextItemSame">&nbsp; &nbsp; &quot;&quot;&quot;Sets the learning rate to the initial LR decayed by 10 every 30 epochs&quot;&quot;&quot;</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">259</td>
<td class="TextItemSame">&nbsp; &nbsp; lr = args.lr * (0.1 ** (epoch // 30))</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">289</td>
<td class="TextItemSame">&nbsp; &nbsp; lr = args.lr * (0.1 ** (epoch // 30))</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">260</td>
<td class="TextItemSame">&nbsp; &nbsp; for param_group in optimizer.param_groups:</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">290</td>
<td class="TextItemSame">&nbsp; &nbsp; for param_group in optimizer.param_groups:</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">261</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; param_group['lr'] = lr</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">291</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; param_group['lr'] = lr</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">262</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">292</td>
<td class="TextItemSame">&nbsp;</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">263</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">293</td>
<td class="TextItemSame">&nbsp;</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">264</td>
<td class="TextItemSame">def accuracy(output, target, topk=(1,)):</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">294</td>
<td class="TextItemSame">def accuracy(output, target, topk=(1,)):</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">265</td>
<td class="TextItemSame">&nbsp; &nbsp; &quot;&quot;&quot;Computes the accuracy over the k top predictions for the specified values of k&quot;&quot;&quot;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">295</td>
<td class="TextItemSame">&nbsp; &nbsp; &quot;&quot;&quot;Computes the accuracy over the k top predictions for the specified values of k&quot;&quot;&quot;</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">266</td>
<td class="TextItemSame">&nbsp; &nbsp; with torch.no_grad():</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">296</td>
<td class="TextItemSame">&nbsp; &nbsp; with torch.no_grad():</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">267</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; maxk = max(topk)</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">297</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; maxk = max(topk)</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">268</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; batch_size = target.size(0)</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">298</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; batch_size = target.size(0)</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">269</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">299</td>
<td class="TextItemSame">&nbsp;</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">270</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; _, pred = output.topk(maxk, 1, True, True)</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">300</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; _, pred = output.topk(maxk, 1, True, True)</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">271</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; pred = pred.t()</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">301</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; pred = pred.t()</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">272</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; correct = pred.eq(target.view(1, -1).expand_as(pred))</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">302</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; correct = pred.eq(target.view(1, -1).expand_as(pred))</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">273</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">303</td>
<td class="TextItemSame">&nbsp;</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">274</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; res = []</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">304</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; res = []</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">275</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; for k in topk:</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">305</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; for k in topk:</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">276</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">306</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">277</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; res.append(correct_k.mul_(100.0 / batch_size))</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">307</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; res.append(correct_k.mul_(100.0 / batch_size))</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">278</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; return res</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">308</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; return res</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">279</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">309</td>
<td class="TextItemSame">&nbsp;</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">280</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">310</td>
<td class="TextItemSame">&nbsp;</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">281</td>
<td class="TextItemSame">if __name__ == '__main__':</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">311</td>
<td class="TextItemSame">if __name__ == '__main__':</td>
</tr>
<tr class="SectionEnd">
<td class="TextItemNum AlignRight">282</td>
<td class="TextItemSame">&nbsp; &nbsp; main()</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">312</td>
<td class="TextItemSame">&nbsp; &nbsp; main()</td>
</tr>
</table>


## horovod实现单机多卡

左侧为线下单机单卡代码，右侧为线上单机多卡代码

注：线上horovod分布式，只需修改下载数据的进程编号，保证每个计算节点有且只有一个进程下载数据即可

<table class="fc" cellspacing="0" cellpadding="0">
<tr class="SectionBegin">
<td class="TextItemNum AlignRight">1</td>
<td class="TextItemSame">import argparse</td>
<td class="AlignCenter">=</td>
<td class="TextItemNum AlignRight">1</td>
<td class="TextItemSame">import argparse</td>
</tr>
<tr class="SectionEnd">
<td class="TextItemNum AlignRight">2</td>
<td class="TextItemSame">import os</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">2</td>
<td class="TextItemSame">import os</td>
</tr>
<tr class="SectionAll">
<td class="TextItemNum AlignRight">&nbsp;</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">-+</td>
<td class="TextItemNum AlignRight">3</td>
<td class="TextItemSigRightMod"><span class="TextSegSigDiff">os.environ[&quot;CUDA_VISIBLE_DEVICES&quot;]</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">=</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">'0,</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">1'</span></td>
</tr>
<tr class="SectionBegin">
<td class="TextItemNum AlignRight">3</td>
<td class="TextItemSame">import shutil</td>
<td class="AlignCenter">=</td>
<td class="TextItemNum AlignRight">4</td>
<td class="TextItemSame">import shutil</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">4</td>
<td class="TextItemSame">import time</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">5</td>
<td class="TextItemSame">import time</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">5</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">6</td>
<td class="TextItemSame">&nbsp;</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">6</td>
<td class="TextItemSame">import torch</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">7</td>
<td class="TextItemSame">import torch</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">7</td>
<td class="TextItemSame">import torch.nn as nn</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">8</td>
<td class="TextItemSame">import torch.nn as nn</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">8</td>
<td class="TextItemSame">import torch.backends.cudnn as cudnn</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">9</td>
<td class="TextItemSame">import torch.backends.cudnn as cudnn</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">9</td>
<td class="TextItemSame">import torch.optim</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">10</td>
<td class="TextItemSame">import torch.optim</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">10</td>
<td class="TextItemSame">import torch.utils.data</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">11</td>
<td class="TextItemSame">import torch.utils.data</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">11</td>
<td class="TextItemSame">import torchvision.transforms as transforms</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">12</td>
<td class="TextItemSame">import torchvision.transforms as transforms</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">12</td>
<td class="TextItemSame">import torchvision.datasets as datasets</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">13</td>
<td class="TextItemSame">import torchvision.datasets as datasets</td>
</tr>
<tr class="SectionEnd">
<td class="TextItemNum AlignRight">13</td>
<td class="TextItemSame">import torchvision.models as models</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">14</td>
<td class="TextItemSame">import torchvision.models as models</td>
</tr>
<tr class="SectionBegin">
<td class="TextItemNum AlignRight">&nbsp;</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">-+</td>
<td class="TextItemNum AlignRight">15</td>
<td class="TextItemInsigRightMod">&nbsp;</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">&nbsp;</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">16</td>
<td class="TextItemSigRightMod"><span class="TextSegSigDiff">import</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">horovod.torch</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">as</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">hvd</span></td>
</tr>
<tr class="SectionEnd">
<td class="TextItemNum AlignRight">&nbsp;</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">17</td>
<td class="TextItemSigRightMod"><span class="TextSegSigDiff">hvd.init()</span></td>
</tr>
<tr class="SectionBegin">
<td class="TextItemNum AlignRight">14</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">=</td>
<td class="TextItemNum AlignRight">18</td>
<td class="TextItemSame">&nbsp;</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">15</td>
<td class="TextItemSame">model_names = sorted(name for name in models.__dict__</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">19</td>
<td class="TextItemSame">model_names = sorted(name for name in models.__dict__</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">16</td>
<td class="TextItemSame">&nbsp; &nbsp; if name.islower() and not name.startswith(&quot;__&quot;)</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">20</td>
<td class="TextItemSame">&nbsp; &nbsp; if name.islower() and not name.startswith(&quot;__&quot;)</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">17</td>
<td class="TextItemSame">&nbsp; &nbsp; and callable(models.__dict__[name]))</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">21</td>
<td class="TextItemSame">&nbsp; &nbsp; and callable(models.__dict__[name]))</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">18</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">22</td>
<td class="TextItemSame">&nbsp;</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">19</td>
<td class="TextItemSame">parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">23</td>
<td class="TextItemSame">parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">20</td>
<td class="TextItemSame">parser.add_argument('--data', help='path to dataset')</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">24</td>
<td class="TextItemSame">parser.add_argument('--data', help='path to dataset')</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">21</td>
<td class="TextItemSame">parser.add_argument('--arch', default='resnet18',</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">25</td>
<td class="TextItemSame">parser.add_argument('--arch', default='resnet18',</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">22</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; choices=model_names,</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">26</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; choices=model_names,</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">23</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; help='model architecture: ' +</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">27</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; help='model architecture: ' +</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">24</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; ' | '.join(model_names) +</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">28</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; ' | '.join(model_names) +</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">25</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; ' (default: resnet18)')</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">29</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; ' (default: resnet18)')</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">26</td>
<td class="TextItemSame">parser.add_argument('--workers', default=4, type=int, metavar='N',</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">30</td>
<td class="TextItemSame">parser.add_argument('--workers', default=4, type=int, metavar='N',</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">27</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; help='number of data loading workers (default: 4)')</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">31</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; help='number of data loading workers (default: 4)')</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">28</td>
<td class="TextItemSame">parser.add_argument('--print-freq', default=20, type=int,</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">32</td>
<td class="TextItemSame">parser.add_argument('--print-freq', default=20, type=int,</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">29</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; metavar='N', help='print frequency (default: 10)')</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">33</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; metavar='N', help='print frequency (default: 10)')</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">30</td>
<td class="TextItemSame">parser.add_argument('--epochs', default=10, type=int, metavar='N',</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">34</td>
<td class="TextItemSame">parser.add_argument('--epochs', default=10, type=int, metavar='N',</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">31</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; help='number of total epochs to run')</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">35</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; help='number of total epochs to run')</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">32</td>
<td class="TextItemSame">parser.add_argument('--batch_size', default=256, type=int,</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">36</td>
<td class="TextItemSame">parser.add_argument('--batch_size', default=256, type=int,</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">33</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; metavar='N',</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">37</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; metavar='N',</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">34</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; help='mini-batch size (default: 256), this is the total '</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">38</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; help='mini-batch size (default: 256), this is the total '</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">35</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; 'batch size of all GPUs on the current node when '</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">39</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; 'batch size of all GPUs on the current node when '</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">36</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; 'using Data Parallel or Distributed Data Parallel')</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">40</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; 'using Data Parallel or Distributed Data Parallel')</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">37</td>
<td class="TextItemSame">parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">41</td>
<td class="TextItemSame">parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">38</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; metavar='LR', help='initial learning rate', dest='lr')</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">42</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; metavar='LR', help='initial learning rate', dest='lr')</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">39</td>
<td class="TextItemSame">parser.add_argument('--momentum', default=0.9, type=float, metavar='M',</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">43</td>
<td class="TextItemSame">parser.add_argument('--momentum', default=0.9, type=float, metavar='M',</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">40</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; help='momentum')</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">44</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; help='momentum')</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">41</td>
<td class="TextItemSame">parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">45</td>
<td class="TextItemSame">parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">42</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; metavar='W', help='weight decay (default: 1e-4)',</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">46</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; metavar='W', help='weight decay (default: 1e-4)',</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">43</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; dest='weight_decay')</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">47</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; dest='weight_decay')</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">44</td>
<td class="TextItemSame">parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">48</td>
<td class="TextItemSame">parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">45</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">49</td>
<td class="TextItemSame">&nbsp;</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">46</td>
<td class="TextItemSame">best_acc1 = 0</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">50</td>
<td class="TextItemSame">best_acc1 = 0</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">47</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">51</td>
<td class="TextItemSame">&nbsp;</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">48</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">52</td>
<td class="TextItemSame">&nbsp;</td>
</tr>
<tr class="SectionEnd">
<td class="TextItemNum AlignRight">49</td>
<td class="TextItemSame">def main():</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">53</td>
<td class="TextItemSame">def main():</td>
</tr>
<tr class="SectionAll">
<td class="TextItemNum AlignRight">50</td>
<td class="TextItemSigDiffMod">&nbsp; &nbsp; args = parser.parse_args()</td>
<td class="AlignCenter">&lt;&gt;</td>
<td class="TextItemNum AlignRight">54</td>
<td class="TextItemSigDiffMod">&nbsp; &nbsp; args<span class="TextSegSigDiff">,</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">_</span> = parser.parse<span class="TextSegSigDiff">_known</span>_args()</td>
</tr>
<tr class="SectionAll">
<td class="TextItemNum AlignRight">51</td>
<td class="TextItemSame">&nbsp; &nbsp; global best_acc1</td>
<td class="AlignCenter">=</td>
<td class="TextItemNum AlignRight">55</td>
<td class="TextItemSame">&nbsp; &nbsp; global best_acc1</td>
</tr>
<tr class="SectionBegin">
<td class="TextItemNum AlignRight">&nbsp;</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">-+</td>
<td class="TextItemNum AlignRight">56</td>
<td class="TextItemSigRightMod"><span class="TextSegInsigDiff">&nbsp; &nbsp; </span><span class="TextSegSigDiff">print</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">('------</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">'</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">+</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">str(hvd.size())</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">+</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">'--------------</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">'+</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">str(hvd.local_rank()))</span></td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">&nbsp;</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">57</td>
<td class="TextItemSigRightMod"><span class="TextSegInsigDiff">&nbsp; &nbsp; </span><span class="TextSegSigDiff">start</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">=</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">time.time()</span></td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">&nbsp;</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">58</td>
<td class="TextItemSigRightMod"><span class="TextSegInsigDiff">&nbsp; &nbsp; </span><span class="TextSegSigDiff">print('---------Copy</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">DATA</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">to</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">cache---------')</span></td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">&nbsp;</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">59</td>
<td class="TextItemSigRightMod"><span class="TextSegInsigDiff">&nbsp; &nbsp; </span><span class="TextSegSigDiff">if</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">hvd.local_rank()</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">==</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">0:</span></td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">&nbsp;</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">60</td>
<td class="TextItemSigRightMod"><span class="TextSegInsigDiff">&nbsp; &nbsp; &nbsp; &nbsp; </span><span class="TextSegSigDiff">import</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">mox_patch_0603_v4</span></td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">&nbsp;</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">61</td>
<td class="TextItemSigRightMod"><span class="TextSegInsigDiff">&nbsp; &nbsp; &nbsp; &nbsp; </span><span class="TextSegSigDiff">import</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">moxing</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">as</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">mox</span></td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">&nbsp;</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">62</td>
<td class="TextItemSigRightMod"><span class="TextSegInsigDiff">&nbsp; &nbsp; &nbsp; &nbsp; </span><span class="TextSegSigDiff">mox.file.set_auth(negotiation=False)</span></td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">&nbsp;</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">63</td>
<td class="TextItemSigRightMod"><span class="TextSegInsigDiff">&nbsp; &nbsp; &nbsp; &nbsp; </span><span class="TextSegSigDiff">mox.file.make_dirs('/cache/data')</span></td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">&nbsp;</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">64</td>
<td class="TextItemSigRightMod"><span class="TextSegInsigDiff">&nbsp; &nbsp; &nbsp; &nbsp; </span><span class="TextSegSigDiff">mox.file.make_dirs('/cache/results')</span></td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">&nbsp;</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">65</td>
<td class="TextItemSigRightMod"><span class="TextSegInsigDiff">&nbsp; &nbsp; &nbsp; &nbsp; </span><span class="TextSegSigDiff">mox.file.copy_parallel(args.data,</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">'/cache/fruit/')</span></td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">&nbsp;</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">66</td>
<td class="TextItemSigRightMod"><span class="TextSegInsigDiff">&nbsp; &nbsp; </span><span class="TextSegSigDiff">print('---------Copy</span><span class="TextSegInsigDiff">&nbsp; </span><span class="TextSegSigDiff">Finished---------</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">'</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">+</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">str(time.time()</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">-</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">start)</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">+</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">'</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">s')</span></td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">&nbsp;</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">67</td>
<td class="TextItemSigRightMod"><span class="TextSegInsigDiff">&nbsp; &nbsp; </span><span class="TextSegSigDiff">print</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">('------------------sync-------------------------')</span></td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">&nbsp;</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">68</td>
<td class="TextItemSigRightMod"><span class="TextSegInsigDiff">&nbsp; &nbsp; </span><span class="TextSegSigDiff">args.data</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">=</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">'/cache/fruit'</span></td>
</tr>
<tr class="SectionEnd">
<td class="TextItemNum AlignRight">&nbsp;</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">69</td>
<td class="TextItemSigRightMod"><span class="TextSegInsigDiff">&nbsp; &nbsp; </span><span class="TextSegSigDiff">sync_load_data</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">=</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">hvd.allreduce(torch.tensor(0),</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">name='sync_load_data')</span></td>
</tr>
<tr class="SectionAll">
<td class="TextItemNum AlignRight">52</td>
<td class="TextItemSame">&nbsp; &nbsp; model = models.__dict__[args.arch]()</td>
<td class="AlignCenter">=</td>
<td class="TextItemNum AlignRight">70</td>
<td class="TextItemSame">&nbsp; &nbsp; model = models.__dict__[args.arch]()</td>
</tr>
<tr class="SectionAll">
<td class="TextItemNum AlignRight">&nbsp;</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">-+</td>
<td class="TextItemNum AlignRight">71</td>
<td class="TextItemSigRightMod"><span class="TextSegInsigDiff">&nbsp; &nbsp; </span><span class="TextSegSigDiff">args.gpu</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">=</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">hvd.local_rank()</span></td>
</tr>
<tr class="SectionBegin">
<td class="TextItemNum AlignRight">53</td>
<td class="TextItemSame">&nbsp; &nbsp; if args.gpu is not None:</td>
<td class="AlignCenter">=</td>
<td class="TextItemNum AlignRight">72</td>
<td class="TextItemSame">&nbsp; &nbsp; if args.gpu is not None:</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">54</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; print(&quot;Use GPU: {} for training&quot;.format(args.gpu))</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">73</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; print(&quot;Use GPU: {} for training&quot;.format(args.gpu))</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">55</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; torch.cuda.set_device(args.gpu)</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">74</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; torch.cuda.set_device(args.gpu)</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">56</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; model = model.cuda(args.gpu)</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">75</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; model = model.cuda(args.gpu)</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">57</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">76</td>
<td class="TextItemSame">&nbsp;</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">58</td>
<td class="TextItemSame">&nbsp; &nbsp; # define loss function (criterion) and optimizer</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">77</td>
<td class="TextItemSame">&nbsp; &nbsp; # define loss function (criterion) and optimizer</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">59</td>
<td class="TextItemSame">&nbsp; &nbsp; criterion = nn.CrossEntropyLoss().cuda(args.gpu)</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">78</td>
<td class="TextItemSame">&nbsp; &nbsp; criterion = nn.CrossEntropyLoss().cuda(args.gpu)</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">60</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">79</td>
<td class="TextItemSame">&nbsp;</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">61</td>
<td class="TextItemSame">&nbsp; &nbsp; optimizer = torch.optim.SGD(model.parameters(), args.lr,</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">80</td>
<td class="TextItemSame">&nbsp; &nbsp; optimizer = torch.optim.SGD(model.parameters(), args.lr,</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">62</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; momentum=args.momentum,</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">81</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; momentum=args.momentum,</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">63</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; weight_decay=args.weight_decay)</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">82</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; weight_decay=args.weight_decay)</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">64</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">83</td>
<td class="TextItemSame">&nbsp;</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">65</td>
<td class="TextItemSame">&nbsp; &nbsp; cudnn.benchmark = True</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">84</td>
<td class="TextItemSame">&nbsp; &nbsp; cudnn.benchmark = True</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">66</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">85</td>
<td class="TextItemSame">&nbsp;</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">67</td>
<td class="TextItemSame">&nbsp; &nbsp; # Data loading code</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">86</td>
<td class="TextItemSame">&nbsp; &nbsp; # Data loading code</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">68</td>
<td class="TextItemSame">&nbsp; &nbsp; traindir = os.path.join(args.data, 'train')</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">87</td>
<td class="TextItemSame">&nbsp; &nbsp; traindir = os.path.join(args.data, 'train')</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">69</td>
<td class="TextItemSame">&nbsp; &nbsp; valdir = os.path.join(args.data, 'eval')</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">88</td>
<td class="TextItemSame">&nbsp; &nbsp; valdir = os.path.join(args.data, 'eval')</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">70</td>
<td class="TextItemSame">&nbsp; &nbsp; normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">89</td>
<td class="TextItemSame">&nbsp; &nbsp; normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">71</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; std=[0.229, 0.224, 0.225])</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">90</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; std=[0.229, 0.224, 0.225])</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">72</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">91</td>
<td class="TextItemSame">&nbsp;</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">73</td>
<td class="TextItemSame">&nbsp; &nbsp; train_dataset = datasets.ImageFolder(</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">92</td>
<td class="TextItemSame">&nbsp; &nbsp; train_dataset = datasets.ImageFolder(</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">74</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; traindir,</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">93</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; traindir,</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">75</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; transforms.Compose([</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">94</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; transforms.Compose([</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">76</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; transforms.RandomResizedCrop(224),</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">95</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; transforms.RandomResizedCrop(224),</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">77</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; transforms.RandomHorizontalFlip(),</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">96</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; transforms.RandomHorizontalFlip(),</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">78</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; transforms.ToTensor(),</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">97</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; transforms.ToTensor(),</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">79</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; normalize,</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">98</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; normalize,</td>
</tr>
<tr class="SectionEnd">
<td class="TextItemNum AlignRight">80</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; ]))</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">99</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; ]))</td>
</tr>
<tr class="SectionBegin">
<td class="TextItemNum AlignRight">81</td>
<td class="TextItemSigDiffMod">&nbsp;</td>
<td class="AlignCenter">&lt;&gt;</td>
<td class="TextItemNum AlignRight">100</td>
<td class="TextItemSigDiffMod"><span class="TextSegInsigDiff">&nbsp; &nbsp; </span><span class="TextSegSigDiff">train_sampler</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">=</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">torch.utils.data.distributed.DistributedSampler(</span></td>
</tr>
<tr class="SectionEnd">
<td class="TextItemNum AlignRight">&nbsp;</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">101</td>
<td class="TextItemSigRightMod"><span class="TextSegInsigDiff">&nbsp; &nbsp; &nbsp; &nbsp; </span><span class="TextSegSigDiff">train_dataset,</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">num_replicas=hvd.size(),</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">rank=hvd.rank())</span></td>
</tr>
<tr class="SectionAll">
<td class="TextItemNum AlignRight">82</td>
<td class="TextItemSame">&nbsp; &nbsp; train_loader = torch.utils.data.DataLoader(</td>
<td class="AlignCenter">=</td>
<td class="TextItemNum AlignRight">102</td>
<td class="TextItemSame">&nbsp; &nbsp; train_loader = torch.utils.data.DataLoader(</td>
</tr>
<tr class="SectionBegin">
<td class="TextItemNum AlignRight">83</td>
<td class="TextItemSigDiffMod">&nbsp; &nbsp; &nbsp; &nbsp; train_dataset, batch_size=args.batch_size, shuffle=<span class="TextSegSigDiff">Tru</span>e,</td>
<td class="AlignCenter">&lt;&gt;</td>
<td class="TextItemNum AlignRight">103</td>
<td class="TextItemSigDiffMod">&nbsp; &nbsp; &nbsp; &nbsp; train_dataset, batch_size=args.batch_size, shuffle=<span class="TextSegSigDiff">Fals</span>e,</td>
</tr>
<tr class="SectionEnd">
<td class="TextItemNum AlignRight">84</td>
<td class="TextItemSigDiffMod">&nbsp; &nbsp; &nbsp; &nbsp; num_workers=args.workers, pin_memory=True, sampler=<span class="TextSegSigDiff">Non</span><span class="TextSegSigDiff">e</span>)</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">104</td>
<td class="TextItemSigDiffMod">&nbsp; &nbsp; &nbsp; &nbsp; num_workers=args.workers, pin_memory=True, sampler=<span class="TextSegSigDiff">train_sampler</span>)</td>
</tr>
<tr class="SectionBegin">
<td class="TextItemNum AlignRight">85</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">=</td>
<td class="TextItemNum AlignRight">105</td>
<td class="TextItemSame">&nbsp;</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">86</td>
<td class="TextItemSame">&nbsp; &nbsp; val_dataset = datasets.ImageFolder(</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">106</td>
<td class="TextItemSame">&nbsp; &nbsp; val_dataset = datasets.ImageFolder(</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">87</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; valdir,</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">107</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; valdir,</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">88</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; transforms.Compose([</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">108</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; transforms.Compose([</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">89</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; transforms.Resize(256),</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">109</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; transforms.Resize(256),</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">90</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; transforms.CenterCrop(224),</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">110</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; transforms.CenterCrop(224),</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">91</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; transforms.ToTensor(),</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">111</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; transforms.ToTensor(),</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">92</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; normalize,</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">112</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; normalize,</td>
</tr>
<tr class="SectionEnd">
<td class="TextItemNum AlignRight">93</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; ]))</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">113</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; ]))</td>
</tr>
<tr class="SectionBegin">
<td class="TextItemNum AlignRight">&nbsp;</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">-+</td>
<td class="TextItemNum AlignRight">114</td>
<td class="TextItemSigRightMod"><span class="TextSegInsigDiff">&nbsp; &nbsp; </span><span class="TextSegSigDiff">val_sampler</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">=</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">torch.utils.data.distributed.DistributedSampler(</span></td>
</tr>
<tr class="SectionEnd">
<td class="TextItemNum AlignRight">&nbsp;</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">115</td>
<td class="TextItemSigRightMod"><span class="TextSegInsigDiff">&nbsp; &nbsp; &nbsp; &nbsp; </span><span class="TextSegSigDiff">val_dataset,</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">num_replicas=hvd.size(),</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">rank=hvd.rank())</span></td>
</tr>
<tr class="SectionBegin">
<td class="TextItemNum AlignRight">94</td>
<td class="TextItemSame">&nbsp; &nbsp; val_loader = torch.utils.data.DataLoader(</td>
<td class="AlignCenter">=</td>
<td class="TextItemNum AlignRight">116</td>
<td class="TextItemSame">&nbsp; &nbsp; val_loader = torch.utils.data.DataLoader(</td>
</tr>
<tr class="SectionEnd">
<td class="TextItemNum AlignRight">95</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; val_dataset, batch_size=args.batch_size, shuffle=False,</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">117</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; val_dataset, batch_size=args.batch_size, shuffle=False,</td>
</tr>
<tr class="SectionBegin">
<td class="TextItemNum AlignRight">96</td>
<td class="TextItemSigDiffMod">&nbsp; &nbsp; &nbsp; &nbsp; num_workers=args.workers, pin_memory=True, sampler=<span class="TextSegSigDiff">Non</span><span class="TextSegSigDiff">e</span>)</td>
<td class="AlignCenter">&lt;&gt;</td>
<td class="TextItemNum AlignRight">118</td>
<td class="TextItemSigDiffMod">&nbsp; &nbsp; &nbsp; &nbsp; num_workers=args.workers, pin_memory=True, sampler=<span class="TextSegSigDiff">val_sampler</span>)</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">&nbsp;</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">119</td>
<td class="TextItemInsigRightMod">&nbsp;</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">&nbsp;</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">120</td>
<td class="TextItemSigRightMod"><span class="TextSegInsigDiff">&nbsp; &nbsp; </span><span class="TextSegSigDiff">optimizer</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">=</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">hvd.DistributedOptimizer(</span></td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">&nbsp;</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">121</td>
<td class="TextItemSigRightMod"><span class="TextSegInsigDiff">&nbsp; &nbsp; &nbsp; &nbsp; </span><span class="TextSegSigDiff">optimizer,</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">named_parameters=model.named_parameters(),</span></td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">&nbsp;</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">122</td>
<td class="TextItemSigRightMod"><span class="TextSegInsigDiff">&nbsp; &nbsp; &nbsp; &nbsp; </span><span class="TextSegSigDiff">compression=hvd.Compression.none,</span><span class="TextSegInsigDiff">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; </span><span class="TextSegSigDiff">#</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">hvd.Compression.fp16</span></td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">&nbsp;</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">123</td>
<td class="TextItemSigRightMod"><span class="TextSegInsigDiff">&nbsp; &nbsp; &nbsp; &nbsp; </span><span class="TextSegSigDiff">backward_passes_per_step=1)</span></td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">&nbsp;</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">124</td>
<td class="TextItemInsigRightMod">&nbsp;</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">&nbsp;</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">125</td>
<td class="TextItemSigRightMod"><span class="TextSegInsigDiff">&nbsp; &nbsp; </span><span class="TextSegSigDiff">hvd.broadcast_parameters(model.state_dict(),</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">root_rank=0)</span></td>
</tr>
<tr class="SectionEnd">
<td class="TextItemNum AlignRight">&nbsp;</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">126</td>
<td class="TextItemSigRightMod"><span class="TextSegInsigDiff">&nbsp; &nbsp; </span><span class="TextSegSigDiff">hvd.broadcast_optimizer_state(optimizer,</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">root_rank=0)</span></td>
</tr>
<tr class="SectionBegin">
<td class="TextItemNum AlignRight">97</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">=</td>
<td class="TextItemNum AlignRight">127</td>
<td class="TextItemSame">&nbsp;</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">98</td>
<td class="TextItemSame">&nbsp; &nbsp; for epoch in range(0, args.epochs):</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">128</td>
<td class="TextItemSame">&nbsp; &nbsp; for epoch in range(0, args.epochs):</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">99</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; adjust_learning_rate(optimizer, epoch, args)</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">129</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; adjust_learning_rate(optimizer, epoch, args)</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">100</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">130</td>
<td class="TextItemSame">&nbsp;</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">101</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; # train for one epoch</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">131</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; # train for one epoch</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">102</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; train(train_loader, model, criterion, optimizer, epoch, args)</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">132</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; train(train_loader, model, criterion, optimizer, epoch, args)</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">103</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">133</td>
<td class="TextItemSame">&nbsp;</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">104</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; # evaluate on validation set</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">134</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; # evaluate on validation set</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">105</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; acc1 = validate(val_loader, model, criterion, args)</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">135</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; acc1 = validate(val_loader, model, criterion, args)</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">106</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">136</td>
<td class="TextItemSame">&nbsp;</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">107</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; # remember best acc@1 and save checkpoint</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">137</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; # remember best acc@1 and save checkpoint</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">108</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; is_best = acc1 &gt; best_acc1</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">138</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; is_best = acc1 &gt; best_acc1</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">109</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; best_acc1 = max(acc1, best_acc1)</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">139</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; best_acc1 = max(acc1, best_acc1)</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">110</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">140</td>
<td class="TextItemSame">&nbsp;</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">111</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; save_checkpoint({</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">141</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; save_checkpoint({</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">112</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 'epoch': epoch + 1,</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">142</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 'epoch': epoch + 1,</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">113</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 'arch': args.arch,</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">143</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 'arch': args.arch,</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">114</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 'state_dict': model.state_dict(),</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">144</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 'state_dict': model.state_dict(),</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">115</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 'best_acc1': best_acc1,</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">145</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 'best_acc1': best_acc1,</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">116</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 'optimizer' : optimizer.state_dict(),</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">146</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; 'optimizer' : optimizer.state_dict(),</td>
</tr>
<tr class="SectionEnd">
<td class="TextItemNum AlignRight">117</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; }, is_best)</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">147</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; }, is_best)</td>
</tr>
<tr class="SectionBegin">
<td class="TextItemNum AlignRight">&nbsp;</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">-+</td>
<td class="TextItemNum AlignRight">148</td>
<td class="TextItemSigRightMod"><span class="TextSegInsigDiff">&nbsp; &nbsp; </span><span class="TextSegSigDiff">if</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">hvd.local_rank()</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">==</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">0:</span></td>
</tr>
<tr class="SectionEnd">
<td class="TextItemNum AlignRight">&nbsp;</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">149</td>
<td class="TextItemSigRightMod"><span class="TextSegInsigDiff">&nbsp; &nbsp; &nbsp; &nbsp; </span><span class="TextSegSigDiff">mox.file.copy_parallel('/cache/results',</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">'s3://modelarts-no1/ckpt/pytorch_example_results/')</span></td>
</tr>
<tr class="SectionBegin">
<td class="TextItemNum AlignRight">118</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">=</td>
<td class="TextItemNum AlignRight">150</td>
<td class="TextItemSame">&nbsp;</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">119</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">151</td>
<td class="TextItemSame">&nbsp;</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">120</td>
<td class="TextItemSame">def train(train_loader, model, criterion, optimizer, epoch, args):</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">152</td>
<td class="TextItemSame">def train(train_loader, model, criterion, optimizer, epoch, args):</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">121</td>
<td class="TextItemSame">&nbsp; &nbsp; batch_time = AverageMeter('Time', ':6.3f')</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">153</td>
<td class="TextItemSame">&nbsp; &nbsp; batch_time = AverageMeter('Time', ':6.3f')</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">122</td>
<td class="TextItemSame">&nbsp; &nbsp; data_time = AverageMeter('Data', ':6.3f')</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">154</td>
<td class="TextItemSame">&nbsp; &nbsp; data_time = AverageMeter('Data', ':6.3f')</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">123</td>
<td class="TextItemSame">&nbsp; &nbsp; losses = AverageMeter('Loss', ':.4e')</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">155</td>
<td class="TextItemSame">&nbsp; &nbsp; losses = AverageMeter('Loss', ':.4e')</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">124</td>
<td class="TextItemSame">&nbsp; &nbsp; top1 = AverageMeter('Acc@1', ':6.2f')</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">156</td>
<td class="TextItemSame">&nbsp; &nbsp; top1 = AverageMeter('Acc@1', ':6.2f')</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">125</td>
<td class="TextItemSame">&nbsp; &nbsp; top5 = AverageMeter('Acc@5', ':6.2f')</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">157</td>
<td class="TextItemSame">&nbsp; &nbsp; top5 = AverageMeter('Acc@5', ':6.2f')</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">126</td>
<td class="TextItemSame">&nbsp; &nbsp; progress = ProgressMeter(</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">158</td>
<td class="TextItemSame">&nbsp; &nbsp; progress = ProgressMeter(</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">127</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; len(train_loader),</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">159</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; len(train_loader),</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">128</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; [batch_time, data_time, losses, top1, top5],</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">160</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; [batch_time, data_time, losses, top1, top5],</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">129</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; prefix=&quot;Epoch: [{}]&quot;.format(epoch))</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">161</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; prefix=&quot;Epoch: [{}]&quot;.format(epoch))</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">130</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">162</td>
<td class="TextItemSame">&nbsp;</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">131</td>
<td class="TextItemSame">&nbsp; &nbsp; # switch to train mode</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">163</td>
<td class="TextItemSame">&nbsp; &nbsp; # switch to train mode</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">132</td>
<td class="TextItemSame">&nbsp; &nbsp; model.train()</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">164</td>
<td class="TextItemSame">&nbsp; &nbsp; model.train()</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">133</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">165</td>
<td class="TextItemSame">&nbsp;</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">134</td>
<td class="TextItemSame">&nbsp; &nbsp; end = time.time()</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">166</td>
<td class="TextItemSame">&nbsp; &nbsp; end = time.time()</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">135</td>
<td class="TextItemSame">&nbsp; &nbsp; for i, (images, target) in enumerate(train_loader):</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">167</td>
<td class="TextItemSame">&nbsp; &nbsp; for i, (images, target) in enumerate(train_loader):</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">136</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; # measure data loading time</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">168</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; # measure data loading time</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">137</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; data_time.update(time.time() - end)</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">169</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; data_time.update(time.time() - end)</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">138</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">170</td>
<td class="TextItemSame">&nbsp;</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">139</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; if args.gpu is not None:</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">171</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; if args.gpu is not None:</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">140</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; images = images.cuda(args.gpu, non_blocking=True)</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">172</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; images = images.cuda(args.gpu, non_blocking=True)</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">141</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; target = target.cuda(args.gpu, non_blocking=True)</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">173</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; target = target.cuda(args.gpu, non_blocking=True)</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">142</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">174</td>
<td class="TextItemSame">&nbsp;</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">143</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; # compute output</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">175</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; # compute output</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">144</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; output = model(images)</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">176</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; output = model(images)</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">145</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; loss = criterion(output, target)</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">177</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; loss = criterion(output, target)</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">146</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">178</td>
<td class="TextItemSame">&nbsp;</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">147</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; # measure accuracy and record loss</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">179</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; # measure accuracy and record loss</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">148</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; acc1, acc5 = accuracy(output, target, topk=(1, 5))</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">180</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; acc1, acc5 = accuracy(output, target, topk=(1, 5))</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">149</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; losses.update(loss.item(), images.size(0))</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">181</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; losses.update(loss.item(), images.size(0))</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">150</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; top1.update(acc1[0], images.size(0))</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">182</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; top1.update(acc1[0], images.size(0))</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">151</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; top5.update(acc5[0], images.size(0))</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">183</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; top5.update(acc5[0], images.size(0))</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">152</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">184</td>
<td class="TextItemSame">&nbsp;</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">153</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; # compute gradient and do SGD step</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">185</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; # compute gradient and do SGD step</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">154</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; optimizer.zero_grad()</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">186</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; optimizer.zero_grad()</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">155</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; loss.backward()</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">187</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; loss.backward()</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">156</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; optimizer.step()</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">188</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; optimizer.step()</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">157</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">189</td>
<td class="TextItemSame">&nbsp;</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">158</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; # measure elapsed time</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">190</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; # measure elapsed time</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">159</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; batch_time.update(time.time() - end)</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">191</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; batch_time.update(time.time() - end)</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">160</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; end = time.time()</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">192</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; end = time.time()</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">161</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">193</td>
<td class="TextItemSame">&nbsp;</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">162</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; if i % args.print_freq == 0:</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">194</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; if i % args.print_freq == 0:</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">163</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; progress.display(i)</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">195</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; progress.display(i)</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">164</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">196</td>
<td class="TextItemSame">&nbsp;</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">165</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">197</td>
<td class="TextItemSame">&nbsp;</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">166</td>
<td class="TextItemSame">def validate(val_loader, model, criterion, args):</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">198</td>
<td class="TextItemSame">def validate(val_loader, model, criterion, args):</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">167</td>
<td class="TextItemSame">&nbsp; &nbsp; batch_time = AverageMeter('Time', ':6.3f')</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">199</td>
<td class="TextItemSame">&nbsp; &nbsp; batch_time = AverageMeter('Time', ':6.3f')</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">168</td>
<td class="TextItemSame">&nbsp; &nbsp; losses = AverageMeter('Loss', ':.4e')</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">200</td>
<td class="TextItemSame">&nbsp; &nbsp; losses = AverageMeter('Loss', ':.4e')</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">169</td>
<td class="TextItemSame">&nbsp; &nbsp; top1 = AverageMeter('Acc@1', ':6.2f')</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">201</td>
<td class="TextItemSame">&nbsp; &nbsp; top1 = AverageMeter('Acc@1', ':6.2f')</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">170</td>
<td class="TextItemSame">&nbsp; &nbsp; top5 = AverageMeter('Acc@5', ':6.2f')</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">202</td>
<td class="TextItemSame">&nbsp; &nbsp; top5 = AverageMeter('Acc@5', ':6.2f')</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">171</td>
<td class="TextItemSame">&nbsp; &nbsp; progress = ProgressMeter(</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">203</td>
<td class="TextItemSame">&nbsp; &nbsp; progress = ProgressMeter(</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">172</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; len(val_loader),</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">204</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; len(val_loader),</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">173</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; [batch_time, losses, top1, top5],</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">205</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; [batch_time, losses, top1, top5],</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">174</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; prefix='Test: ')</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">206</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; prefix='Test: ')</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">175</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">207</td>
<td class="TextItemSame">&nbsp;</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">176</td>
<td class="TextItemSame">&nbsp; &nbsp; # switch to evaluate mode</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">208</td>
<td class="TextItemSame">&nbsp; &nbsp; # switch to evaluate mode</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">177</td>
<td class="TextItemSame">&nbsp; &nbsp; model.eval()</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">209</td>
<td class="TextItemSame">&nbsp; &nbsp; model.eval()</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">178</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">210</td>
<td class="TextItemSame">&nbsp;</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">179</td>
<td class="TextItemSame">&nbsp; &nbsp; with torch.no_grad():</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">211</td>
<td class="TextItemSame">&nbsp; &nbsp; with torch.no_grad():</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">180</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; end = time.time()</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">212</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; end = time.time()</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">181</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; for i, (images, target) in enumerate(val_loader):</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">213</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; for i, (images, target) in enumerate(val_loader):</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">182</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; if args.gpu is not None:</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">214</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; if args.gpu is not None:</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">183</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; images = images.cuda(args.gpu, non_blocking=True)</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">215</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; images = images.cuda(args.gpu, non_blocking=True)</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">184</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; target = target.cuda(args.gpu, non_blocking=True)</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">216</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; target = target.cuda(args.gpu, non_blocking=True)</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">185</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">217</td>
<td class="TextItemSame">&nbsp;</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">186</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; # compute output</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">218</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; # compute output</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">187</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; output = model(images)</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">219</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; output = model(images)</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">188</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; loss = criterion(output, target)</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">220</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; loss = criterion(output, target)</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">189</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">221</td>
<td class="TextItemSame">&nbsp;</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">190</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; # measure accuracy and record loss</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">222</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; # measure accuracy and record loss</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">191</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; acc1, acc5 = accuracy(output, target, topk=(1, 5))</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">223</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; acc1, acc5 = accuracy(output, target, topk=(1, 5))</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">192</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; losses.update(loss.item(), images.size(0))</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">224</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; losses.update(loss.item(), images.size(0))</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">193</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; top1.update(acc1[0], images.size(0))</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">225</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; top1.update(acc1[0], images.size(0))</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">194</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; top5.update(acc5[0], images.size(0))</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">226</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; top5.update(acc5[0], images.size(0))</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">195</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">227</td>
<td class="TextItemSame">&nbsp;</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">196</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; # measure elapsed time</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">228</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; # measure elapsed time</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">197</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; batch_time.update(time.time() - end)</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">229</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; batch_time.update(time.time() - end)</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">198</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; end = time.time()</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">230</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; end = time.time()</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">199</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">231</td>
<td class="TextItemSame">&nbsp;</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">200</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; if i % args.print_freq == 0:</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">232</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; if i % args.print_freq == 0:</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">201</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; progress.display(i)</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">233</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; progress.display(i)</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">202</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">234</td>
<td class="TextItemSame">&nbsp;</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">203</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; # TODO: this should also be done with the ProgressMeter</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">235</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; # TODO: this should also be done with the ProgressMeter</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">204</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">236</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">205</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; .format(top1=top1, top5=top5))</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">237</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; .format(top1=top1, top5=top5))</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">206</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">238</td>
<td class="TextItemSame">&nbsp;</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">207</td>
<td class="TextItemSame">&nbsp; &nbsp; return top1.avg</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">239</td>
<td class="TextItemSame">&nbsp; &nbsp; return top1.avg</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">208</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">240</td>
<td class="TextItemSame">&nbsp;</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">209</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">241</td>
<td class="TextItemSame">&nbsp;</td>
</tr>
<tr class="SectionEnd">
<td class="TextItemNum AlignRight">210</td>
<td class="TextItemSame">def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">242</td>
<td class="TextItemSame">def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):</td>
</tr>
<tr class="SectionBegin">
<td class="TextItemNum AlignRight">&nbsp;</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&lt;&gt;</td>
<td class="TextItemNum AlignRight">243</td>
<td class="TextItemSigRightMod"><span class="TextSegInsigDiff">&nbsp; &nbsp; </span><span class="TextSegSigDiff">if</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">hvd.local_rank()</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">==</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">0:</span></td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">&nbsp;</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">244</td>
<td class="TextItemSigRightMod"><span class="TextSegInsigDiff">&nbsp; &nbsp; &nbsp; &nbsp; </span><span class="TextSegSigDiff">filename</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">=</span><span class="TextSegInsigDiff"> </span><span class="TextSegSigDiff">'/cache/results/checkpoint.pth.tar'</span></td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">211</td>
<td class="TextItemInsigDiffMod">&nbsp; &nbsp; torch.save(state, filename)</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">245</td>
<td class="TextItemInsigDiffMod">&nbsp; &nbsp; <span class="TextSegInsigDiff">&nbsp; &nbsp; </span>torch.save(state, filename)</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">212</td>
<td class="TextItemInsigDiffMod">&nbsp; &nbsp; if is_best:</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">246</td>
<td class="TextItemInsigDiffMod">&nbsp; &nbsp; <span class="TextSegInsigDiff">&nbsp; &nbsp; </span>if is_best:</td>
</tr>
<tr class="SectionEnd">
<td class="TextItemNum AlignRight">213</td>
<td class="TextItemSigDiffMod">&nbsp; &nbsp; &nbsp; &nbsp; shutil.copyfile(filename, 'model_best.pth.tar')</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">247</td>
<td class="TextItemSigDiffMod">&nbsp; &nbsp; &nbsp; &nbsp; <span class="TextSegInsigDiff">&nbsp; &nbsp; </span>shutil.copyfile(filename, '<span class="TextSegSigDiff">/cache/results/</span>model_best.pth.tar')</td>
</tr>
<tr class="SectionBegin">
<td class="TextItemNum AlignRight">214</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">=</td>
<td class="TextItemNum AlignRight">248</td>
<td class="TextItemSame">&nbsp;</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">215</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">249</td>
<td class="TextItemSame">&nbsp;</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">216</td>
<td class="TextItemSame">class AverageMeter(object):</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">250</td>
<td class="TextItemSame">class AverageMeter(object):</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">217</td>
<td class="TextItemSame">&nbsp; &nbsp; &quot;&quot;&quot;Computes and stores the average and current value&quot;&quot;&quot;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">251</td>
<td class="TextItemSame">&nbsp; &nbsp; &quot;&quot;&quot;Computes and stores the average and current value&quot;&quot;&quot;</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">218</td>
<td class="TextItemSame">&nbsp; &nbsp; def __init__(self, name, fmt=':f'):</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">252</td>
<td class="TextItemSame">&nbsp; &nbsp; def __init__(self, name, fmt=':f'):</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">219</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; self.name = name</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">253</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; self.name = name</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">220</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; self.fmt = fmt</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">254</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; self.fmt = fmt</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">221</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; self.reset()</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">255</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; self.reset()</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">222</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">256</td>
<td class="TextItemSame">&nbsp;</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">223</td>
<td class="TextItemSame">&nbsp; &nbsp; def reset(self):</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">257</td>
<td class="TextItemSame">&nbsp; &nbsp; def reset(self):</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">224</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; self.val = 0</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">258</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; self.val = 0</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">225</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; self.avg = 0</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">259</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; self.avg = 0</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">226</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; self.sum = 0</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">260</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; self.sum = 0</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">227</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; self.count = 0</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">261</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; self.count = 0</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">228</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">262</td>
<td class="TextItemSame">&nbsp;</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">229</td>
<td class="TextItemSame">&nbsp; &nbsp; def update(self, val, n=1):</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">263</td>
<td class="TextItemSame">&nbsp; &nbsp; def update(self, val, n=1):</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">230</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; self.val = val</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">264</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; self.val = val</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">231</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; self.sum += val * n</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">265</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; self.sum += val * n</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">232</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; self.count += n</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">266</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; self.count += n</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">233</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; self.avg = self.sum / self.count</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">267</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; self.avg = self.sum / self.count</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">234</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">268</td>
<td class="TextItemSame">&nbsp;</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">235</td>
<td class="TextItemSame">&nbsp; &nbsp; def __str__(self):</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">269</td>
<td class="TextItemSame">&nbsp; &nbsp; def __str__(self):</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">236</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">270</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">237</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; return fmtstr.format(**self.__dict__)</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">271</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; return fmtstr.format(**self.__dict__)</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">238</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">272</td>
<td class="TextItemSame">&nbsp;</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">239</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">273</td>
<td class="TextItemSame">&nbsp;</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">240</td>
<td class="TextItemSame">class ProgressMeter(object):</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">274</td>
<td class="TextItemSame">class ProgressMeter(object):</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">241</td>
<td class="TextItemSame">&nbsp; &nbsp; def __init__(self, num_batches, meters, prefix=&quot;&quot;):</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">275</td>
<td class="TextItemSame">&nbsp; &nbsp; def __init__(self, num_batches, meters, prefix=&quot;&quot;):</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">242</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; self.batch_fmtstr = self._get_batch_fmtstr(num_batches)</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">276</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; self.batch_fmtstr = self._get_batch_fmtstr(num_batches)</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">243</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; self.meters = meters</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">277</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; self.meters = meters</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">244</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; self.prefix = prefix</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">278</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; self.prefix = prefix</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">245</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">279</td>
<td class="TextItemSame">&nbsp;</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">246</td>
<td class="TextItemSame">&nbsp; &nbsp; def display(self, batch):</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">280</td>
<td class="TextItemSame">&nbsp; &nbsp; def display(self, batch):</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">247</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; entries = [self.prefix + self.batch_fmtstr.format(batch)]</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">281</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; entries = [self.prefix + self.batch_fmtstr.format(batch)]</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">248</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; entries += [str(meter) for meter in self.meters]</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">282</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; entries += [str(meter) for meter in self.meters]</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">249</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; print('\t'.join(entries))</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">283</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; print('\t'.join(entries))</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">250</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">284</td>
<td class="TextItemSame">&nbsp;</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">251</td>
<td class="TextItemSame">&nbsp; &nbsp; def _get_batch_fmtstr(self, num_batches):</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">285</td>
<td class="TextItemSame">&nbsp; &nbsp; def _get_batch_fmtstr(self, num_batches):</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">252</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; num_digits = len(str(num_batches // 1))</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">286</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; num_digits = len(str(num_batches // 1))</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">253</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; fmt = '{:' + str(num_digits) + 'd}'</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">287</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; fmt = '{:' + str(num_digits) + 'd}'</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">254</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; return '[' + fmt + '/' + fmt.format(num_batches) + ']'</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">288</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; return '[' + fmt + '/' + fmt.format(num_batches) + ']'</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">255</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">289</td>
<td class="TextItemSame">&nbsp;</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">256</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">290</td>
<td class="TextItemSame">&nbsp;</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">257</td>
<td class="TextItemSame">def adjust_learning_rate(optimizer, epoch, args):</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">291</td>
<td class="TextItemSame">def adjust_learning_rate(optimizer, epoch, args):</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">258</td>
<td class="TextItemSame">&nbsp; &nbsp; &quot;&quot;&quot;Sets the learning rate to the initial LR decayed by 10 every 30 epochs&quot;&quot;&quot;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">292</td>
<td class="TextItemSame">&nbsp; &nbsp; &quot;&quot;&quot;Sets the learning rate to the initial LR decayed by 10 every 30 epochs&quot;&quot;&quot;</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">259</td>
<td class="TextItemSame">&nbsp; &nbsp; lr = args.lr * (0.1 ** (epoch // 30))</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">293</td>
<td class="TextItemSame">&nbsp; &nbsp; lr = args.lr * (0.1 ** (epoch // 30))</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">260</td>
<td class="TextItemSame">&nbsp; &nbsp; for param_group in optimizer.param_groups:</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">294</td>
<td class="TextItemSame">&nbsp; &nbsp; for param_group in optimizer.param_groups:</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">261</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; param_group['lr'] = lr</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">295</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; param_group['lr'] = lr</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">262</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">296</td>
<td class="TextItemSame">&nbsp;</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">263</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">297</td>
<td class="TextItemSame">&nbsp;</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">264</td>
<td class="TextItemSame">def accuracy(output, target, topk=(1,)):</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">298</td>
<td class="TextItemSame">def accuracy(output, target, topk=(1,)):</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">265</td>
<td class="TextItemSame">&nbsp; &nbsp; &quot;&quot;&quot;Computes the accuracy over the k top predictions for the specified values of k&quot;&quot;&quot;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">299</td>
<td class="TextItemSame">&nbsp; &nbsp; &quot;&quot;&quot;Computes the accuracy over the k top predictions for the specified values of k&quot;&quot;&quot;</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">266</td>
<td class="TextItemSame">&nbsp; &nbsp; with torch.no_grad():</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">300</td>
<td class="TextItemSame">&nbsp; &nbsp; with torch.no_grad():</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">267</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; maxk = max(topk)</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">301</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; maxk = max(topk)</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">268</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; batch_size = target.size(0)</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">302</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; batch_size = target.size(0)</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">269</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">303</td>
<td class="TextItemSame">&nbsp;</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">270</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; _, pred = output.topk(maxk, 1, True, True)</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">304</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; _, pred = output.topk(maxk, 1, True, True)</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">271</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; pred = pred.t()</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">305</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; pred = pred.t()</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">272</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; correct = pred.eq(target.view(1, -1).expand_as(pred))</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">306</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; correct = pred.eq(target.view(1, -1).expand_as(pred))</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">273</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">307</td>
<td class="TextItemSame">&nbsp;</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">274</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; res = []</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">308</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; res = []</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">275</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; for k in topk:</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">309</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; for k in topk:</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">276</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">310</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">277</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; res.append(correct_k.mul_(100.0 / batch_size))</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">311</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; res.append(correct_k.mul_(100.0 / batch_size))</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">278</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; return res</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">312</td>
<td class="TextItemSame">&nbsp; &nbsp; &nbsp; &nbsp; return res</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">279</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">313</td>
<td class="TextItemSame">&nbsp;</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">280</td>
<td class="TextItemSame">&nbsp;</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">314</td>
<td class="TextItemSame">&nbsp;</td>
</tr>
<tr class="SectionMiddle">
<td class="TextItemNum AlignRight">281</td>
<td class="TextItemSame">if __name__ == '__main__':</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">315</td>
<td class="TextItemSame">if __name__ == '__main__':</td>
</tr>
<tr class="SectionEnd">
<td class="TextItemNum AlignRight">282</td>
<td class="TextItemSame">&nbsp; &nbsp; main()</td>
<td class="AlignCenter">&nbsp;</td>
<td class="TextItemNum AlignRight">316</td>
<td class="TextItemSame">&nbsp; &nbsp; main()</td>
</tr>
</table>



## 多进程实现分布式或多卡注意事项

1. 通过进程号控制数据读取，尽量使得每个进程读取不同的数据块
2. OBS文件读写操作尽量使用某一进程负责，不要出现多进程同时操作同一文件的情况
3. 每个进程都会打印各自的日志，可以根据进程编号自行控制
4. 验证集如果切分到多个进程并行执行，结果打印需要进程间同步或自行控制
5. 模型参数初始化后，需要从某一进程广播到其它进程，使得所有进程模型的初始化参数保持一致，（imagenet实际测试，不一致的初始化参数会导致最终训练精度降低2%）