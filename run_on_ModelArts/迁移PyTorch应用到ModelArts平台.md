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

下方源码为线下单机单卡分类程序，修改为线上多卡程序，需要修改的地方

```diff
import argparse
import os
+ os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'
import shutil
import time

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

+ import torch.distributed as dist
+ import torch.multiprocessing as mp

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', help='path to dataset')
parser.add_argument('--arch', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--print-freq', default=20, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch_size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
- parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
+ parser.add_argument('--gpu', default='0,1,2,3', type=str, help='GPU id to use.')

best_acc1 = 0


def main():
-   args = parser.parse_args()
+   args, _ = parser.parse_known_args()

+   start = time.time()
+   print('---------Copy DATA to cache---------')
+   import moxing as mox
+   mox.file.set_auth(negotiation=False)
+   mox.file.make_dirs('/cache/fruit/')
+   mox.file.make_dirs('/cache/results')
+   mox.file.copy_parallel(args.data, '/cache/fruit/')
+   args.data = '/cache/fruit/'
+   print('---------Copy  Finished--------- ' + str(time.time() - start) + ' s')
+   ngpus_per_node = len(args.gpu.split(','))
+   mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
+   save_dir  = '/cache/results'
+   s3_results_path = 's3://modelarts-no1/ckpt/pytorch_example_results/'
+   mox.file.copy_parallel(src_url=save_dir, dst_url=s3_results_path)

+ def main_worker(gpu, ngpus_per_node, args):
+   args.gpu = gpu
+   dist.init_process_group(backend='nccl', init_method='tcp://0.0.0.0:6666', world_size=ngpus_per_node, rank=args.gpu)
    global best_acc1
    model = models.__dict__[args.arch]()
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
+       args.workers = int(args.workers / ngpus_per_node)
+       args.lr = args.lr * ngpus_per_node
+       model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'eval')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

+   train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
-       train_dataset, batch_size=args.batch_size, shuffle=True,
-       num_workers=args.workers, pin_memory=True, sampler=None)
+       train_dataset, batch_size=args.batch_size, shuffle=False,
+       num_workers=args.workers, pin_memory=True, sampler=train_sampler) 

    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))
+   val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
-       num_workers=args.workers, pin_memory=True, sampler=None)
+       num_workers=args.workers, pin_memory=True, sampler=val_sampler)

    for epoch in range(0, args.epochs):
+       train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

+       if args.gpu == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best)


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
+   filename = '/cache/results/checkpoint.pth.tar'    
    torch.save(state, filename)
    if is_best:
-       shutil.copyfile(filename, 'model_best.pth.tar')
+       shutil.copyfile(filename, '/cache/results/model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()

```

## horovod实现单机多卡

下方源码为线下单机单卡分类程序，修改为horovod多卡程序，需要修改的地方

注：线上horovod分布式，只需修改下载数据的进程编号，保证每个计算节点有且只有一个进程下载数据即可

```diff
import argparse
import os
+ os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'
import shutil
import time

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

+ import horovod.torch as hvd
+ hvd.init()

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', help='path to dataset')
parser.add_argument('--arch', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--print-freq', default=20, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch_size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')

best_acc1 = 0


def main():
-   args = parser.parse_args()
+   args, _ = parser.parse_known_args()
+   print ('------ ' + str(hvd.size()) + '-------------- '+ str(hvd.local_rank()))
+   start = time.time()
+   print('---------Copy DATA to cache---------')
+   if hvd.local_rank() == 0:
+       import moxing as mox
+       mox.file.set_auth(negotiation=False)
+       mox.file.make_dirs('/cache/data')
+       mox.file.make_dirs('/cache/results')
+       mox.file.copy_parallel(args.data, '/cache/fruit/')
+   print('---------Copy  Finished--------- ' + str(time.time() - start) + ' s')
+   print ('------------------sync-------------------------')
+   args.data = '/cache/fruit'
+   sync_load_data = hvd.allreduce(torch.tensor(0), name='sync_load_data')
+   args.gpu = hvd.local_rank()
    global best_acc1
    model = models.__dict__[args.arch]()
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'eval')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
+   train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    train_loader = torch.utils.data.DataLoader(
-       train_dataset, batch_size=args.batch_size, shuffle=True,
-       num_workers=args.workers, pin_memory=True, sampler=None)
+		train_dataset, batch_size=args.batch_size, shuffle=False,
+       num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))
+   val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
-       num_workers=args.workers, pin_memory=True, sampler=None)
+       num_workers=args.workers, pin_memory=True, sampler=val_sampler)

+	optimizer = hvd.DistributedOptimizer(
+       optimizer, named_parameters=model.named_parameters(),
+       compression=hvd.Compression.none,       # hvd.Compression.fp16
+       backward_passes_per_step=1)

+   hvd.broadcast_parameters(model.state_dict(), root_rank=0)
+   hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    for epoch in range(0, args.epochs):
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer' : optimizer.state_dict(),
        }, is_best)
+   if hvd.local_rank() == 0:
+       mox.file.copy_parallel('/cache/results', 's3://modelarts-no1/ckpt/pytorch_example_results/')


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
-   torch.save(state, filename)
-   if is_best:
-       shutil.copyfile(filename, 'model_best.pth.tar')
+	if hvd.local_rank() == 0:
+       filename = '/cache/results/checkpoint.pth.tar'
+       torch.save(state, filename)
+       if is_best:
+           shutil.copyfile(filename, '/cache/results/model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()

```

## 多进程实现分布式或多卡注意事项

1. 通过进程号控制数据读取，尽量使得每个进程读取不同的数据块
2. OBS文件读写操作尽量使用某一进程负责，不要出现多进程同时操作同一文件的情况
3. 每个进程都会打印各自的日志，可以根据进程编号自行控制
4. 验证集如果切分到多个进程并行执行，结果打印需要进程间同步或自行控制
5. 模型参数初始化后，需要从某一进程广播到其它进程，使得所有进程模型的初始化参数保持一致，（imagenet实际测试，不一致的初始化参数会导致最终训练精度降低2%）