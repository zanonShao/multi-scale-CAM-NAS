from ignite.engine import Events, create_supervised_trainer
from ignite.metrics import Accuracy, Precision, Recall
from trapezoid_supernet import trapezoid_supernet
from MyDataloader import mydataset
import torch
import torch.nn as nn
import logging
import argparse
from torch.utils.data import DataLoader
from utils import *
import sys
from sklearn.metrics import precision_score, recall_score
from tqdm import tqdm
import os
import torch.nn.functional as F

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
parser = argparse.ArgumentParser("VOC_2012")
parser.add_argument('--workers', type=int, default=4, help='number of workers to load dataset')
# parser.add_argument('--data', type=str, default='/tmp/cache/', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=14, help='batch size')
parser.add_argument('--base_lr', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--epochs', type=int, default=60, help='num of training epochs')
# parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
# parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
# parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
# parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
# parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='./', help='experiment name')
# parser.add_argument('--seed', type=int, default=2, help='random seed')
# parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
# parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--base_arch_lr', type=float, default=3e-3, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--begin', type=int, default=35, help='batch size')
args = parser.parse_args()

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


def main():
    # Scale the learning rate by the number of workers.
    init_lr = args.base_lr
    init_arch_lr = args.base_arch_lr

    # Partition dataset among workers using DistributedSampler
    train_dataset = mydataset(dataroot='/NAS_REMOTE/shaozl/dataset/Pascal_VOC/VOC_2012/VOCdevkit/VOC2012/JPEGImages/',
                              lableroot='/NAS_REMOTE/shaozl/MS-CAM-NAS/', phase='train')
    val_dataset = mydataset(dataroot='/NAS_REMOTE/shaozl/dataset/Pascal_VOC/VOC_2012/VOCdevkit/VOC2012/JPEGImages/',
                            lableroot='/NAS_REMOTE/shaozl/MS-CAM-NAS/', phase='val')

    model = trapezoid_supernet(max_scale=4, num_layers=12).cuda()
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    criterion = nn.MultiLabelSoftMarginLoss().cuda()
    optimizer = torch.optim.SGD(model.weight_parameters(),
                                init_lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    optimizer_a = torch.optim.Adam(model.arch_parameters(), lr=init_arch_lr, betas=(0.9, 0.999),
                                   weight_decay=args.arch_weight_decay)


    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)


    for epoch in range(args.epochs):
        current_lr = scheduler.get_lr()[0]
        logging.info('Epoch: %d lr: %e', epoch, current_lr)
        # genotype = model.genotype(
        logging.info('genotype = %s', 'wait to complete')
        arch_param = model.arch_parameters()

        precision, recall, train_obj = train(train_loader, val_loader, model, optimizer, optimizer_a, criterion,
                                             current_lr,
                                             epoch)
        scheduler.step()


def train(train_loader, val_loader, model, optimizer, optimizer_a, criterion, lr, epoch):
    objs = AvgrageMeter()
    recalls = AvgrageMeter()
    precisions = AvgrageMeter()

    with tqdm(total=len(train_loader),
              desc='Train Epoch #{}'.format(epoch))as t:
        # disable=not run_manager.is_root, dynamic_ncols=True) as t:
        for step, (input, target) in enumerate(train_loader):
            model.train()
            n = input.size(0)

            input = input.cuda()
            target = target.cuda()

            # get a random minibatch from the search queue with replacement
            try:
                input_search, target_search = next(valid_queue_iter)
            except:
                valid_queue_iter = iter(val_loader)
                input_search, target_search = next(valid_queue_iter)
            input_search = input_search.cuda(non_blocking=True)
            target_search = target_search.cuda(non_blocking=True)

            if epoch >= args.begin:
                optimizer_a.zero_grad()
                logits = model(input_search)
                loss_a = criterion(logits, target_search)
                loss_a.backward()
                optimizer_a.step()

            optimizer.zero_grad()
            logits = model(input)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()

            precision = precision_score(F.sigmoid(logits).cpu().data.numpy().round(),
                                        target.cpu().data.numpy().round(), average='micro')
            recall = recall_score(F.sigmoid(logits).cpu().data.numpy().round(),
                                  target.cpu().data.numpy().round(), average='micro')
            objs.update(loss.cpu().data.numpy(), n)
            precisions.update(precision, n)
            recalls.update(recall, n)

            # if step % args.report_freq == 0:
            #     logging.info('TRAIN Step: %03d Objs: %e precision: %f recall: %f', step, objs.avg, precisions.avg,
            #                  recalls.avg)
            t.set_postfix({
                'loss': objs.avg,
                'precision': precisions.avg,
                'recall': recalls.avg,
                'lr': lr,
            })
            t.update(1)
    return precisions.avg, recalls.avg, objs.avg


if __name__ == '__main__':
    main()
