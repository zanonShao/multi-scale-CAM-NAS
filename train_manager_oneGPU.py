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
from sklearn.metrics import precision_score, recall_score, average_precision_score
from tqdm import tqdm
import os
import torch.nn.functional as F
import utils
from CAM import CAM_example, CAM_example_2
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser("VOC_2012")
parser.add_argument('--workers', type=int, default=8, help='number of workers to load dataset')
# parser.add_argument('--data', type=str, default='/tmp/cache/', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=36, help='batch size')
parser.add_argument('--base_lr', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--epochs', type=int, default=60, help='num of training epochs')
# parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
# parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--result_path', type=str, default='./saved_models', help='path to save the model')
parser.add_argument('--exp_name', type=str, default='exp_temp', help='folder under result_path to save this run result')
# parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
# parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
# parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
# parser.add_argument('--seed', type=int, default=2, help='random seed')
# parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
# parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--base_arch_lr', type=float, default=6e-3, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--begin', type=int, default=6, help='batch size')
parser.add_argument('--val_frequency', type=int, default=5, help='val frequency')
parser.add_argument('--tensorboard_frequency', type=int, default=50,
                    help='every x times to save in tensorboardX during training')
parser.add_argument('--gpus', type=str, default='0')
parser.add_argument('--reuse', type=str, default=None,
                    help='if reuse is not None, load the checkpoint form --result path/reuse')
args = parser.parse_args()

savepath_for_this_exp = os.path.join(args.result_path, args.exp_name)
if not os.path.exists(savepath_for_this_exp):
    os.makedirs(savepath_for_this_exp)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(savepath_for_this_exp, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
writer = SummaryWriter(log_dir=savepath_for_this_exp)


def main():
    # Scale the learning rate by the number of workers.
    init_lr = args.base_lr
    init_arch_lr = args.base_arch_lr

    # Partition dataset among workers using DistributedSampler
    train_dataset = mydataset(dataroot='/data/shaozilong666/datasets/Pascal_VOC/VOC_2012/VOCdevkit/VOC2012/JPEGImages/',
                              lableroot='/NAS_REMOTE/shaozl/MS-CAM-NAS/', phase='train')
    val_dataset = mydataset(dataroot='/data/shaozilong666/datasets/Pascal_VOC/VOC_2012/VOCdevkit/VOC2012/JPEGImages/',
                            lableroot='/NAS_REMOTE/shaozl/MS-CAM-NAS/', phase='val')

    model = trapezoid_supernet(max_scale=4, num_layers=12).cuda()
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,pin_memory=True, num_workers=args.workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, pin_memory=True, num_workers=args.workers)
    val_loader_2 = DataLoader(val_dataset, batch_size=3600, pin_memory=True, num_workers=args.workers)

    criterion = nn.BCEWithLogitsLoss().cuda()
    optimizer = torch.optim.SGD(model.weight_parameters(),
                                init_lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    optimizer_a = torch.optim.Adam(model.arch_parameters(), lr=init_arch_lr, betas=(0.9, 0.999),
                                   weight_decay=args.arch_weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)
    best_mAP = 0
    start_epoch = 0
    if args.reuse != None:
        print('load_state_dict from %s' % os.path.join(args.result_path, args.reuse))
        checkpoint = torch.load(os.path.join(args.result_path, args.reuse))
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        optimizer_a.load_state_dict(checkpoint['optimizer_a'])
        start_epoch = checkpoint['epoch'] + 1
        best_mAP = checkpoint['best_acc_top1']

    for epoch in range(start_epoch, args.epochs):
        current_lr = scheduler.get_last_lr()[0]

        mAP, obj = train(train_loader, val_loader, model, optimizer, optimizer_a, criterion,
                         current_lr, epoch)

        is_best = False
        if epoch % args.val_frequency == 0:
            mAP, obj = infer(val_loader_2, model, criterion, epoch)

            if mAP > best_mAP:
                best_mAP = mAP
                is_best = True
            with torch.no_grad():
                htitch_raw, heatmap, results = CAM_example_2(val_dataset, model, 10, 4)

            logging.info('Val Epoch: %d best_mAP: %3f mAP: %3f loss: %3f', epoch, best_mAP, mAP, obj)
            writer.add_scalar('val/loss', obj, epoch)
            writer.add_scalar('val/mAP', mAP, epoch)
            writer.add_scalar('val/best_mAP', best_mAP, epoch)
            writer.add_image('heatmap', heatmap, epoch, dataformats='HWC')
            writer.add_image('results', results, epoch, dataformats='HWC')
            writer.add_image('htitch_raw', htitch_raw, epoch, dataformats='HWC')

        scheduler.step()

        utils.save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_acc_top1': best_mAP,
            'optimizer': optimizer.state_dict(),
            'optimizer_a': optimizer_a.state_dict(),
        }, is_best, savepath_for_this_exp)
        logging.info('Train Epoch: %d lr: %e loss: %3f', epoch, current_lr, obj)


def train(train_loader, val_loader, model, optimizer, optimizer_a, criterion, lr, epoch):
    objs = AvgrageMeter()
    mAPs = AvgrageMeter()
    # recalls = AvgrageMeter()
    # precisions = AvgrageMeter()
    model.train()
    val_loader = iter(val_loader)

    with tqdm(total=len(train_loader),
              desc='Train Epoch #{}'.format(epoch))as t:
        # disable=not run_manager.is_root, dynamic_ncols=True) as t:
        for step, (input, target) in enumerate(train_loader):
            n = input.size(0)

            input = input.cuda()
            target = target.cuda()

            # get a random minibatch from the search queue with replacement
            # try:
            #     input_search, target_search = next(valid_queue_iter)
            # except:
            # valid_queue_iter = iter(val_loader)
            if epoch < args.begin:
                if step == 0:
                    input_search, target_search = next(val_loader)
                    input_search = input_search.cuda(non_blocking=True)
                    target_search = target_search.cuda(non_blocking=True)

            elif epoch >= args.begin:
                input_search, target_search = next(val_loader)
                input_search = input_search.cuda(non_blocking=True)
                target_search = target_search.cuda(non_blocking=True)
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
            objs.update(loss.cpu().data.numpy(), n)
            # mAP = utils.compute_mAP(target,F.sigmoid(logits))
            # mAPs.update(mAP,n)

            if step % args.tensorboard_frequency == 0:
                writer.add_scalar('Train/loss', objs.avg, epoch * len(train_loader) + step)
                writer.add_scalar('Train/lr', lr, epoch * len(train_loader) + step)

            t.set_postfix({
                'loss': objs.avg,
                'lr': lr,
                'mAP': 'wait until Val',
            })
            t.update(1)
        # return mAPs.avg, objs.avg
        return None, objs.avg


def infer(val_loader, model, criterion, epoch):
    objs = AvgrageMeter()
    mAPs = AvgrageMeter()
    model.eval()
    with tqdm(total=len(val_loader),
              desc='Val Epoch #{}'.format(epoch))as t:
        for step, (input, target) in enumerate(val_loader):
            input = input.cuda()
            target = target.cuda()

            with torch.no_grad():
                logits = model(input)
            loss = criterion(logits, target)

            n = input.size(0)
            mAP = utils.compute_mAP(target, F.sigmoid(logits))
            objs.update(loss.cpu().data.numpy(), n)
            mAPs.update(mAP, n)

            t.set_postfix({
                'loss': objs.avg,
                'mAP': mAPs.avg,
            })
            t.update(1)

    return mAPs.avg, objs.avg


if __name__ == '__main__':
    main()
