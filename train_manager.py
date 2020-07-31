from ignite.engine import Events,create_supervised_trainer
from ignite.metrics import Accuracy,Precision,Recall
from trapezoid_supernet import trapezoid_supernet
from MyDataloader import mydataset
import torch
import torch.nn as nn
import argparse

parser = argparse.ArgumentParser("VOC_2012")
parser.add_argument('--workers', type=int, default=4, help='number of workers to load dataset')
# parser.add_argument('--data', type=str, default='/tmp/cache/', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=14, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.5, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.0, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
# parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--epochs', type=int, default=500, help='num of training epochs')
# parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
# parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
# parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
# parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
# parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='/tmp/checkpoints/', help='experiment name')
# parser.add_argument('--seed', type=int, default=2, help='random seed')
# parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
# parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=6e-3, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
# parser.add_argument('--begin', type=int, default=35, help='batch size')

args = parser.parse_args()

model = trapezoid_supernet().cuda()
train_loader = mydataset(dataroot='/NAS_REMOTE/shaozl/dataset/Pascal_VOC/VOC_2012/VOCdevkit/VOC2012/JPEGImages/',lableroot='/NAS_REMOTE/shaozl/MS-CAM-NAS/',phase='train')
val_loader = mydataset(dataroot='/NAS_REMOTE/shaozl/dataset/Pascal_VOC/VOC_2012/VOCdevkit/VOC2012/JPEGImages/',lableroot='/NAS_REMOTE/shaozl/MS-CAM-NAS/',phase='val')
criterion = nn.MultiLabelMarginLoss().cuda()
optimizer = torch.optim.SGD(model.parameters(),
                args.learning_rate,
                momentum=args.momentum,
                weight_decay=args.weight_decay)
architect_optimizer = torch.optim.Adam(model.arch_parameters(),
                                                    lr=args.arch_learning_rate, betas=(0.9, 0.999),
                                                    weight_decay=args.arch_weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, float(args.epochs), eta_min=args.learning_rate_min)

metrics={
Accuracy(is_multilabel=True),
Precision(is_multilabel=True),
Recall(is_multilabel=True)
}

trainer = create_supervised_trainer(model, optimizer, criterion)
trainer.run(train_loader, max_epochs=100)