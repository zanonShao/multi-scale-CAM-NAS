import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES
from genotypes import Genotype
import pytorch_lightning as pl
from MyDataloader import mydataset
from torch.utils.data import DataLoader


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class MixedOp(nn.Module):

    def __init__(self, C, stride, pc_k):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        self.mp = nn.MaxPool2d(2, 2)
        self.pc_k = pc_k
        for primitive in PRIMITIVES:
            op = OPS[primitive](C // pc_k, stride, False)
            if 'pool' in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(C // pc_k, affine=False))
            self._ops.append(op)

    def forward(self, x, weights):

        dim_2 = x.shape[1]
        xtemp = x[:, :  dim_2 // self.pc_k, :, :]
        xtemp2 = x[:, dim_2 // self.pc_k:, :, :]

        # <editor-fold desc="Debug">
        # try:
        # temp1 = sum(w * op(xtemp) for w, op in zip(weights, self._ops))
        count = []
        for w, op in zip(weights, self._ops):
            count.append(w * op(xtemp))
        temp1 = sum(count)

        # except:
        #     print(w)
        #     print(op)
        #     exit(0)
        # </editor-fold>
        if temp1.shape[2] == x.shape[2]:
            # ans = torch.cat([temp1,self.bn(self.conv1(xtemp3))],dim=1)
            # ans = torch.cat([ans,xtemp4],dim=1)
            ans = torch.cat([temp1, xtemp2], dim=1)
            # ans = torch.cat([ans,x[:, 2*dim_2// 4: , :, :]],dim=1)
        else:
            # ans = torch.cat([temp1,self.bn(self.conv1(self.mp(xtemp3)))],dim=1)
            # ans = torch.cat([ans,self.mp(xtemp4)],dim=1)

            ans = torch.cat([temp1, self.mp(xtemp2)], dim=1)

        ans = channel_shuffle(ans, 2)
        return ans

        # return sum(w.to(x.device) * op(x) for w, op in zip(weights, self._ops))


class Cell(nn.Module):
    def __init__(self, C_pre, multipl=3, step=3, stride=1, pc_k=4, type='same'):
        super(Cell, self).__init__()
        self.C_pre = C_pre
        self.C = C_pre // step
        self.stride = stride
        self._pc_k = pc_k
        self._step = step
        self._multiply = multipl
        self._operations = nn.ModuleList()
        self.type = type
        if type == 'up':
            self.C = self.C // 2
            self.preprocess = ReLUConvBN(C_pre, self.C, 1, 1, 0, affine=False)
        if type == 'same':
            self.preprocess = ReLUConvBN(C_pre, self.C, 1, 1, 0, affine=False)
        if type == 'down':
            self.C = self.C * 2
            self.preprocess = ReLUConvBN(C_pre, self.C, 1, 1, 0, affine=False)
        for i in range(self._step):
            for j in range(1 + i):
                op = MixedOp(self.C, self.stride, self._pc_k)
                self._operations.append(op)
        # self._operations.append(nn.Upsample(size=x.size()[2:], mode='bilinear', align_corners=True))

    # alpha[1 + 2 + ...+ n-1 + n][num_ops] stands for op selection in a Block after softmax activation
    # p[1 + 2 + ...+ n-1 + n] from PCdarts stands for edge selection in a Block after softmax activation
    def forward(self, x, w_alpha, w_p):
        s = self.preprocess(x)
        states = [s]
        offset = 0
        for i in range(self._step):
            s = sum(
                w_p[offset + j] * self._operations[offset + j](hide_fteature, w_alpha[offset + j]) for j, hide_fteature
                in enumerate(states))
            offset += len(states)
            states.append(s)
        cat = torch.cat(states[-self._multiply:], dim=1)

        if self.type == 'up':
            return F.upsample(cat, size=(cat.size(2) * 2, cat.size(2) * 2), mode='bilinear', align_corners=True)
        elif self.type == 'same':
            return cat
        else:
            return F.max_pool2d(cat, kernel_size=(2, 2))


class Block(nn.Module):
    def __init__(self, x, y, max_scale=6, fl_channel=48):  # fl_channel should can be divisible by step
        super(Block, self).__init__()
        self.cells = nn.ModuleList()
        if x == 0 and y == 0:
            self.cells.append(None)
            self.cells.append(Cell(fl_channel * pow(2, y), type='same'))
            self.cells.append(None)
        elif x - y == 0:
            assert y != 0
            self.cells.append(Cell(fl_channel * pow(2, y - 1), type='down'))
            self.cells.append(None)
            self.cells.append(None)
        elif x - y == 1:
            if y == 0:
                self.cells.append(None)
                self.cells.append(Cell(fl_channel * pow(2, y), type='same'))
                self.cells.append(None)
            else:
                self.cells.append(Cell(fl_channel * pow(2, y - 1), type='down'))
                self.cells.append(Cell(fl_channel * pow(2, y), type='same'))
                self.cells.append(None)
        elif x - y > 1:
            if y == 0:
                self.cells.append(None)
                self.cells.append(Cell(fl_channel * pow(2, y), type='same'))
                self.cells.append(Cell(fl_channel * pow(2, y + 1), type='up'))
            elif y == max_scale - 1:  # from 0-max_scale-1
                self.cells.append(Cell(fl_channel * pow(2, y - 1), type='down'))
                self.cells.append(Cell(fl_channel * pow(2, y), type='same'))
                self.cells.append(None)
            else:
                self.cells.append(Cell(fl_channel * pow(2, y - 1), type='down'))
                self.cells.append(Cell(fl_channel * pow(2, y), type='same'))
                self.cells.append(Cell(fl_channel * pow(2, y + 1), type='up'))

    #
    def forward(self, x, w_alpha, w_beta, w_p):
        flag = True
        for i, _ in enumerate(x):
            if _ == None:
                continue
            if i == 1:
                if flag:
                    out = x[1] * w_beta[3]
                    flag = False
                else:
                    out += x[1] * w_beta[3]
            if flag:
                out = self.cells[i](x[i], w_alpha[i], w_p[i]) * w_beta[i]
                flag = False
            else:
                out += self.cells[i](x[i], w_alpha[i], w_p[i]) * w_beta[i]

        # out = x[1] * w_beta[3]  # skipconnection
        # for i, cell in enumerate(self.cells):
        #     if cell == None:
        #         continue
        #     out += cell(x[i], w_alpha[i], w_p[i]) * w_beta[i]
        return out


class trapezoid_supernet(pl.LightningModule):
    def __init__(self, steps=3, num_layers=15, max_scale=6, fl_channel=48):
        super(trapezoid_supernet, self).__init__()
        # give to self.
        self._steps = steps
        self.num_layers = num_layers
        self.max_scale = max_scale
        self.fl_channel = fl_channel

        C_curr = self.fl_channel

        # self.stem = nn.Sequential(
        #     nn.Conv2d(3, C_curr, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(C_curr),
        #     nn.ReLU(inplace=True),
        # )
        self.stem0 = nn.Sequential(
            nn.Conv2d(3, 24, 3, stride=2, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU()
        )
        self.stem1 = nn.Sequential(
            nn.Conv2d(24, 24, 3, stride=1,
                      padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU()
        )
        self.stem2 = nn.Sequential(
            nn.Conv2d(24, 48, 3, stride=2,
                      padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU()
        )

        self.Layers = nn.ModuleList()
        for x in range(self.num_layers):
            Blocks = nn.ModuleList()
            if x < self.max_scale:
                for y in range(0, x + 1):
                    Blocks.append(Block(x, y, self.max_scale, C_curr))
            else:
                for y in range(0, self.max_scale):
                    Blocks.append(Block(x, y, self.max_scale, C_curr))
            self.Layers.append(Blocks)

        self.Classfiers = nn.ModuleList()
        for i in range(self.max_scale):
            self.Classfiers.append(nn.Conv2d(fl_channel * pow(2, i), 20, 1))

        self._initialize_alphas()

    def forward(self, x):
        normalized_betas = torch.zeros(self.num_layers, 4, self.max_scale)

        # softmax the alphas
        for i, blocks in enumerate(self.Layers):
            j = len(blocks)
            for k in range(4):
                start = 0
                end = j
                if k == 0:  # down has no j=0 blocks
                    start = start + 1
                if k == 2:  # up has no j = max Blocks
                    end = j - 1
                try:
                    normalized_betas[i][k][:j] = F.softmax(self.betas[i][k][start:end], dim=-1)
                except:
                    pass
        normalized_betas = normalized_betas.cuda()

        normalized_alph = []
        normalized_alph.append(F.softmax(self.alphas_down, dim=-1))
        normalized_alph.append(F.softmax(self.alphas_same, dim=-1))
        normalized_alph.append(F.softmax(self.alphas_up, dim=-1))

        n = 2
        start = 1
        weights2_down = F.softmax(self.p_down[0:1], dim=-1)
        weights2_same = F.softmax(self.p_same[0:1], dim=-1)
        weights2_up = F.softmax(self.p_up[0:1], dim=-1)
        for i in range(self._steps - 1):
            end = start + n
            tw2_down = F.softmax(self.p_down[start:end], dim=-1)
            tw2_same = F.softmax(self.p_same[start:end], dim=-1)
            tw2_up = F.softmax(self.p_up[start:end], dim=-1)
            start = end
            n += 1
            weights2_down = torch.cat([weights2_down, tw2_down], dim=0)
            weights2_same = torch.cat([weights2_same, tw2_down], dim=0)
            weights2_up = torch.cat([weights2_up, tw2_down], dim=0)
        normalized_p = [weights2_down, weights2_same, weights2_up]

        xs_pre = [None for i in range(self.max_scale)]
        temp = self.stem0(x)
        temp = self.stem1(temp)
        xs_pre[0] = self.stem2(temp)

        block_counter = 0
        for i, blocks in enumerate(self.Layers):
            xs_curr = [None for n in range(self.max_scale)]
            for j, block in enumerate(blocks):
                # input to a block: _
                _ = [None, None, None]
                if j != 0:
                    _[0] = xs_pre[j - 1]
                _[1] = xs_pre[j]
                if j != self.max_scale - 1:
                    _[2] = xs_pre[j + 1]
                # alphas to block: w_beta,normalized_alph,normalized_p
                w_beta = [normalized_betas[i][n][j] for n in range(4)]
                # w_alpha = normalized_alph
                # w_p = normalized_p
                xs_curr[j] = block(_, normalized_alph, w_beta, normalized_p)
            xs_pre = xs_curr

        # classify layers
        xs_curr = [F.adaptive_avg_pool2d(self.Classfiers[i](x), 1) for i, x in enumerate(xs_curr)]
        xs_curr = torch.stack(xs_curr)
        xs_curr = torch.mean(xs_curr, dim=0)
        return xs_curr.squeeze()

    def _initialize_alphas(self):
        k_edge = sum(1 for i in range(self._steps) for n in range(1 + i))
        k_block = 0
        for x in range(self.num_layers):
            if x < self.max_scale:
                for y in range(0, x + 1):
                    k_block += 1
            else:
                for y in range(0, self.max_scale):
                    k_block += 1

        num_ops = len(PRIMITIVES)

        alphas_down = Variable(1e-3 * torch.randn(k_edge, num_ops).cuda(), requires_grad=True)
        alphas_same = Variable(1e-3 * torch.randn(k_edge, num_ops).cuda(), requires_grad=True)
        alphas_up = Variable(1e-3 * torch.randn(k_edge, num_ops).cuda(), requires_grad=True)
        p_down = Variable(1e-3 * torch.randn(k_edge).cuda(), requires_grad=True)
        p_same = Variable(1e-3 * torch.randn(k_edge).cuda(), requires_grad=True)
        p_up = Variable(1e-3 * torch.randn(k_edge).cuda(), requires_grad=True)
        betas = Variable(1e-3 * torch.randn(self.num_layers, 4, self.max_scale).cuda(), requires_grad=True)
        self._arch_parameters = [
            alphas_down,
            alphas_same,
            alphas_up,
            p_down,
            p_same,
            p_up,
            betas,
        ]
        self._arch_param_names = [
            'alphas_down',
            'alphas_same',
            'alphas_up',
            'p_down',
            'p_same',
            'p_up',
            'betas',
        ]

        [self.register_parameter(name, torch.nn.Parameter(param)) for name, param in
         zip(self._arch_param_names, self._arch_parameters)]

    def arch_parameters(self):
        return [param for name, param in self.named_parameters() if name in self._arch_param_names]

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if name not in self._arch_param_names]

    def named_parameters_arch(self, prefix='', recurse=True):
        gen = self._named_members(
            lambda module: module._parameters.items(),
            prefix=prefix, recurse=recurse)
        for elem in gen:
            if elem[0] in self._arch_param_names:
                yield elem
            else:
                continue

    def named_parameters_weight(self, prefix='', recurse=True):
        gen = self._named_members(
            lambda module: module._parameters.items(),
            prefix=prefix, recurse=recurse)
        for elem in gen:
            if elem[0] not in self._arch_param_names:
                yield elem
            else:
                continue


    def training_step(self, batch, batch_nb):
        x, y = batch
        # print(x.shape)
        logits = self(x)
        loss = F.multilabel_soft_margin_loss(logits, y)
        return {'loss': loss}

    # def validation_step(self,batch,batch_nb):
    #     x, y = batch
    #     logits = self(x)
    #     loss = F.MultiLabelSoftMarginLoss(logits, y)
    #     self.outputs.append({'val_loss': loss})
    #     return loss

    def train_dataloader(self):
        return DataLoader(
            mydataset(dataroot='/NAS_REMOTE/shaozl/dataset/Pascal_VOC/VOC_2012/VOCdevkit/VOC2012/JPEGImages/',
                      lableroot='/NAS_REMOTE/shaozl/MS-CAM-NAS/', phase='train'), batch_size=14)
        # DataLoader(mydataset(dataroot='/NAS_REMOTE/shaozl/dataset/Pascal_VOC/VOC_2012/VOCdevkit/VOC2012/JPEGImages/',
        #                      lableroot='/NAS_REMOTE/shaozl/MS-CAM-NAS/', phase='val'), batch_size=14)

    # def val_dataloader(self):
    #     return mydataset(dataroot='/NAS_REMOTE/shaozl/dataset/Pascal_VOC/VOC_2012/VOCdevkit/VOC2012/JPEGImages/',lableroot='/NAS_REMOTE/shaozl/MS-CAM-NAS/',phase='val')

    def configure_optimizers(self):
        optimizer_1 = torch.optim.SGD(self.parameters(), lr=0.05)
        optimizer_2 = torch.optim.Adam(self.arch_parameters, lr=0.001)
        return [optimizer_1, optimizer_2]

    def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_i, second_order_closure, on_tpu,
                       using_native_amp, using_lbfgs):
        if optimizer_i == 0:
            optimizer.step()
            optimizer.zero_grad()

        if optimizer_i == 1:
            if current_epoch >= 0:
                optimizer.step()
                optimizer.zero_grad()



# CUDA_VISIBLE_DEVICES=0
if __name__ == '__main__':
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    max_scale = 4
    net = trapezoid_supernet(max_scale=max_scale, num_layers=12).cuda()
    # for i in net.parameters():
    #     print(type(i))
    #
    # print(type(net.arch_parameters()))
    dataLoader = DataLoader(
        mydataset(dataroot='/NAS_REMOTE/shaozl/dataset/Pascal_VOC/VOC_2012/VOCdevkit/VOC2012/JPEGImages/',
                  lableroot='/NAS_REMOTE/shaozl/MS-CAM-NAS/', phase='train'), batch_size=14)
    for x, y in dataLoader:
        # x = torch.randn(14, 3, 224, 224).cuda()
        x = x.cuda()
        print(x.shape)
        out = net(x)
        print(out)
        exit(0)
    # print('ok')
    # for i in range(max_scale):
    #     print(out[i].shape)
    import time

    time.sleep(3600)
