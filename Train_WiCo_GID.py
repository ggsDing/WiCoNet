import os
import time
import torch.autograd
from skimage import io
from torch import optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

working_path = os.path.dirname(os.path.abspath(__file__))
from utils.loss import CrossEntropyLoss2d
from utils.utils import accuracy, AverageMeter

# Choose model and data
##################################################
from datasets import GID_WiCo as GID
from models.WiCoNet import WiCoNet as Net

NET_NAME = 'WiCoNet_3hw4L'
DATA_NAME = 'GID'
##################################################


# Change training parameters here
args = {
    'train_batch_size': 32,
    'val_batch_size': 32,
    'lr': 0.1,
    'epochs': 50,
    'gpu': True,
    'size_local': 256,
    'size_context': 256 * 3,
    'momentum': 0.9,
    'crop_nums': 100,
    'weight_decay': 5e-4,
    'lr_decay_power': 1.5,
    'print_freq': 100,
    'save_pred': True,
    'num_workers': 4,
    'data_dir': 'YOUR_DATA_DIR',
    'pred_dir': os.path.join(working_path, 'results', DATA_NAME),
    'chkpt_dir': os.path.join(working_path, 'checkpoints', DATA_NAME),
    'log_dir': os.path.join(working_path, 'logs', DATA_NAME, NET_NAME)
}

if not os.path.exists(args['chkpt_dir']): os.makedirs(args['chkpt_dir'])
if not os.path.exists(args['pred_dir']): os.makedirs(args['pred_dir'])
if not os.path.exists(args['log_dir']): os.makedirs(args['log_dir'])
writer = SummaryWriter(args['log_dir'])


def main():
    net = Net(4, num_classes=GID.num_classes + 1, size_context=args['size_context'],
              size_local=args['size_local']).cuda()

    train_set = GID.Loader(args['data_dir'], 'train', random_crop=True, crop_nums=args['crop_nums'], random_flip=True,
                           size_context=args['size_context'], size_local=args['size_local'])
    train_loader = DataLoader(train_set, batch_size=args['train_batch_size'], num_workers=args['num_workers'], shuffle=True)
    val_set = GID.Loader(args['data_dir'], 'val', sliding_crop=True, size_context=args['size_context'], size_local=args['size_local'])
    val_loader = DataLoader(val_set, batch_size=args['val_batch_size'], num_workers=args['num_workers'], shuffle=False)

    criterion = CrossEntropyLoss2d(ignore_index=0).cuda()
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args['lr'],
                          weight_decay=args['weight_decay'], momentum=args['momentum'], nesterov=True)

    train(train_loader, net, criterion, optimizer, val_loader)
    writer.close()
    print('Training finished.')


def train(train_loader, net, criterion, optimizer, val_loader):
    bestaccT = 0
    bestaccV = 0.5
    bestloss = 1
    curr_epoch = 0
    begin_time = time.time()
    all_iters = float(len(train_loader) * args['epochs'])
    while True:
        torch.cuda.empty_cache()
        net.train()
        start = time.time()
        acc_meter = AverageMeter()
        train_loss = AverageMeter()

        curr_iter = curr_epoch * len(train_loader)
        for i, data in enumerate(train_loader):
            running_iter = curr_iter + i + 1
            adjust_learning_rate(optimizer, running_iter, all_iters)
            imgs_s, labels_s, imgs, labels = data
            if args['gpu']:
                imgs = imgs.cuda().float()
                labels = labels.cuda().long()
                imgs_s = imgs_s.cuda().float()
                labels_s = labels_s.cuda().long()

            optimizer.zero_grad()
            outputs, aux = net(imgs_s, imgs)

            alpha = calc_alpha(running_iter, all_iters)
            main_loss = criterion(outputs, labels)
            aux_loss = criterion(aux, labels_s)
            loss = main_loss + alpha * aux_loss
            loss.backward()
            optimizer.step()

            labels = labels.cpu().detach().numpy()
            outputs = outputs.cpu().detach()
            preds = torch.argmax(outputs, dim=1)
            preds = preds.numpy()
            # batch_valid_sum = 0
            acc_curr_meter = AverageMeter()
            for (pred, label) in zip(preds, labels):
                acc, valid_sum = accuracy(pred, label)
                acc_curr_meter.update(acc)
            acc_meter.update(acc_curr_meter.avg)
            train_loss.update(loss.cpu().detach().numpy())
            curr_time = time.time() - start

            if (i + 1) % args['print_freq'] == 0:
                print('[epoch %d] [iter %d / %d %.1fs] [lr %f] [train loss %.4f acc %.2f]' % (
                    curr_epoch, i + 1, len(train_loader), curr_time, optimizer.param_groups[0]['lr'],
                    train_loss.val, acc_meter.val * 100))
                writer.add_scalar('train loss', train_loss.val, running_iter)
                writer.add_scalar('train accuracy', acc_meter.val, running_iter)
                writer.add_scalar('lr', optimizer.param_groups[0]['lr'], running_iter)

        acc_v, loss_v = validate(val_loader, net, criterion, curr_epoch)
        if acc_meter.avg > bestaccT: bestaccT = acc_meter.avg
        if acc_v > bestaccV:
            bestaccV = acc_v
            bestloss = loss_v
            save_path = os.path.join(args['chkpt_dir'], NET_NAME + '_%de_OA%.2f.pth' % (curr_epoch, acc_v * 100))
            torch.save(net.state_dict(), save_path)
        print('Total time: %.1fs Best rec: Train %.2f, Val %.2f, Val_loss %.4f' \
              % (time.time() - begin_time, bestaccT * 100, bestaccV * 100, bestloss))
        curr_epoch += 1
        if curr_epoch >= args['epochs']:
            return


def validate(val_loader, net, criterion, curr_epoch):
    # the following code is written assuming that batch size is 1
    net.eval()
    torch.cuda.empty_cache()
    start = time.time()

    val_loss = AverageMeter()
    acc_meter = AverageMeter()

    for vi, data in enumerate(val_loader):
        imgs_s, labels_s, imgs, labels = data

        if args['gpu']:
            imgs = imgs.cuda().float()
            labels = labels.cuda().long()
            imgs_s = imgs_s.cuda().float()

        with torch.no_grad():
            outputs, _ = net(imgs_s, imgs)
            loss = criterion(outputs, labels)
        val_loss.update(loss.cpu().detach().numpy())

        outputs = outputs.cpu().detach()
        labels = labels.cpu().detach().numpy()
        preds = torch.argmax(outputs, dim=1)
        preds = preds.numpy()
        for (pred, label) in zip(preds, labels):
            acc, valid_sum = accuracy(pred, label)
            acc_meter.update(acc)

        if args['save_pred'] and vi == 0:
            pred_color = GID.Index2Color(preds[0])
            pred_path = os.path.join(args['pred_dir'], NET_NAME + '.png')
            io.imsave(pred_path, pred_color)
            print('Prediction saved!')

    curr_time = time.time() - start
    print('%.1fs Val loss: %.2f Accuracy: %.2f' % (curr_time, val_loss.average(), acc_meter.average() * 100))

    writer.add_scalar('val_loss', val_loss.average(), curr_epoch)
    writer.add_scalar('val_Accuracy', acc_meter.average(), curr_epoch)

    return acc_meter.avg, val_loss.avg


def calc_alpha(curr_iter, all_iters, weight=1.0):
    r = (1.0 - float(curr_iter) / all_iters) ** 2.0
    return weight * r


def adjust_learning_rate(optimizer, curr_iter, all_iter):
    scale_running_lr = ((1. - float(curr_iter) / all_iter) ** args['lr_decay_power'])
    running_lr = args['lr'] * scale_running_lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = running_lr


if __name__ == '__main__':
    main()
