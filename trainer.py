import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from albumentations import RandomRotate90,Resize
from albumentations.core.composition import Compose, OneOf
from utils import DiceLoss
from torchvision import transforms
from albumentations import augmentations
from datasets.dataset_isic import isic_loader
from glob import glob
from sklearn.model_selection import train_test_split
import albumentations as albu
from collections import OrderedDict
from datasets.dataset_busi import Dataset
from metircs import iou_score
from utils import AverageMeter
from utils import AverageMeter
import torch.nn.functional as F
from datasets.dataset_mix import *
from test import inference
from torch.optim import lr_scheduler
def trainer_synapse(args, model, snapshot_path):
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + f"/{args.config_file}_nocbam_log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu # 这里算的是每个  gpu × 每张卡的batchsize
    db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        model.train()
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.type(torch.FloatTensor).cuda(), label_batch.type(torch.FloatTensor).cuda()
            outputs = model(image_batch)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.4 * loss_ce + 0.6 * loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)

            if iter_num % 10 ==0:
                logging.info('iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item()))

            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

        save_interval = 10
        if epoch_num >= (max_epoch-40) : #保存最后30轮的结果
            args.Dataset = Synapse_dataset
            args.volume_path = "/home/JianjianYin/Swin-transUper/data/Synapse/test_vol_h5"
            performance = inference(args, model, None)
            if performance > best_performance:
                filename = f'{args.config_file}_epoch_best.pth'
                save_mode_path = os.path.join("/home/JianjianYin/Swin-transUper/Synapse_4_6", filename)
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

        # if epoch_num >= max_epoch - 1: # 保存最后一次
        #     filename = f'{args.config_file}_epoch_{epoch_num}.pth'
        #     save_mode_path = os.path.join(snapshot_path, filename)
        #     torch.save(model.state_dict(), save_mode_path)
        #     logging.info("save model to {}".format(save_mode_path))
        #     iterator.close()
        #     break

    writer.close()
    return "Training Finished!"



def trainer_ISIC(args, model, snapshot_path):
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + f"/{args.config_file}_isic.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu # 这里算的是每个  gpu × 每张卡的batchsize
    db_train = isic_loader(path_Data = args.root_path, train = True)
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    #ce_loss = CrossEntropyLoss()
    ce_loss = torch.nn.BCEWithLogitsLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            outputs = model(image_batch)
            loss_ce = ce_loss(outputs.float(), label_batch[:])
            label_batch = label_batch.squeeze(1)
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.9 * loss_ce + 0.1 * loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            if i_batch % 30==0:
                logging.info('iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item()))

            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

        save_interval = 10
        if epoch_num >= int(max_epoch-40) : #保存最后30轮的结果
            filename = f'{args.config_file}_epoch_{epoch_num}.pth'
            save_mode_path = os.path.join(snapshot_path,"isic_9_1", filename)
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        # if epoch_num >= max_epoch - 1: # 保存最后一次
        #     filename = f'{args.config_file}_epoch_{epoch_num}.pth'
        #     save_mode_path = os.path.join(snapshot_path, filename)
        #     torch.save(model.state_dict(), save_mode_path)
        #     logging.info("save model to {}".format(save_mode_path))
        #     iterator.close()
        #     break

    writer.close()
    return "Training Finished!"

def intersectionAndUnion(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert output.ndim in [1, 2, 3]
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = ignore_index
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K + 1))
    area_output, _ = np.histogram(output, bins=np.arange(K + 1))
    area_target, _ = np.histogram(target, bins=np.arange(K + 1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def trainer_busi(args, model, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    model.train()
    ce_loss = torch.nn.BCEWithLogitsLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.Adam(model.parameters(), lr=base_lr, weight_decay=0.0001)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-5)
    #optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    # img_ids = glob(os.path.join('inputs', 'busi', 'images', '*' + '.png'))
    # img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids] # 获得所有的照片的名称
    # print(img_ids)
    # train_img_ids, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=41) # 划分比例
    train_transform = Compose([
        RandomRotate90(),
        albu.Flip(),
        Resize(224, 224), 
        augmentations.transforms.Normalize(),
    ])
    # val_transform = Compose([
    #     Resize(224, 224),
    #     augmentations.transforms.Normalize(),
    # ])
    train_dataset = Dataset(
        img_dir=os.path.join('inputs', 'busi', 'images'),
        mask_dir=os.path.join('inputs', 'busi', 'masks'),
        img_ext='.png',
        mask_ext='.png',
        num_classes=1,
        transform=train_transform)
    # val_dataset = Dataset(
    #     img_ids=val_img_ids,
    #     img_dir=os.path.join('inputs', 'busi', 'images'),
    #     mask_dir=os.path.join('inputs', 'busi', 'masks'),
    #     img_ext='.png',
    #     mask_ext='.png',
    #     num_classes=1,
    #     transform=val_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0,
        drop_last=True)
    # val_loader = torch.utils.data.DataLoader(
    #     val_dataset,
    #     batch_size=1,
    #     shuffle=False,
    #     num_workers=0,
    #     drop_last=False)
    log = OrderedDict([
        ('epoch', []),
        ('lr', []),
        ('loss', []),
        ('iou', []),
        ('val_loss', []),
        ('val_iou', []),
        ('val_dice', []),
    ])
    best_iou = 0
    max_train_epoch = 200
    for epoch in range(max_train_epoch):
        print('Epoch [%d/%d]' % (epoch, max_train_epoch))
        # train for one epoch
        train_log = train(train_loader,base_lr,ce_loss,dice_loss, model, optimizer,max_train_epoch)
        # evaluate on validation set
        #val_log = validate( val_loader, model)
        scheduler.step()

        #print('- iou %.4f - - val_iou %.4f  --val_dice %.4f' % ( train_log['iou'],  val_log['iou'],val_log['dice']))
        print('- iou %.4f' % ( train_log['iou']))
        if train_log['iou'] > best_iou:
            torch.save(model.state_dict(), 'model.pth')
            best_iou = train_log['iou']
            print("=> saved best model")

# def trainer_busi(args, model, snapshot_path):
#     base_lr = args.base_lr
#     num_classes = args.num_classes
#     model.train()
#     ce_loss = torch.nn.BCEWithLogitsLoss()
#     dice_loss = DiceLoss(num_classes)
#     optimizer = optim.Adam(model.parameters(), lr=base_lr, weight_decay=0.0001)
#     scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-5)
#     #optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
#     # img_ids = glob(os.path.join('inputs', 'busi', 'images', '*' + '.png'))
#     # img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids] # 获得所有的照片的名称
#     # print(img_ids)
#     # train_img_ids, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=41) # 划分比例
#     train_transform = Compose([
#         RandomRotate90(),
#         albu.Flip(),
#         Resize(224, 224), 
#         augmentations.transforms.Normalize(),
#     ])
#     # val_transform = Compose([
#     #     Resize(224, 224),
#     #     augmentations.transforms.Normalize(),
#     # ])
#     train_dataset = Dataset(
#         img_dir=os.path.join('inputs', 'busi', 'images'),
#         mask_dir=os.path.join('inputs', 'busi', 'masks'),
#         img_ext='.png',
#         mask_ext='.png',
#         num_classes=1,
#         transform=train_transform)
#     # val_dataset = Dataset(
#     #     img_ids=val_img_ids,
#     #     img_dir=os.path.join('inputs', 'busi', 'images'),
#     #     mask_dir=os.path.join('inputs', 'busi', 'masks'),
#     #     img_ext='.png',
#     #     mask_ext='.png',
#     #     num_classes=1,
#     #     transform=val_transform)

#     train_loader = torch.utils.data.DataLoader(
#         train_dataset,
#         batch_size=8,
#         shuffle=True,
#         num_workers=0,
#         drop_last=True)
#     # val_loader = torch.utils.data.DataLoader(
#     #     val_dataset,
#     #     batch_size=1,
#     #     shuffle=False,
#     #     num_workers=0,
#     #     drop_last=False)
#     log = OrderedDict([
#         ('epoch', []),
#         ('lr', []),
#         ('loss', []),
#         ('iou', []),
#         ('val_loss', []),
#         ('val_iou', []),
#         ('val_dice', []),
#     ])
#     best_iou = 0
#     max_train_epoch = 200
#     for epoch in range(max_train_epoch):
#         print('Epoch [%d/%d]' % (epoch, max_train_epoch))
#         # train for one epoch
#         train_log = train(train_loader,base_lr,ce_loss,dice_loss, model, optimizer,max_train_epoch)
#         # evaluate on validation set
#         #val_log = validate( val_loader, model)
#         scheduler.step()

#         #print('- iou %.4f - - val_iou %.4f  --val_dice %.4f' % ( train_log['iou'],  val_log['iou'],val_log['dice']))
#         print('- iou %.4f' % ( train_log['iou']))
#         if train_log['iou'] > best_iou:
#             torch.save(model.state_dict(), 'model.pth')
#             best_iou = train_log['iou']
#             print("=> saved best model")

def trainer_mix(args,model,snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    print(num_classes)
    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    train_transform = Compose([
        RandomRotate90(),
        albu.Flip(),
        Resize(224, 224), 
        augmentations.transforms.Normalize(),
    ])
    train_dataset = mix_data(args.root_path,transform = train_transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0,
        drop_last=True)
    
    best_iou = 0
    iter_num = 0
    max_train_epoch = 200
    max_iterations = args.max_epochs * len(train_loader)
    for epoch in range(max_train_epoch):
        print('Epoch [%d/%d]' % (epoch, max_train_epoch))
        intersection_meter = AverageMeter()
        union_meter = AverageMeter()
        pbar = tqdm(total=len(train_loader))
        for image, mask,id in train_loader:
            image = image.cuda()
            mask = mask.cuda()
            output = model(image)
            mask = mask.squeeze(1)
            #print(mask.unique(),id)
            loss_ce = ce_loss(output, mask[:].long())
            loss_dice = dice_loss(output, mask, softmax=True)
            loss = 0.4 * loss_ce + 0.6 * loss_dice

            pred = torch.softmax(output,dim=1)
            pred = torch.argmax(pred,dim=1)

            intersection, union, _ = intersectionAndUnion(pred.cpu().numpy(), mask.cpu().numpy(), 4, 255)
            intersection_meter.update(intersection)
            union_meter.update(union)   
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1

            logging.info('iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item()))
        iou_class = intersection_meter.sum / (union_meter.sum + 1e-10) * 100.0
        mIOU = np.mean(iou_class)
        print(iou_class,mIOU)
        logging.info("每个类的iou为:%s"%(iou_class))
        logging.info('平均的IOU为%s'%(mIOU))
    torch.save(model.state_dict(), 'model.pth')


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.5 * bce + dice
    

def train(train_loader,base_lr,ce_loss,dice_loss, model, optimizer,max_train_epoch):
    avg_meters = {
                  'iou': AverageMeter()}

    model.train()

    pbar = tqdm(total=len(train_loader))
    critection = BCEDiceLoss()
    for sample in train_loader:
        input = sample['img'].cuda()
        target = sample['mask'].cuda()
        output = model(input)
        loss = critection(output,target)
        iou,dice = iou_score(output,target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_meters['iou'].update(iou, input.size(0))
        postfix = OrderedDict([
            ('iou', avg_meters['iou'].avg),
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
    
    pbar.close()

    return OrderedDict([('iou', avg_meters['iou'].avg)])


def validate( val_loader, model):
    avg_meters = {
                  'iou': AverageMeter(),
                   'dice': AverageMeter()}
     # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for input, target, _ in val_loader:
            input = input.cuda()
            target = target.cuda()
            output = model(input)
            iou,dice = iou_score(output, target)
            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['dice'].update(dice, input.size(0))
            postfix = OrderedDict([
                ('iou', avg_meters['iou'].avg),
                ('dice', avg_meters['dice'].avg)
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

    return OrderedDict([('iou', avg_meters['iou'].avg),
                        ('dice', avg_meters['dice'].avg)])

    


    

