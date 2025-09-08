import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from dataset.MYO.MYO_dataset import CAMUS3KDataset, HMCQUDataset
from dataset.Thyroid.Thyroid_dataset import TN3KDataset, DDTIDataset
from dataset.transform import transforms, joint_transforms
from dataset.BUSI.BUSI_dataset import BUSIDataset, DatasetBDataset, STUDataset
from dataset.transform.normalize import normalize_image
from test import inference
from utils import dice_coeff, bce_loss




def calc_loss(outputs, label_batch, ce_loss, dice_loss, dice_weight:float=0.8):
    logits = outputs['masks']
    pred = torch.nn.Sigmoid()(logits)
    loss_ce = ce_loss(pred=pred, label=label_batch)
    loss_dice = dice_loss(pred=pred, label=label_batch)
    loss = (1 - dice_weight) * loss_ce + dice_weight * loss_dice
    return loss, loss_ce, loss_dice

def trainer_busi(args, model, snapshot_path, multimask_output, low_res):
    #single-source-domain
    source_name = args.Source_Dataset
    print('Training Phase')
    print("source:"+source_name)

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations

    train_joint_transform = joint_transforms.Compose([
        joint_transforms.To_PIL_Image(),
        joint_transforms.RandomAffine(0, translate=(0.125, 0.125)),
        joint_transforms.RandomHorizontallyFlip(),
        #joint_transforms.RandomRotate((-30, 30)),
        joint_transforms.FixResize(args.img_size)
    ])
    transform = transforms.Compose([
        # transforms.RandomContrast(0.5),
        # transforms.RandomGammaEnhancement(0.5),
        # transforms.RandomGaussianBlur(),
        transforms.to_Tensor(),
        # standard_augment.normalize([cfg.DATASET.MEAN], [cfg.DATASET.STD])
    ])
    target_transform = transforms.Compose([
        transforms.to_Tensor()])

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)
    source_dataset = BUSIDataset(root=args.source_root_path, list_path=args.train_list_path, num_class=num_classes,
                            joint_augment=train_joint_transform,
                            augment=transform, target_augment=target_transform)

    trainloader = DataLoader(dataset=source_dataset,
                             batch_size=batch_size,
                             shuffle=True,
                             pin_memory=True,
                             num_workers=args.num_workers,
                             worker_init_fn=worker_init_fn)

    #to rest target domain
    target_name = args.Target_Dataset
    print("target:"+str(target_name))

    eval_transform = joint_transforms.Compose([
        joint_transforms.To_PIL_Image(),
        joint_transforms.FixResize(args.img_size)
        ])
    transform = transforms.Compose([
        transforms.to_Tensor(),
    ])
    target_transform = transforms.Compose([
        transforms.to_Tensor()])
    testloader = []
    for t_n in target_name:
        if t_n=='BUSI':
            test_set = BUSIDataset(root=args.source_root_path, list_path='dataset/BUSI/test.txt', num_class=1,
                                   joint_augment=eval_transform,
                                   augment=transform, target_augment=target_transform)
        elif t_n=='DatasetB':
            test_set = DatasetBDataset(root=args.target_root_path1, num_class=1,
                                   joint_augment=eval_transform,
                                   augment=transform, target_augment=target_transform)
        elif t_n=='STU':
            test_set = STUDataset(root=args.target_root_path2, num_class=1,
                                   joint_augment=eval_transform,
                                   augment=transform, target_augment=target_transform)
        testloader.append(DataLoader(dataset=test_set, batch_size=1, pin_memory=True,
                                 num_workers=4, shuffle=False))


    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    ce_loss = bce_loss
    dice_loss = dice_coeff
    if args.warmup:
        b_lr = base_lr / args.warmup_period
    else:
        b_lr = base_lr
    if args.AdamW:
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=b_lr, betas=(0.9, 0.999), weight_decay=0.1)
    else:
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=b_lr, momentum=0.9, weight_decay=0.0001)  # Even pass the model.parameters(), the `requires_grad=False` layers will not update
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    stop_epoch = args.stop_epoch
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = (range(max_epoch))
    for epoch_num in iterator:
        logging.info('Epoch %d / %d' % (epoch_num, max_epoch))
        for batch, data in enumerate(trainloader):
            model.train()
            x, y = data['image'], data['mask'][0]
            x = normalize_image(x)
            x, y = x.cuda(), y.cuda()

            outputs = model(x, multimask_output, args.img_size)
            loss, loss_ce, loss_dice = calc_loss(outputs, y, ce_loss, dice_loss, args.dice_param)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if args.warmup and iter_num < args.warmup_period:
                lr_ = base_lr * ((iter_num + 1) / args.warmup_period)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            else:
                if args.warmup:
                    shift_iter = iter_num - args.warmup_period
                    assert shift_iter >= 0, f'Shift iter is {shift_iter}, smaller than zero'
                else:
                    shift_iter = iter_num
                lr_ = base_lr * (1.0 - shift_iter / max_iterations) ** 0.9  # learning rate adjustment depends on the max iterations
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)
            logging.info('iteration %d : loss : %f, loss_ce: %f, loss_dice: %f, lr: %f' % (iter_num, loss.item(), loss_ce.item(), loss_dice.item(), lr_))

        save_interval = 20 # int(max_epoch/6)
        if (epoch_num + 1) % save_interval == 0:
            #test
            dice0, asd0 = inference(args=args, epoch=epoch_num, snapshot_path=snapshot_path, test_loader=testloader[0], model=model, dataset_name=target_name[0])
            dice1, asd1 = inference(args=args, epoch=epoch_num, snapshot_path=snapshot_path, test_loader=testloader[1], model=model, dataset_name=target_name[1])
            dice2, asd2 = inference(args=args, epoch=epoch_num, snapshot_path=snapshot_path, test_loader=testloader[2], model=model, dataset_name=target_name[2])
            if dice0 > 0.8300 and dice1 > 0.8920 and dice2 > 0.7800:
                #save model
                save_mode_path = os.path.join(snapshot_path, 'bus_weight.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1 or epoch_num >= stop_epoch - 1:

            #test

            # save model
            # save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            # torch.save(model.state_dict(), save_mode_path)
            # logging.info("save model to {}".format(save_mode_path))
            break

    writer.close()
    return "Training Finished!"


def trainer_thyroid(args, model, snapshot_path, multimask_output, low_res):
    #single-source-domain
    source_name = args.Source_Dataset
    print('Training Phase')
    print("source:"+source_name)

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations

    train_joint_transform = joint_transforms.Compose([
        joint_transforms.To_PIL_Image(),
        joint_transforms.RandomAffine(0, translate=(0.125, 0.125)),
        joint_transforms.RandomHorizontallyFlip(),
        #joint_transforms.RandomRotate((-30, 30)),
        joint_transforms.FixResize(args.img_size)
    ])
    transform = transforms.Compose([
        # transforms.RandomContrast(0.5),
        # transforms.RandomGammaEnhancement(0.5),
        # transforms.RandomGaussianBlur(),
        transforms.to_Tensor(),
        # standard_augment.normalize([cfg.DATASET.MEAN], [cfg.DATASET.STD])
    ])
    target_transform = transforms.Compose([
        transforms.to_Tensor()])

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)
    source_dataset = TN3KDataset(root=args.source_root_path, list_path=args.train_list_path, mode="train", num_class=num_classes,
                            joint_augment=train_joint_transform,
                            augment=transform, target_augment=target_transform)

    trainloader = DataLoader(dataset=source_dataset,
                             batch_size=batch_size,
                             shuffle=True,
                             pin_memory=True,
                             num_workers=args.num_workers,
                             worker_init_fn=worker_init_fn)

    #to rest target domain
    target_name = args.Target_Dataset
    print("target:"+str(target_name))

    eval_transform = joint_transforms.Compose([
        joint_transforms.To_PIL_Image(),
        joint_transforms.FixResize(args.img_size)
        ])
    transform = transforms.Compose([
        transforms.to_Tensor(),
    ])
    target_transform = transforms.Compose([
        transforms.to_Tensor()])
    testloader = []
    for t_n in target_name:
        if t_n=='TN3K':
            test_set = TN3KDataset(root=args.source_root_path, list_path=args.train_list_path, mode="test", num_class=1,
                                   joint_augment=eval_transform,
                                   augment=transform, target_augment=target_transform)
        elif t_n=='DDTI':
            test_set = DDTIDataset(root=args.target_root_path, num_class=1,
                                   joint_augment=eval_transform,
                                   augment=transform, target_augment=target_transform)
        testloader.append(DataLoader(dataset=test_set, batch_size=1, pin_memory=True,
                                 num_workers=4, shuffle=False))


    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    ce_loss = bce_loss
    dice_loss = dice_coeff
    if args.warmup:
        b_lr = base_lr / args.warmup_period
    else:
        b_lr = base_lr
    if args.AdamW:
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=b_lr, betas=(0.9, 0.999), weight_decay=0.1)
    else:
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=b_lr, momentum=0.9, weight_decay=0.0001)  # Even pass the model.parameters(), the `requires_grad=False` layers will not update
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    stop_epoch = args.stop_epoch
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = (range(max_epoch))
    for epoch_num in iterator:
        logging.info('Epoch %d / %d' % (epoch_num, max_epoch))
        for batch, data in enumerate(trainloader):
            model.train()
            x, y = data['image'], data['mask'][0]
            x = normalize_image(x)
            x, y = x.cuda(), y.cuda()

            outputs = model(x, multimask_output, args.img_size)
            loss, loss_ce, loss_dice = calc_loss(outputs, y, ce_loss, dice_loss, args.dice_param)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if args.warmup and iter_num < args.warmup_period:
                lr_ = base_lr * ((iter_num + 1) / args.warmup_period)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            else:
                if args.warmup:
                    shift_iter = iter_num - args.warmup_period
                    assert shift_iter >= 0, f'Shift iter is {shift_iter}, smaller than zero'
                else:
                    shift_iter = iter_num
                lr_ = base_lr * (1.0 - shift_iter / max_iterations) ** 0.9  # learning rate adjustment depends on the max iterations
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)
            logging.info('iteration %d : loss : %f, loss_ce: %f, loss_dice: %f, lr: %f' % (iter_num, loss.item(), loss_ce.item(), loss_dice.item(), lr_))

        save_interval = 20 # int(max_epoch/6)
        if (epoch_num + 1) % save_interval == 0:
            #test
            dice0, asd0 = inference(args=args, epoch=epoch_num, snapshot_path=snapshot_path, test_loader=testloader[0], model=model, dataset_name=target_name[0])
            dice1, asd1 = inference(args=args, epoch=epoch_num, snapshot_path=snapshot_path, test_loader=testloader[1], model=model, dataset_name=target_name[1])
            if dice0 > 0.7750:
                #save model
                save_mode_path = os.path.join(snapshot_path, 'thyroid_weight.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1 or epoch_num >= stop_epoch - 1:

            #test

            # save model
            # save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            # torch.save(model.state_dict(), save_mode_path)
            # logging.info("save model to {}".format(save_mode_path))
            break

    writer.close()
    return "Training Finished!"



def trainer_myo(args, model, snapshot_path, multimask_output, low_res):
    #single-source-domain
    source_name = args.Source_Dataset
    print('Training Phase')
    print("source:"+source_name)

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations

    train_joint_transform = joint_transforms.Compose([
        joint_transforms.To_PIL_Image(),
        joint_transforms.RandomAffine(0, translate=(0.125, 0.125)),
        joint_transforms.RandomHorizontallyFlip(),
        #joint_transforms.RandomRotate((-30, 30)),
        joint_transforms.FixResize(args.img_size)
    ])
    transform = transforms.Compose([
        # transforms.RandomContrast(0.5),
        # transforms.RandomGammaEnhancement(0.5),
        # transforms.RandomGaussianBlur(),
        transforms.to_Tensor(),
        # standard_augment.normalize([cfg.DATASET.MEAN], [cfg.DATASET.STD])
    ])
    target_transform = transforms.Compose([
        transforms.to_Tensor()])

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)
    source_dataset = CAMUS3KDataset(root=args.source_root_path, list_path=args.train_list_path, mode="train", num_class=num_classes,
                            joint_augment=train_joint_transform,
                            augment=transform, target_augment=target_transform)

    trainloader = DataLoader(dataset=source_dataset,
                             batch_size=batch_size,
                             shuffle=True,
                             pin_memory=True,
                             num_workers=args.num_workers,
                             worker_init_fn=worker_init_fn)

    #to rest target domain
    target_name = args.Target_Dataset
    print("target:"+str(target_name))

    eval_transform = joint_transforms.Compose([
        joint_transforms.To_PIL_Image(),
        joint_transforms.FixResize(args.img_size)
        ])
    transform = transforms.Compose([
        transforms.to_Tensor(),
    ])
    target_transform = transforms.Compose([
        transforms.to_Tensor()])
    testloader = []
    for t_n in target_name:
        if t_n=='HMCQU':
            test_set = HMCQUDataset(root=args.target_root_path, num_class=1,
                                   joint_augment=eval_transform,
                                   augment=transform, target_augment=target_transform)

        testloader.append(DataLoader(dataset=test_set, batch_size=1, pin_memory=True,
                                 num_workers=4, shuffle=False))


    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    ce_loss = bce_loss
    dice_loss = dice_coeff
    if args.warmup:
        b_lr = base_lr / args.warmup_period
    else:
        b_lr = base_lr
    if args.AdamW:
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=b_lr, betas=(0.9, 0.999), weight_decay=0.1)
    else:
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=b_lr, momentum=0.9, weight_decay=0.0001)  # Even pass the model.parameters(), the `requires_grad=False` layers will not update
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    stop_epoch = args.stop_epoch
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = (range(max_epoch))
    for epoch_num in iterator:
        logging.info('Epoch %d / %d' % (epoch_num, max_epoch))
        for batch, data in enumerate(trainloader):
            model.train()
            x, y = data['image'], data['mask'][0]
            x = normalize_image(x)
            x, y = x.cuda(), y.cuda()

            outputs = model(x, multimask_output, args.img_size)
            loss, loss_ce, loss_dice = calc_loss(outputs, y, ce_loss, dice_loss, args.dice_param)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if args.warmup and iter_num < args.warmup_period:
                lr_ = base_lr * ((iter_num + 1) / args.warmup_period)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            else:
                if args.warmup:
                    shift_iter = iter_num - args.warmup_period
                    assert shift_iter >= 0, f'Shift iter is {shift_iter}, smaller than zero'
                else:
                    shift_iter = iter_num
                lr_ = base_lr * (1.0 - shift_iter / max_iterations) ** 0.9  # learning rate adjustment depends on the max iterations
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)
            logging.info('iteration %d : loss : %f, loss_ce: %f, loss_dice: %f, lr: %f' % (iter_num, loss.item(), loss_ce.item(), loss_dice.item(), lr_))

        save_interval = 20 # int(max_epoch/6)
        if (epoch_num + 1) % save_interval == 0:
            #test
            dice0, asd0 = inference(args=args, epoch=epoch_num, snapshot_path=snapshot_path, test_loader=testloader[0], model=model, dataset_name=target_name[0])
            if dice0 > 0.7770:
                #save model
                save_mode_path = os.path.join(snapshot_path, 'myo_weight.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1 or epoch_num >= stop_epoch - 1:

            #test

            # save model
            # save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            # torch.save(model.state_dict(), save_mode_path)
            # logging.info("save model to {}".format(save_mode_path))
            break

    writer.close()
    return "Training Finished!"
