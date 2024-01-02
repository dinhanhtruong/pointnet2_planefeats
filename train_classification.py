"""
Author: Benny
Date: Nov 2019
"""

import os
import sys
import torch
import numpy as np
import wandb
import datetime
import logging
import provider
import importlib
import shutil
import argparse
import pytorch3d.loss

from pathlib import Path
from tqdm import tqdm
from data_utils.ModelNetDataLoader import ModelNetDataLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
    parser.add_argument('--model', default='pointnet_cls', help='model name [default: pointnet_cls]')
    parser.add_argument('--num_category', default=40, type=int, choices=[10, 40],  help='training on ModelNet10/40')
    parser.add_argument('--epoch', default=400, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--num_point', type=int, default=2048, help='Point Number')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    return parser.parse_args()


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True


def test(model, loader, loss_fn, num_class=40):
    losses = []
    model = model.eval()

    for j, (points, target) in tqdm(enumerate(loader), total=len(loader)):

        if not args.use_cpu:
            points, target = points.cuda(), target.cuda()

        points = points.transpose(2, 1)
        pred, _ = model(points)
        losses.append(loss_fn(pred, points.transpose(2, 1))[0].item())

    #     for cat in np.unique(target.cpu()):
    #         classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
    #         class_acc[cat, 0] += classacc.item() / float(points[target == cat].size()[0])
    #         class_acc[cat, 1] += 1

    #     correct = pred_choice.eq(target.long().data).cpu().sum()
    #     mean_correct.append(correct.item() / float(points.size()[0]))

    # class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    # class_acc = np.mean(class_acc[:, 2])
    # instance_acc = np.mean(mean_correct)

    return np.mean(losses)



def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('reconstruction')
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    wandb.init(
        project="shapenet_autoencoder",
        # track hyperparameters and run metadata
        config=args
    )

    '''DATA LOADING'''
    log_string('Load dataset ...')
    data_path = '../shapenet_all_planes'# 'data/modelnet40_normal_resampled/'

    train_dataset = ModelNetDataLoader(root=data_path, args=args, split='train', process_data=False)
    test_dataset = ModelNetDataLoader(root=data_path, args=args, split='test', process_data=False)
    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=10, drop_last=True)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)

    '''MODEL LOADING'''
    num_class = args.num_category
    model = importlib.import_module(args.model)
    shutil.copy('./models/%s.py' % args.model, str(exp_dir))
    shutil.copy('models/pointnet2_utils.py', str(exp_dir))
    shutil.copy('./train_classification.py', str(exp_dir))

    classifier = model.get_model(args.num_point, num_class, normal_channel=args.use_normals)
    # criterion = model.get_loss()
    criterion = pytorch3d.loss.chamfer_distance
    classifier.apply(inplace_relu)

    if not args.use_cpu:
        classifier = classifier.cuda()
        # criterion = criterion.cuda()

    try:
        checkpoint = torch.load(str(exp_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    global_epoch = 0
    global_step = 0
    lowest_recon_loss = float('inf')

    '''TRANING'''
    logger.info('Start training...')
    for epoch in range(start_epoch, args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        recon_losses = []
        classifier = classifier.train()

        scheduler.step()
        for batch_id, (points, target) in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
            optimizer.zero_grad()

            points = points.data.numpy()
            points = provider.random_point_dropout(points)
            points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            points = torch.Tensor(points)
            points = points.transpose(2, 1) # [B, N_pts, 3] -> [B, 3, N_pts]

            if not args.use_cpu:
                points, target = points.cuda(), target.cuda()

            pred, latent_feat = classifier(points)
            loss = criterion(pred, points.transpose(2, 1))[0]
            recon_losses.append(loss.item())
            loss.backward()
            optimizer.step()
            global_step += 1

        train_recon_loss = np.mean(recon_losses)
        log_string('Train recon loss: %f' % train_recon_loss)

        with torch.no_grad():
            val_recon_loss = test(classifier.eval(), testDataLoader, criterion, num_class=num_class)

            if (val_recon_loss < lowest_recon_loss):
                lowest_recon_loss = val_recon_loss
                best_epoch = epoch + 1

                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': best_epoch,
                    'val_recon_loss': val_recon_loss,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)

            log_string('Val recon loss: %f' % (val_recon_loss))
            log_string('Lowest loss: %f' % (lowest_recon_loss))

            global_epoch += 1
        wandb.log({"train_recon_loss": train_recon_loss, "val_recon_loss": val_recon_loss})

    logger.info('End of training...')


if __name__ == '__main__':
    args = parse_args()
    main(args)
