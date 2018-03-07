import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")
import torch
from torch.utils.data import DataLoader
from torch import optim, nn

from trainval import Trainer
import os
from layers import Clip_loss, l1_loss, Feat_loss, get_loss, resnet18
import time
from data import Dataset
from model import get_model
import argparse
import logging

parser = argparse.ArgumentParser(description='Autoencoder')
parser.add_argument('--model', '-m', metavar='MODEL', default='model',
                    help='model')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N', help='mini-batch size (default: 32)')
parser.add_argument('--lr', '--learning-rate', default=None, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--weight-decay', '--wd', default=0.001, type=float,
                    metavar='W', help='weight decay (default: 0)')
parser.add_argument('--save-freq', default='5', type=int, metavar='S',
                    help='save frequency')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--loadw', default='', type=str, metavar='PATH',
                    help='load parameters (default: none)')
parser.add_argument('--save-dir', default='', type=str, metavar='SAVE',
                    help='directory to save checkpoint (default: none)')
parser.add_argument('--debug', action = 'store_true',
                    help='debug mode')
parser.add_argument('--optimizer', default='adam', type=str, metavar='O',
                    help='optimizer')
parser.add_argument('--loss', default='l2_im_naive', type=str, metavar='O',
                    help='loss function')
parser.add_argument('--gpu', default='0', type=str, metavar='O',
                    help='id of gpu used')
parser.add_argument('--test', default ='', type=str, metavar='O', help='which picture to reconstruct')

logging.basicConfig(level=logging.INFO,
                format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                datefmt='%a, %d %b %Y %H:%M:%S',
                filename='./history/' + time.strftime('%Y_%m_%d_%H_%M_%S',time.localtime(time.time())) + '_compress.log',
                filemode='w')
logging.info('Start:')




def main(mode = 'run', args = None):        
    assert mode in ['run', 'test']
    os.environ['CUDA_VISIBLE_DEVICES']=args.gpu

    dataset = Dataset('/home/adv/compress-src/professional_train/','train')
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.workers)
    if args.test != '':
        dataset = Dataset('/home/adv/compress-src/professional_valid/' + args.test,'test')
        val_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.workers)
    else:
        dataset = Dataset('/home/adv/compress-src/professional_valid/','val')
        val_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.workers)
    
    
    ##############################################
    #             start_epoch and save_dir
    ##############################################
    start_epoch = args.start_epoch
    if args.resume:
        start_epoch = int(os.path.basename(args.resume).split('.')[0])
    start_epoch += 1
            
    if args.save_dir != '':
        save_dir = os.path.join('results', args.save_dir)
    elif args.resume:
        save_dir = os.path.dirname(args.resume)
    else:
        exp_id = time.strftime('%Y%m%d-%H%M%S', time.localtime())
        save_dir = os.path.join('results', args.model+'_autosave')
    if args.debug:
        save_dir = os.path.join('results', 'tmp')

    print(save_dir)
        
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')


    ##############################################
    #             model and loss
    ##############################################
    model = get_model(args.model)
    model = model.cuda()
    

    
    if args.resume or args.loadw:
        if args.resume:
            weight_path = args.resume
        if args.loadw:
            weight_path = args.loadw
        checkpoint = torch.load(weight_path)
        model.load_state_dict(checkpoint['state_dict'])
  
    if not args.debug:
        model = nn.DataParallel(model)
        
    feat_model = resnet18(pretrained = True).cuda()
    feat_model = nn.DataParallel(feat_model)
        
    loss = get_loss(args.loss,feat_model)
    ##############################################
    #             training paras
    ##############################################
    N_epoch = args.epochs
    train_scheme = {0:0.001, 15:0.0001, 20: 0.00001}

    if isinstance(model, nn.DataParallel):
        params = model.module.parameters()
    else:
        params = model.parameters()
        
    if args.lr is None:
        init_lr = train_scheme[0]
    else:
        init_lr = args.lr
        
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(params,lr = init_lr, momentum = 0.9,weight_decay = args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(params, lr = init_lr, weight_decay = args.weight_decay)
    else:
        exit('Wrong optimizer')

    ##############################################
    #             main loop
    ##############################################
    
    trainer = Trainer(model, loss, train_loader, val_loader, optimizer, args, train_scheme, save_dir)

    if args.test != '':
        trainer.val(0)
    else:
        if mode == 'run':
            for epoch in range(start_epoch, N_epoch):
                trainer.train(epoch)
                trainer.val(epoch)
                if epoch % args.save_freq == 0:
                    trainer.save_model(epoch)
        else:
            return model, loss, train_loader, val_loader, optimizer, args, train_scheme, save_dir ,trainer
    

    
if __name__ == '__main__':
    args = parser.parse_args()
    main(mode = 'run', args = args)
