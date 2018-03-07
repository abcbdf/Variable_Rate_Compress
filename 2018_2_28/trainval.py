import numpy as np
import torch
from torch import nn
import time
import os
from collections import OrderedDict

from torch.autograd import Variable
from matplotlib import pyplot as plt
import logging

class Trainer():
    def __init__(self, model, loss, train_loader, val_loader, optimizer, args, scheme, save_dir):
        self.model = model
        self.loss = loss
        self.valloss = {'L1': nn.L1Loss(), 'L2': nn.MSELoss()}
        self.valloss = OrderedDict(sorted(self.valloss.items()))

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.args = args
        self.scheme = scheme
        self.save_dir = save_dir
        self.logfile = os.path.join(self.save_dir,'log')
        with open(self.logfile,'a') as f:
            f.write('\n -------------------------')
            f.write(str(args))
    def get_lr(self, epoch):
        if self.args.lr is None:
            nodes = np.sort(np.array([n for n in self.scheme.keys()]))
            id = np.sum(nodes<=epoch)-1
            return self.scheme[nodes[id]]
        else:
            return self.args.lr
        
    def train(self, epoch):
        logging.info("train")
        start_time = time.time()
        lr = self.get_lr(epoch)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        self.model.train()
        loss_hist = []
        loss_sep_hist = []
        for iter, batch in enumerate(self.train_loader):
            logging.debug("iter:" + str(iter))
            logging.debug("batch:" + str(batch))
            batch = Variable(batch).cuda()
            rec = self.model(batch)
            l_sep, l = self.loss(rec, batch)
            self.optimizer.zero_grad()
            l.backward()
            self.optimizer.step()
            loss_hist.append(l.data.cpu().numpy())
            #print(loss_hist[-1])
            l_sep = [li.data.cpu().numpy() for li in l_sep]
            loss_sep_hist.append(l_sep)
            logging.debug("for end")
            
            if self.args.debug and iter ==4:
                break
        mean_loss = np.mean(loss_hist)
        mean_loss_sep = np.mean(loss_sep_hist, 0)
        dt = time.time()-start_time
        info = 'Train Epoch %d, time %.1f, loss %.5f' %(epoch, dt, mean_loss)
        for id, li in enumerate(mean_loss_sep):
            info = info + ', lossid %d: %.5f' %(id, li)
        info = info+' lr %.5f'%lr
        print(info)
        with open(self.logfile, 'a') as f:
            f.write(info+'\n')
        return mean_loss


    def val(self, epoch):
        start_time = time.time()
        self.model.eval()
        loss_hist = []
        loss_sep_hist = []
        self.valloss_hist = {k:[] for k in self.valloss.keys()}
        
        for iter, batch in enumerate(self.val_loader):
            batch = Variable(batch).cuda()
            rec = self.model(batch)
            l_sep, l = self.loss( rec, batch)
            loss_hist.append(l.data.cpu().numpy())
            l_sep = [li.data.cpu().numpy() for li in l_sep]
            loss_sep_hist.append(l_sep)
            
            rec = torch.clamp(rec,-1,1)
            for key, lossfun in self.valloss.items():
                l = lossfun(rec, batch)
                self.valloss_hist[key].append(l.data.cpu().numpy())

            if self.args.debug and iter ==4:
                break
        mean_loss = np.mean(loss_hist)
        mean_loss_sep = np.mean(loss_sep_hist, 0)
        dt = time.time()-start_time
        info = 'Val   Epoch %d, time %.1f, loss %.5f' %(epoch, dt, mean_loss)
        for id, li in enumerate(mean_loss_sep):
            info = info + ', lossid %d: %.5f' %(id, li)
        for key, hist in self.valloss_hist.items():
            info = info + ', Eval %s: %.5f' %(key, np.mean(hist))
        print(info)
        with open(self.logfile, 'a') as f:
            f.writelines(info+'\n')

        im1 = batch[0].data.cpu().numpy()/2 +0.5
        im2 = rec[0].data.cpu().numpy()/2 + 0.5
        im2 = np.clip(im2,0,1)
        im3 = np.concatenate([im1, im2], 1).transpose([1,2,0])
        plt.imsave(os.path.join(self.save_dir, '%03d.png' % epoch), im3)
        return mean_loss

    def save_model(self, epoch):
        try:
            state_dict = self.model.module.state_dict()
        except:
            state_dict = self.model.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].cpu()
        torch.save({'state_dict': state_dict, 'args':self.args, 'scheme':self.scheme},
            os.path.join(self.save_dir, '%03d.ckpt' % epoch))
