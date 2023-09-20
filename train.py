import os
import numpy as np
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from lightning_fabric.utilities.seed import seed_everything

from basenet.model import Model_factory
from data_loader import AppleDataset, ImageTransform
from loss import SWM_FPEM_Loss
from utils.lr_scheduler import WarmupPolyLR

seed_everything(44)


def get_args():
    parser = argparse.ArgumentParser(description='Training Object Detection Module')
    parser.add_argument('--root', type=str, help='Root directory of dataset')
    parser.add_argument('--dataset', type=str, default='apple_2', help='Training dataset')
    parser.add_argument('--input_size', type=int, default=512, help='Input size')
    parser.add_argument('--workers', default=4, type=int, help='Number of workers')
    parser.add_argument('--batch_size', type=int, default=4, help='Training batch size')
    parser.add_argument('--backbone', type=str, default='hourglass104_MRCB_cascade', 
                        help='[hourglass104_MRCB_cascade, hourglass104_MRCB, hhrnet48, DLA_dcn, uesnet101_dcn]')
    parser.add_argument('--epochs', type=int, default=2, help='number of train epochs')
    parser.add_argument('--lr', type=float, default=2.5e-4, help='learning rate')
    parser.add_argument('--resume', default=None, type=str,  help='training restore')
    parser.add_argument('--print_freq', default=100, type=int, help='interval of showing training conditions')
    parser.add_argument('--train_iter', default=0, type=int, help='number of total iterations for training')
    parser.add_argument('--curr_iter', default=0, type=int, help='current iteration')
    parser.add_argument('--alpha', type=float, default=10, help='weight for positive loss, default=10')
    parser.add_argument('--amp', action='store_true', help='half precision')
    parser.add_argument('--save_path', type=str, default='./weight', help='Model save path')
    
    
    args = parser.parse_args()
    
    return args
    
def main():
    args = get_args()
    
    if type(args.input_size) == int:
        args.input_size = (args.input_size, args.input_size)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    NUM_CLASSES = {'apple_2': 2, 'apple_6': 6}
    num_classes = NUM_CLASSES[args.dataset]
    
    model = Model_factory(args.backbone, num_classes).to(device)
    
    # Scale learning rate based on global batch size
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # define loss function (criterion) and optimizer
    criterion = SWM_FPEM_Loss(num_classes=num_classes, alpha=args.alpha, neg_pos_ratio=0.3)
    
    transform_train = ImageTransform()
    transform_test = ImageTransform()
    
    train_dataset = AppleDataset('train', args.root, 
                                 args.input_size, transform=transform_train)
    test_dataset = AppleDataset('test', args.root,
                                args.input_size, transform=transform_test)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)
    
    """"Learning rate scheduler"""
    args.train_iter = len(train_loader) * args.epochs
    
    scheduler = WarmupPolyLR(
        optimizer=optimizer,
        max_iters=args.train_iter,
        warmup_iters=1000,
        power=0.9,
    )
    
    best_loss = 1e10
    best_dist = 1e10
    best_loss_checkpoint = None
    best_dist_checkpoint = None
    
    start = time.time()
    
    for epoch in range(0, args.epochs):
        
        train(train_loader=train_loader, model=model, criterion=criterion, optimizer=optimizer,
              scheduler=scheduler, device=device, start=start, epoch=epoch, args=args)
        
        val_loss, val_dist = test(test_loader=test_loader, model=model, criterion=criterion, device=device, epoch=epoch, args=args)
        
        if best_loss >= val_loss:
            best_loss = val_loss
            if best_loss_checkpoint is not None:
                # Delete the old best loss checkpoint
                os.remove(best_loss_checkpoint)
            
            best_loss_checkpoint = save_checkpoint(model, optimizer, epoch, f'best_loss_epoch{epoch}_{val_loss:0.4f}', args.save_path)

        if best_dist >= val_dist:
            best_dist = val_dist
            if best_dist_checkpoint is not None:
                # Delete the old best distance checkpoint
                os.remove(best_dist_checkpoint)
            best_dist_checkpoint = save_checkpoint(model, optimizer, epoch, f'best_dist_epoch{epoch}_{val_dist:0.4f}', args.save_path)
        
        
def train(train_loader, model, criterion, optimizer,
          scheduler, device, start, epoch, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    
    model.train()
    end = time.time()
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    
    for x, y, w, s in train_loader:
        args.curr_iter += 1
        
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        w = w.to(device, non_blocking=True)
        s = s.to(device, non_blocking=True)

        outs = model(x)
        with torch.cuda.amp.autocast(enabled=args.amp):
            if type(outs) == list:
                loss = 0
                for out in outs:
                    loss += criterion(y, out, w, s)
                    
                loss /= len(outs)
                    
                outs = outs[-1]

            else:
                loss = criterion(y, outs, w, s)
    
        # compute gradient and backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        # reduced_loss = reduce_tensor(loss.data, world_size)
        
        losses.update(loss.item())
        batch_time.update(time.time() - end)
        end = time.time()
        
        if args.curr_iter % args.print_freq == 0:
            train_log = "Epoch: [%d/%d][%d/%d] " % (epoch, args.epochs, args.curr_iter, args.train_iter)
            train_log += "({0:.1f}%, {1:.1f} min) | ".format(args.curr_iter/args.train_iter*100, (end-start) / 60)
            train_log += "Time %.1f ms | Left %.1f min | " % (batch_time.avg * 1000, (args.train_iter - args.curr_iter) * batch_time.avg / 60)
            train_log += "Loss %.6f " % (losses.avg)
            print(train_log)
            
            # Append the log to a text file
            with open('train_log.txt', 'a') as log_file:
                log_file.write(train_log + '\n')

def test(test_loader, model, criterion, device, epoch, args):
    losses = AverageMeter()
    distances = AverageMeter()
    
    # evaluation mode, no gradient calculation
    model.eval()
    
    end = time.time()
    
    for x, y, w, s in test_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        w = w.to(device, non_blocking=True)
        s = s.to(device, non_blocking=True)

        # compute output
        with torch.no_grad():
            outs = model(x)
            
            if type(outs) == list:
                outs = outs[-1]

            loss = criterion(y, outs, w, s)

        # measure accuracy and record loss
        dist = torch.sqrt((y - outs)**2).mean()
    
        losses.update(loss.item())
        distances.update(dist.item())
    
    valid_log = "\n============== validation ==============\n"
    valid_log += "valid time : %.1f s | " % (time.time() - end)
    valid_log += "valid loss : %.6f | " % (losses.avg)
    valid_log += "valid dist : %.6f \n" % (distances.avg)
    print(valid_log)
    
    
    # with open('test_log.txt', 'a') as log_file:
    #     log_file.write(valid_log + '\n')
        
    # Append the log to a text file
    with open('train_log.txt', 'a') as log_file:
        log_file.write(valid_log + '\n')
    
    return losses.avg, distances.avg

def save_checkpoint(model, optimizer, epoch, name, save_path):
    dict_state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch
    }
    model_path = os.path.join(save_path, f'{name}.pt')
    torch.save(dict_state, model_path)
    return model_path
    
class AverageMeter():
    """Computes and stores the avarage and current value"""
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val = 0  # the current (most recent) value
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):  # n: specify how many times the val is added (typically batch_size)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        


if __name__ == '__main__':
    main()
