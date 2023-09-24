import os
import numpy as np
import time
import argparse
import yaml
import torch
import torch.nn as nn
import datetime
import torch.optim as optim

from torch.utils.data import DataLoader, Dataset
from lightning_fabric.utilities.seed import seed_everything

from basenet.model import Model_factory
from data_loader import AppleDataset, ImageTransform
from loss import SWM_FPEM_Loss
from utils.lr_scheduler import WarmupPolyLR
from utils.augmentations import Transform
from torch.utils.tensorboard import SummaryWriter

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
    parser.add_argument('--print_freq', default=64, type=int, help='interval of showing training conditions')
    parser.add_argument('--train_iter', default=0, type=int, help='number of total iterations for training')
    parser.add_argument('--curr_iter', default=0, type=int, help='current iteration')
    parser.add_argument('--alpha', type=float, default=10, help='weight for positive loss, default=10')
    parser.add_argument('--amp', action='store_true', help='half precision')
    parser.add_argument('--save_path', type=str, default='./weight', help='Model save path')
    parser.add_argument("--trained", default=None, help='Path to pre-trained model')
    parser.add_argument("--store_path", default=None, help='Path to save trained model')
    
    args = parser.parse_args()
    
    return args
    
def main():
    args = get_args()
    
    if args.store_path is None:
        args.store_path = './store'
    
    if not os.path.isdir(args.store_path):
        os.mkdir(args.store_path)
    
    if type(args.input_size) == int:
        args.input_size = (args.input_size, args.input_size)
    
    # Set cuda device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    NUM_CLASSES = {'apple_2': 2, 'apple_6': 6}
    num_classes = NUM_CLASSES[args.dataset]
    
    # Check and load pretrained model
    if args.trained is not None and args.trained != 'None':
        model.load_state_dict(torch.load(args.trained, map_location=device))
        print("Restored from {}".format(args.trained))
    else:
        print('No trained model specifying ...')
    
    model = Model_factory(args.backbone, num_classes).to(device)
    
    transform_train = Transform(is_train=True, size=args.input_size)
    transform_valid = Transform(is_train=True, size=args.input_size)

    # Training data loader
    train_dataset = AppleDataset('train', args.root, 
                                 args.input_size, transform=transform_train)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.workers, pin_memory=True)
    
    # Validation data loader
    valid_dataset = AppleDataset('valid', args.root,
                                args.input_size, transform=transform_valid)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)
    
    # Scale learning rate based on global batch size
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # define loss function (criterion) and optimizer
    criterion = SWM_FPEM_Loss(num_classes=num_classes, alpha=args.alpha, neg_pos_ratio=0.3)
    
    # Set up summary writer
    current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    log_dir = f'{args.store_path}/logs/{current_time}'
    summary_writer = SummaryWriter(log_dir)
    
    hparams = vars(args)
    with open(os.path.join(log_dir, 'hparams.yaml'), 'w') as f:
        yaml.dump(hparams, f)
    
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
        print('{:-^50s}'.format(''))
        # Training phase
        train(train_loader=train_loader, model=model, criterion=criterion, optimizer=optimizer,
              scheduler=scheduler, summary_writer=summary_writer , device=device, start=start, epoch=epoch, args=args, log_dir=log_dir)
        
        # Validation phase
        val_loss, val_dist = validate(valid_loader=valid_loader, model=model, criterion=criterion,
                                      device=device, epoch=epoch, summary_writer=summary_writer, log_dir=log_dir)
        
        if best_loss >= val_loss:
            best_loss = val_loss
            if best_loss_checkpoint is not None:
                # Delete the old best loss checkpoint
                os.remove(best_loss_checkpoint)
            
            best_loss_checkpoint = save_checkpoint(model, optimizer, epoch, f'best_loss_epoch{epoch}_{val_loss:0.4f}', log_dir)

        if best_dist >= val_dist:
            best_dist = val_dist
            if best_dist_checkpoint is not None:
                # Delete the old best distance checkpoint
                os.remove(best_dist_checkpoint)
            best_dist_checkpoint = save_checkpoint(model, optimizer, epoch, f'best_dist_epoch{epoch}_{val_dist:0.4f}', log_dir)

    summary_writer.close()
    
        
def train(train_loader, model, criterion, optimizer, scheduler, summary_writer, device, start, epoch, args, log_dir):
    batch_time = AverageMeter()
    losses = AverageMeter()
    
    model.train()
    temp_loss = []
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
        
        losses.update(loss.item())
        temp_loss.append(loss.item())
        batch_time.update(time.time() - end)
        end = time.time()
        
        if args.curr_iter % args.print_freq == 0:
            train_log = "Epoch: [%d/%d][%d/%d] " % (epoch, args.epochs, args.curr_iter, args.train_iter)
            train_log += "({0:.1f}%, {1:.1f} min) | ".format(args.curr_iter/args.train_iter*100, (end-start) / 60)
            train_log += "Time %.1f ms | Left %.1f min | " % (batch_time.avg * 1000, (args.train_iter - args.curr_iter) * batch_time.avg / 60)
            train_log += "Loss %.6f " % (losses.avg)
            print(train_log)

            # Append the log to a text file
            with open(f'{log_dir}/train_log.txt', 'a') as log_file:
                log_file.write(train_log + '\n')
        
        summary_writer.add_scalars('Loss', {'training': losses.avg}, epoch)

def validate(valid_loader, model, criterion, device, epoch, summary_writer, log_dir):
    losses = AverageMeter()
    distances = AverageMeter()
    
    # evaluation mode, no gradient calculation
    model.eval()
    temp_loss = []
    temp_dist = []
    end = time.time()
    
    for x, y, w, s in valid_loader:
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
        temp_loss.append(loss.item())
        temp_dist.append(dist.item())
        
    valid_log = "\n============== validation ==============\n"
    valid_log += "valid time : %.1f s | " % (time.time() - end)
    valid_log += "valid loss : %.6f | " % (losses.avg)
    valid_log += "valid dist : %.6f \n" % (distances.avg)
    print(valid_log)
    
    summary_writer.add_scalars('Loss', {'validation': losses.avg}, epoch)
    summary_writer.add_scalars('Distance', {'validation': distances.avg}, epoch)
    
    # Append the log to a text file
    with open(f'{log_dir}/train_log.txt', 'a') as log_file:
        log_file.write(valid_log + '\n')
    
    return losses.avg, distances.avg

def save_checkpoint(model, optimizer, epoch, name, save_path):
    state_dict = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch
    }
    model_path = os.path.join(save_path, f'{name}.pt')
    torch.save(state_dict, model_path)
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
