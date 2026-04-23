from dataset import courtDataset
import torch
import torch.nn as nn
from base_trainer import train
from base_validator import val
import os
from torch.utils.tensorboard import SummaryWriter
from tracknet import BallTrackerNet
import argparse
from utils import choose_device
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = REPO_ROOT / 'calib_model_data' / 'courtside_data'
DEFAULT_EXP_ROOT = Path(__file__).resolve().parent / 'exps'

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=2, help='batch size')
    parser.add_argument('--exp_id', type=str, default='default', help='path to saving results')
    parser.add_argument('--num_epochs', type=int, default=500, help='total training epochs')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--val_intervals', type=int, default=5, help='number of epochs to run validation')
    parser.add_argument('--steps_per_epoch', type=int, default=1000, help='number of steps per one epoch')
    parser.add_argument('--data_root', type=str, default=str(DEFAULT_DATA_ROOT), help='dataset root containing images, data_train.json, and data_val.json')
    parser.add_argument('--exp_root', type=str, default=str(DEFAULT_EXP_ROOT), help='directory for experiment outputs')
    parser.add_argument('--num_workers', type=int, default=1, help='DataLoader worker count')
    parser.add_argument('--device', type=str, default='auto', help='auto, cpu, cuda, mps, or a torch device string')
    args = parser.parse_args()
    
    device = choose_device(args.device)

    train_dataset = courtDataset('train', data_root=args.data_root)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == 'cuda'
    )
    
    val_dataset = courtDataset('val', data_root=args.data_root)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == 'cuda'
    )

    model = BallTrackerNet(out_channels=15)
    model = model.to(device)

    exps_path = os.path.join(args.exp_root, args.exp_id)
    tb_path = os.path.join(exps_path, 'plots')
    if not os.path.exists(tb_path):
        os.makedirs(tb_path)
    log_writer = SummaryWriter(tb_path)
    model_last_path = os.path.join(exps_path, 'model_last.pt')
    model_best_path = os.path.join(exps_path, 'model_best.pt')

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), args.lr, betas=(0.9, 0.999), weight_decay=0)

    val_best_accuracy = 0
    for epoch in range(args.num_epochs):
        train_loss = train(model, train_loader, optimizer, criterion, device, epoch, args.steps_per_epoch)
        log_writer.add_scalar('Train/training_loss', train_loss, epoch)

        if (epoch > 0) & (epoch % args.val_intervals == 0):
            val_loss, tp, fp, fn, tn, precision, accuracy = val(model, val_loader, criterion, device, epoch)
            print('val loss = {}'.format(val_loss))
            log_writer.add_scalar('Val/loss', val_loss, epoch)
            log_writer.add_scalar('Val/tp', tp, epoch)
            log_writer.add_scalar('Val/fp', fp, epoch)
            log_writer.add_scalar('Val/fn', fn, epoch)
            log_writer.add_scalar('Val/tn', tn, epoch)
            log_writer.add_scalar('Val/precision', precision, epoch)
            log_writer.add_scalar('Val/accuracy', accuracy, epoch)
            if accuracy > val_best_accuracy:
                val_best_accuracy = accuracy
                torch.save(model.state_dict(), model_best_path)     
            torch.save(model.state_dict(), model_last_path)
