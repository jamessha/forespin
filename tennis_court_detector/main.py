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


def save_checkpoint(path, model, optimizer, epoch, val_best_accuracy, args, metrics=None):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_best_accuracy': val_best_accuracy,
        'args': vars(args),
    }
    if metrics is not None:
        checkpoint['metrics'] = metrics
    torch.save(checkpoint, path)


def resolve_resume_path(resume_arg, checkpoint_last_path):
    if resume_arg is None:
        return None
    if resume_arg == 'auto':
        return checkpoint_last_path
    return os.path.expanduser(resume_arg)


def load_training_state(resume_path, model, optimizer, device):
    checkpoint = torch.load(resume_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = int(checkpoint.get('epoch', -1)) + 1
        val_best_accuracy = float(checkpoint.get('val_best_accuracy', 0))
        return start_epoch, val_best_accuracy

    model.load_state_dict(checkpoint)
    return 0, 0


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
    parser.add_argument(
        '--resume',
        nargs='?',
        const='auto',
        help='Resume from a full checkpoint path. If passed without a value, uses checkpoint_last.pt in the selected experiment.',
    )
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
    checkpoint_last_path = os.path.join(exps_path, 'checkpoint_last.pt')
    checkpoint_best_path = os.path.join(exps_path, 'checkpoint_best.pt')

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), args.lr, betas=(0.9, 0.999), weight_decay=0)

    start_epoch = 0
    val_best_accuracy = 0
    resume_path = resolve_resume_path(args.resume, checkpoint_last_path)
    if resume_path is not None:
        if not os.path.exists(resume_path):
            raise FileNotFoundError(resume_path)
        start_epoch, val_best_accuracy = load_training_state(resume_path, model, optimizer, device)
        print('resumed training from {}, start_epoch = {}, val_best_accuracy = {}'.format(
            resume_path, start_epoch, val_best_accuracy
        ))

    if start_epoch >= args.num_epochs:
        print('checkpoint is already at or beyond requested num_epochs: start_epoch = {}, num_epochs = {}'.format(
            start_epoch, args.num_epochs
        ))

    for epoch in range(start_epoch, args.num_epochs):
        train_loss = train(model, train_loader, optimizer, criterion, device, epoch, args.steps_per_epoch)
        log_writer.add_scalar('Train/training_loss', train_loss, epoch)
        checkpoint_metrics = {'train_loss': float(train_loss)}

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
            checkpoint_metrics.update({
                'val_loss': float(val_loss),
                'tp': int(tp),
                'fp': int(fp),
                'fn': int(fn),
                'tn': int(tn),
                'precision': float(precision),
                'accuracy': float(accuracy),
            })
            if accuracy > val_best_accuracy:
                val_best_accuracy = accuracy
                torch.save(model.state_dict(), model_best_path)
                save_checkpoint(
                    checkpoint_best_path,
                    model,
                    optimizer,
                    epoch,
                    val_best_accuracy,
                    args,
                    checkpoint_metrics,
                )

        torch.save(model.state_dict(), model_last_path)
        save_checkpoint(
            checkpoint_last_path,
            model,
            optimizer,
            epoch,
            val_best_accuracy,
            args,
            checkpoint_metrics,
        )
