import torch
import numpy as np
import random
from dataset import get_train_val_test_loaders
# from model.target import Target
from train_common import *
from utils import config
import utils
import wandb
from model import Source
SEED = 0

import os

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

def count_parameters(model):
    """ count the number of learnable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(model, epoch, checkpoint_dir, stats):
    """Save a checkpoint file to `checkpoint_dir`."""
    state = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "stats": stats,
    }

    filename = os.path.join(checkpoint_dir, "epoch={}.checkpoint.pth.tar".format(epoch))
    torch.save(state, filename)





def main():



    tr_loader, va_loader, te_loader, _ =  get_train_val_test_loaders(
        task = "default",
        batch_size = config("net.batch_size")
    )

    filename = "trial"
    lr = 0.0001
    this_config = dict(csv_file=config("csv_file"), img_path=config("image_path"), learning_rate = lr, num_classes = 4, batchsize = 64)
    wandb.init(project = "cabin_train", name = filename, config = this_config)
    print('successfully loading!')

    model = Source()
    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    print("Number of float-valued parameters:", count_parameters(model))

    model, start_epoch, stats = restore_checkpoint(model, config("cnn.checkpoint"))

    axes = utils.make_training_plot()

    evaluate_epoch(
        axes, tr_loader, va_loader, te_loader, model, criterion, start_epoch, stats, multiclass=True
    )

        # initial val loss for early stopping
    prev_val_loss = stats[0][1]

    # TODO: define patience for early stopping
    patience = 5
    curr_patience = 0
    #

    # Loop over the entire dataset multiple times
    # for epoch in range(start_epoch, config('cnn.num_epochs')):
    epoch = start_epoch

    lowest_val_loss = 1
    train_auroc = 0
    test_auroc = 0
    lowest_round = epoch
    while curr_patience < patience:
        # Train model
        train_epoch(tr_loader, model, criterion, optimizer)

        # Evaluate model
        evaluate_epoch(
            axes, tr_loader, va_loader, te_loader, model, criterion, epoch + 1, stats, multiclass=True
        )

        # Save model parameters
        save_checkpoint(model, epoch + 1, config("net.checkpoint"), stats)

        # update early stopping parameters
        curr_patience, prev_val_loss = early_stopping(
            stats, curr_patience, prev_val_loss
        )


        epoch += 1
        if(prev_val_loss < lowest_val_loss):
            lowest_val_loss = prev_val_loss
            lowest_round = epoch
    print("Finished Training")
    # Save figure and keep plot open
    print("the lowest round: ", lowest_round)
    # utils.save_cnn_training_plot()
    # utils.save_cnn_other()
    utils.hold_training_plot()


if __name__ == "__main__":
    main()

