import torch
import numpy as np
import random
from dataset import get_train_val_test_loaders
# from model.target import Target
from train_common import *
from utils import config
import utils
from lenet import Lenet
SEED = 0

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


def main():


    tr_loader, va_loader, te_loader, _ =  get_train_val_test_loaders(
        task = "default",
        batch_size = config("net.batch_size")
    )
    print('successfully loading!')

    model = Lenet()
    criterion = torch.nn.CrossEntropyLoss()
    # learbubg 
    optimizer = torch.optim.Adam(Lenet.parameters(), lr = 0.001)
    print("Number of float-valued parameters:", count_parameters(model))

    model, start_epoch, stats = restore_checkpoint(model, config("cnn.checkpoint"))

    axes = utils.make_training_plot()

    evaluate_epoch(
        axes, tr_loader, va_loader, te_loader, model, criterion, start_epoch, stats
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
            axes, tr_loader, va_loader, te_loader, model, criterion, epoch + 1, stats
        )

        # Save model parameters
        save_checkpoint(model, epoch + 1, config("cnn.checkpoint"), stats)

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

