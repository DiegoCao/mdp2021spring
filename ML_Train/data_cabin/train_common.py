

from utils import config
import numpy as np
import itertools
import os
import torch
from torch.nn.functional import softmax
from sklearn import metrics
import utils
import wandb


def count_parameters(model):
    """Count number of learnable parameters."""
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


def check_for_augmented_data(data_dir):
    """Ask to use augmented data if `augmented_dogs.csv` exists in the data directory."""
    if "augmented_dogs.csv" in os.listdir(data_dir):
        print("Augmented data found, would you like to use it? y/n")
        print(">> ", end="")
        rep = str(input())
        return rep == "y"
    return False

def restore_checkpoint(model, checkpoint_dir, cuda=False, force=False, pretrain=False):
    """Restore model from checkpoint if it exists.

    Returns the model and the current epoch.
    """
    try:
        cp_files = [
            file_
            for file_ in os.listdir(checkpoint_dir)
            if file_.startswith("epoch=") and file_.endswith(".checkpoint.pth.tar")
        ]
    except FileNotFoundError:
        cp_files = None
        os.makedirs(checkpoint_dir)
    if not cp_files:
        print("No saved model parameters found")
        if force:
            raise Exception("Checkpoint not found")
        else:
            return model, 0, []

    # Find latest epoch
    for i in itertools.count(1):
        if "epoch={}.checkpoint.pth.tar".format(i) in cp_files:
            epoch = i
        else:
            break

    if not force:
        print(
            "Which epoch to load from? Choose in range [0, {}].".format(epoch),
            "Enter 0 to train from scratch.",
        )
        print(">> ", end="")
        inp_epoch = int(input())
        if inp_epoch not in range(epoch + 1):
            raise Exception("Invalid epoch number")
        if inp_epoch == 0:
            print("Checkpoint not loaded")
            clear_checkpoint(checkpoint_dir)
            return model, 0, []
    else:
        print("Which epoch to load from? Choose in range [1, {}].".format(epoch))
        inp_epoch = int(input())
        if inp_epoch not in range(1, epoch + 1):
            raise Exception("Invalid epoch number")

    filename = os.path.join(
        checkpoint_dir, "epoch={}.checkpoint.pth.tar".format(inp_epoch)
    )

    print("Loading from checkpoint {}?".format(filename))

    if cuda:
        checkpoint = torch.load(filename)
    else:
        # Load GPU model on CPU
        checkpoint = torch.load(filename, map_location=lambda storage, loc: storage)

    try:
        start_epoch = checkpoint["epoch"]
        stats = checkpoint["stats"]
        if pretrain:
            model.load_state_dict(checkpoint["state_dict"], strict=False)
        else:
            model.load_state_dict(checkpoint["state_dict"])
        print(
            "=> Successfully restored checkpoint (trained for {} epochs)".format(
                checkpoint["epoch"]
            )
        )
    except:
        print("=> Checkpoint not successfully restored")
        raise

    return model, inp_epoch, stats


def clear_checkpoint(checkpoint_dir):
    """Remove checkpoints in `checkpoint_dir`."""
    filelist = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth.tar")]
    for f in filelist:
        os.remove(os.path.join(checkpoint_dir, f))

    print("Checkpoint successfully removed")


def early_stopping(stats, curr_patience, prev_val_loss):
    """Calculate new patience and validation loss.

    Increment curr_patience by one if new loss is not less than prev_val_loss
    Otherwise, update prev_val_loss with the current val loss

    Returns: new values of curr_patience and prev_val_loss
    """
    # TODO implement early stopping

    #
    new_loss = stats[-1][1]
    if new_loss < prev_val_loss:
        prev_val_loss = new_loss
        curr_patience = 0
    else:
        curr_patience += 1



    return curr_patience, prev_val_loss


def evaluate_epoch(
    axes,
    tr_loader,
    val_loader,
    te_loader,
    model,
    criterion,
    epoch,
    stats,
    prolist,
    include_test=False,
    update_plot=True,
    multiclass=False,
    probabimode = False
):
    """Evaluate the `model` on the train and validation set."""

    def _get_metrics(loader):
        y_true, y_pred, y_score = [], [], []
        correct, total = 0, 0
        running_loss = []
        oneroundlis = []
        mseloss = []
        mselossfunc = torch.nn.MSELoss()
        disagreeloss = []
        for X, y in loader:
            with torch.no_grad():
                output = model(X)
                predicted = predictions(output.data)
                # print('the predicted and the true: ', predicted, ' ', y)
                tmp = softmax(output.data, dim=1)
                oneroundlis.append((output, y))
                mseloss.append(mselossfunc(tmp, y).item())

                for idx, y0 in enumerate(y):
                    if np.count_nonzero(y0) > 1:
                        print(y0)
                        disagreeloss.append(mselossfunc(y0, tmp[idx]))
                        

                y = np.argmax(y, axis=1)
                y_true.append(y)
                y_pred.append(predicted)
                if not multiclass:
                    y_score.append(softmax(output.data, dim=1)[:, 1])
                else:
                    y_score.append(softmax(output.data, dim=1))
                total += y.size(0)
                correct += (predicted == y).sum().item()

                running_loss.append(criterion(output, y).item())

        prolist.append(oneroundlis) 
        y_true = torch.cat(y_true)
        y_pred = torch.cat(y_pred)
        y_score = torch.cat(y_score)
        # print(y_true)
        loss = np.mean(mseloss)
        acc = correct / total
        if not multiclass:
            auroc = metrics.roc_auc_score(y_true, y_score)
        else:
            auroc = metrics.roc_auc_score(y_true, y_score, multi_class="ovo",labels=[0, 1, 2, 3])
        disagreeval = np.mean(disagreeloss)
        return acc, loss, auroc, disagreeval

    train_acc, train_loss, train_auc, train_dis = _get_metrics(tr_loader)
    val_acc, val_loss, val_auc, val_dis = _get_metrics(val_loader)
    test_acc, test_loss, test_auc, test_dis = _get_metrics(te_loader)

    wandb.log({"train_acc":train_acc, "train_loss": train_loss, "train_auc": train_auc, "train_dis": train_dis,\
                "val_acc":val_acc, "val_loss": val_loss, "val_auc": val_auc,  "val_dis": val_dis, \
                "test_acc":test_acc, "test_loss": test_loss, "test_auc": test_auc, "test_dis": test_dis
                }) 

    stats_at_epoch = [
        val_acc,
        val_loss,
        val_auc,
        train_acc,
        train_loss,
        train_auc,
        test_acc,
        test_loss,
        test_acc,
    ]
    if include_test:
        stats_at_epoch += list(_get_metrics(te_loader))

    stats.append(stats_at_epoch)
    utils.log_training(epoch, stats)
    if update_plot:
        utils.update_training_plot(axes, epoch, stats)

import time

def train_epoch(data_loader, model, criterion, optimizer, dupmode = False):
    """Train the `model` for one epoch of data from `data_loader`.

    Use `optimizer` to optimize the specified `criterion`
    """
    print('the length of the data loader is: ', len(data_loader))
    if dupmode == False:
        length = len(data_loader)
        np.random.seed(int(time.time()))
        idxlis = np.random.choice(length, int(length/5))
        
        for i, (X, y) in enumerate(data_loader):
            
            # TODO implement training steps
            optimizer.zero_grad()
            outputs = model(X)
            y = np.argmax(y, axis = 1)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
    else:
        for i, (X, y) in enumerate(data_loader):
            # TODO implement training steps
            optimizer.zero_grad()
            outputs = model(X)
            # y is the label
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

    
        


def train_epoch_pro(data_loader, model, criterion, optimizer, dupmode = False):
    """Train the `model` for one epoch of data from `data_loader`.

    Use `optimizer` to optimize the specified `criterion`
    """
    print('the length of the data loader is: ', len(data_loader))
    for i, (X, y) in enumerate(data_loader):

        # TODO implement training steps
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs.float(), y.float())
        loss.backward()
        optimizer.step()

    
def evaluate_epoch_pro(
    axes,
    tr_loader,
    val_loader,
    te_loader,
    model,
    criterion,
    epoch,
    stats,
    predictlog,
    include_test=False,
    update_plot=True,
    multiclass=False,
    probabimode = False
):
    """Evaluate the `model` on the train and validation set."""

    def _get_metrics(loader):
        y_true, y_pred, y_score = [], [], []
        correct, total = 0, 0
        running_loss = []
        oneroundlis = []
        disagreeloss = []
        mselossfunc = torch.nn.MSELoss()
        for X, y in loader:
            with torch.no_grad():
                output = model(X)
            
                # predicted = predictions(output.data)
                # print('the predicted and the true: ', predicted, ' ', y)
                y_true.append(y)

                for idx, y0 in enumerate(y):
                    if np.count_nonzero(y0) > 1:
                        print(y0)
                        disagreeloss.append(mselossfunc(y0, output[idx]).item())
                        
                # if np.count_nonzero(y) > len(y):
                #     print("FOUND!!!!!!!")
                #     print(y)
                # y_pred.append(predicted)
                # oneroundlis.append((predicted, y))
                oneroundlis.append((output, y))
                # if not multiclass:
                #     y_score.append(softmax(output.data, dim=1)[:, 1])
                # else:
                #     y_score.append(softmax(output.data, dim=1))
                total += y.size(0)
                # correct += (predicted == y).sum().item()
                # print(output)
                # print(output.shape)
                # print(type(output))
                # print(type(y))
                # print(y)
                # print(y.shape)
                running_loss.append(criterion(output, y).item())
        
        predictlog.append(oneroundlis)
        # y_true = torch.cat(y_true)
        # y_pred = torch.cat(y_pred)
        # # y_score = torch.cat(y_score)
        # # print(y_true)
        loss = np.mean(running_loss)
        # acc = correct / total
        # if not multiclass:
        #     auroc = metrics.roc_auc_score(y_true, y_score)
        # else:
        #     auroc = metrics.roc_auc_score(y_true, y_score, multi_class="ovo",labels=[0, 1, 2, 3])
        return loss, np.mean(disagreeloss)

    # train_acc, train_loss, train_auc = _get_metrics(tr_loader)
    # val_acc, val_loss, val_auc = _get_metrics(val_loader)
    # test_acc, test_loss, test_auc = _get_metrics(te_loader)
    
    train_loss , train_dis= _get_metrics(tr_loader)
    val_loss, val_dis = _get_metrics(val_loader)
    test_loss, test_dis = _get_metrics(te_loader)
    

    wandb.log({ "train_loss": train_loss, "train_dis": train_dis, \
                 "val_loss": val_loss, "val_dis": val_dis, \
                "test_loss": test_loss, "test_dis": test_dis,
                }) 

    stats_at_epoch = [
        val_loss,
        train_loss,
        test_loss,

    ]
    if include_test:
        stats_at_epoch += list(_get_metrics(te_loader))

    stats.append(stats_at_epoch)
    utils.log_training(epoch, stats)
    # if update_plot:
    #     utils.update_training_plot(axes, epoch, stats)




def predictions(logits):
    """Determine predicted class index given logits.
    Returns:
            the predicted class output as a PyTorch Tensor
    """
        # TODO implement predictions
    
    # print(logits)
    pred = torch.zeros(len(logits))
    for i in range(len(logits)):
        pred[i] = torch.argmax(logits[i]).item()
    # s, pred = torch.max(logits, 1)
    return pred