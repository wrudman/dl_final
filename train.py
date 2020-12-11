import argparse
import torchvision
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.utils.data as data
from ecg_dataloaders import ar_binclass_loader, sv_ve_nm_loader
from ecg_model import ECGResnet18

def load_checkpoint(model, name, prefix='models/'):
    checkpoint = torch.load(prefix+name)
    model.load_state_dict(checkpoint['model'])
    return model

def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device

def save_checkpoint(state, is_best, filename, prefix='models/'):
    torch.save(state, prefix + filename)


def train(net, dl, valdl, opt):
    net.train()
    device = get_device()
    epochs=opt.epochs
    model_path = opt.model_path
    criterion = nn.CrossEntropyLoss()
    lr = opt.lr
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    train_loss=[]
    best_loss=1000
    net = net.to(device)
    for epoch in range(epochs):
        running_loss = 0.0
        for batch in dl:
            optimizer.zero_grad()
            specs, labels = batch #the spec graph and its label (arrythmia vs normal)
            specs, labels = specs.to(device), labels.to(device)
            preds = net(specs)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            running_loss+=loss.item()
        loss = running_loss/len(dl)
        train_loss.append(loss)
        if loss < best_loss:
            save_checkpoint({'epoch':epoch, 'model': net.state_dict(), 'train_loss':train_loss}, True,model_path)
        print("Epoch {} of {} train loss: {}".format(epoch+1, epochs, loss))
        num_correct, total = validate(net, valdl, opt)
        print("Validation Accuracy: {}/{}={}".format(num_correct, total, num_correct/float(total)))
        net.train()

            

def validate(net, dl, opt):
    net.eval()
    device = get_device()
    total_correct = 0.0
    criterion = nn.CrossEntropyLoss()
    test_loss = 0.0
    total_seen = 0.0
    label_acc = {} #per class accuracy
    for batch in dl:
        specs, labels = batch #the spec graph and its label (arrythmia vs normal)
        specs, labels = specs.to(device), labels.to(device)
        preds = net(specs)
        predmax = torch.argmax(preds, dim=1)
        #print(predmax, labels) 
        total_correct+=torch.sum(predmax == labels)
        for pred, label in zip(predmax, labels):
            label = int(label)
            pred = int(pred)
            if label not in label_acc:
                label_acc[label] = {'num':0.0, 'denom':0.0}
            label_acc[label]['denom']+=1.0
            if pred == label:
                label_acc[label]['num']+=1.0
            label_acc[label]['acc'] = label_acc[label]['num']/label_acc[label]['denom']
            
        total_seen+=len(labels)
        loss = criterion(preds, labels)
        test_loss+=loss.item()
    print("Accs")
    for key in label_acc:
        print("{}: {}/{} = {}".format(key, label_acc[key]['num'], label_acc[key]['denom'], label_acc[key]['acc']))
    print("Validation loss: {}".format(test_loss/float(len(dl))))
    return total_correct, total_seen

def create_dl(datadir, batch_size, shuffle=True):
    dl = sv_ve_nm_loader(datadir)
    loader = data.DataLoader(dataset=dl,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=4)
    return loader



def main(opt):
    device = get_device()
    dl = create_dl(opt.train_path, opt.batch_size)
    val_dl = create_dl(opt.val_path, opt.batch_size)
    net = ECGResnet18(opt)
    train(net, dl, val_dl, opt)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--train_path", default="/gpfs/data/ceickhof/ecg_data/data/ar_train", type=str) 
    parser.add_argument("--val_path", default="/gpfs/data/ceickhof/ecg_data/data/ar_val", type=str)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--hsz1", default=128, type=int)
    parser.add_argument("--outsz", default=2, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--pretrained", action='store_true')
    parser.add_argument("--model_path", type=str)
    opt = parser.parse_args()

    main(opt)
