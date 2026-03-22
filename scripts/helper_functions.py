import torch
import torchaudio

#from __future__ import print_function, division
import os
import torch
import numpy as np
import os
import glob
import random

##################################################################################################
# First things first! Set a seed for reproducibility.
# https://www.cs.mcgill.ca/~ksinha4/practices_for_reproducibility/
def set_seed(seed):
    """Set seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def __get_device__() :
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    print('Device available is', device)
    return device


## Shuffle and pick a quarter of the data
def __shuffle_pick_quarter_data__ (x_f, y_f, x_s, y_s) :
    # Take only quarter dataset
    random.shuffle(x_f)
    random.shuffle(x_s)
    x_s_h = x_s[0:int(len(x_s)/4)]
    y_s_h = y_s[0:int(len(x_s)/4)]
    
    
    # Comment this for unblanced data
    # Stutter is less than fluent
    x_f_h = x_f[0:len(x_s_h)]
    y_f_h = y_f[0:len(x_s_h)]
    
    x_train = x_s_h + x_f_h
    y_train = y_s_h + y_f_h
    return x_train, y_train


def train(epoch):
  print('\nEpoch : %d'%epoch)
  
  model.train()

  running_loss=0
  correct=0
  total=0

  for data in train_loader:
    
    inputs,labels=data[0].to(device),data[1].to(device)
    # forward pass
    outputs=model(inputs)
    loss=criterion(outputs,labels)

    # backward and optimise
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    running_loss += loss.item()
    
    # Compute Training Accuracy
    _, predicted_labels = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += predicted_labels.eq(labels).sum().item()
      
  train_loss=running_loss/len(train_loader)
  
  accu=100.*correct/total
  # visualisation
  writer.add_scalar("Loss/train", loss, epoch)  
  writer.add_scalar("Accuracy/train", accu, epoch)  
  
  train_accu.append(accu)
  train_losses.append(train_loss)
  print('Train Loss: %.3f | Accuracy: %.3f'%(train_loss,accu))

def test(epoch):
  model.eval()

  running_loss=0
  correct=0
  total=0

  with torch.no_grad():
    for data in valid_loader:
      features,labels=data[0].to(device),data[1].to(device)
      
      outputs=model(features)

      _, predicted_valid = torch.max(outputs.data, 1)

      loss= criterion(outputs,labels)
      running_loss+=loss.item()
      total += labels.size(0)
      correct += predicted_valid.eq(labels).sum().item()
  
  test_loss=running_loss/len(valid_loader)
  accu=100.*correct/total
  
  # visualisation
  writer.add_scalar("Loss/test", loss, epoch)  
  writer.add_scalar("Accuracy/test", accu, epoch)  

  eval_losses.append(test_loss)
  eval_accu.append(accu)

  print('Validation Loss: %.3f | Accuracy: %.3f'%(test_loss,accu))  