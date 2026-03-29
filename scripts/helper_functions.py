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

## Shuffle the data
def __shuffle_data__ (x_f, y_f, x_s, y_s) :
    # Take only quarter dataset
    random.shuffle(x_f)
    random.shuffle(x_s)
    x_s_h = x_s[0:int(len(x_s))]
    y_s_h = y_s[0:int(len(x_s))]
    
    
    # Comment this for unblanced data
    # Stutter is less than fluent
    x_f_h = x_f[0:len(x_s_h)]
    y_f_h = y_f[0:len(x_s_h)]
    
    x_train = x_s_h + x_f_h
    y_train = y_s_h + y_f_h
    return x_train, y_train

class EarlyStopping:
  def __init__(self, patience, delta):
    self.patience = patience
    self.delta = delta
    self.best_loss = None
    self.num_no_improvement = 0
    self.best_model = None

  def __call__(self, validate_loss, model):

    if self.best_loss is None:
      self.best_loss = validate_loss
      self.best_model = model.state_dict()

    elif validate_loss >= self.best_loss + self.delta:
      self.num_no_improvement += 1

      if self.num_no_improvement >= self.patience:
          return True # stop here!

    else:
      self.num_no_improvement = 0 # improvement seen, so reset here
      self.best_loss = validate_loss
      self.best_model = model.state_dict()
      
  def load_model(self, model):
    model.load_state_dict(self.best_model)