import torch
import torch.nn as nn
import torchaudio

from transformers import AutoConfig, AutoModel

from huggingface_hub import login
login('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')

#from __future__ import print_function, division
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import os
import glob
from torch.utils.tensorboard import SummaryWriter
import random

from tqdm import tqdm
from datetime import datetime

# Custom functions
from helper_functions import set_seed
from helper_functions import __get_device__
from helper_functions import __shuffle_pick_quarter_data__
from helper_functions import __shuffle_data__
from helper_functions import EarlyStopping

def conv_audio_data (filename) :
        waveform, sample_rate = torchaudio.load(filename)

        # ensures both input and encoder are on the same device
        waveform = waveform.to(device)

        # Extract features
        emission = encoder(waveform)['last_hidden_state']
        emission = emission.cpu().detach().numpy()

        return emission

class AudioDataset(Dataset) :
        def __init__(self,x,y, n_samples) :
            # data loading
            self.x = x
            self.y = y 
            self.n_samples = n_samples
            
            
        def __getitem__(self,index) :
            return self.x[index], self.y[index]

        def __len__(self) :    
            return self.n_samples      

## MODEL DEFINITION ###
class StutterNet(nn.Module):
        def __init__(self, batch_size):
            super(StutterNet, self).__init__()
            # input shape = (batch_size, 1, 149,768)
            # in_channels is batch size
            self.layer1 = nn.Sequential(
                torch.nn.Conv2d(in_channels = 1, out_channels = 8, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(kernel_size=2, stride=2),
                torch.nn.Dropout(p=0.5)
            )
            self.layer1_bn = nn.BatchNorm2d(8)
            # input size = (batch_size, 8, 74, 384)
            self.layer2 = nn.Sequential(
                torch.nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(kernel_size=1, stride=2),
                torch.nn.Dropout(p=0.5)
            )
            self.layer2_bn = nn.BatchNorm2d(16)
            # input size = (batch_size, 16, 37, 192)
            self.flatten = torch.nn.Flatten()
            self.fc1 = nn.Linear(16* 37* 192,500, bias=True)
            self.fc1_bn = nn.BatchNorm1d(500)
            self.fc2 = nn.Linear(500,250, bias=True)
            self.fc2_bn = nn.BatchNorm1d(250)
            self.fc3 = nn.Linear(250,100, bias=True)
            self.fc3_bn = nn.BatchNorm1d(100)
            self.fc4 = nn.Linear(100,10, bias=True)
            self.fc4_bn = nn.BatchNorm1d(10)
            self.fc5 = nn.Linear(10,2, bias=True)

            self.relu = nn.LeakyReLU()
            self.sm = nn.Softmax()
        
        def forward(self, x):
            #print('Before Layer1',np.shape(x))
            out = self.layer1(x)
            # out = self.layer1_bn(out)
            # print('After layer 1',np.shape(out))
            out = self.layer2(out)
            # out = self.layer2_bn(out)
            # print('After layer 2',np.shape(out))
            out  = self.flatten(out)

            out = self.fc1(out)
            out = self.relu(out)
            # out = self.fc1_bn(out)

            out = self.fc2(out)
            out = self.relu(out)
            # out = self.fc2_bn(out)

            out = self.fc3(out)
            out = self.relu(out)
            # out = self.fc3_bn(out)

            out = self.fc4(out)
            out = self.relu(out)
            # out = self.fc4_bn(out)

            out = self.fc5(out)
            out = self.sm(out)
            #print('After final ',np.shape(out))

            # log_probs = torch.nn.functional.log_softmax(out, dim=1)

            return out

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
    return test_loss

def main():
    script_start_time = datetime.now()

    # get all data
    train_path_stutter = 'train_data/blocks'
    train_path_fluent  = 'train_data/fluent'
    test_path_stutter  = 'test_data/blocks'
    test_path_fluent   = 'test_data/fluent'

    set_seed(123)
    global device
    device = __get_device__()
    ##################################################################################################
    global writer
    writer = SummaryWriter()
    writer = SummaryWriter("wav2vec_base_model_quart_data_blocks")
    writer = SummaryWriter(comment="Quart dataset for binary classification blocks;")
    ##################################################################################################

    print("Loading encoder...")
    # MODEL_NAME = "alkiskoudounas/voc2vec-hubert-ls-pt" # voc2vec
    MODEL_NAME = "facebook/wav2vec2-base" # wav2vec
    global encoder_config
    global encoder
    encoder_config = AutoConfig.from_pretrained(MODEL_NAME)   
    encoder = AutoModel.from_pretrained(MODEL_NAME, device_map="cuda")

    print("Loading data...")
    data_start_time = datetime.now()

    x_f = []
    y_f = []
    x_s = []
    y_s = []

    discarded_s = 0
    # Convert to the embeddings for training data
    for filename in tqdm(glob.glob(os.path.join(train_path_stutter, '*.wav')), desc="Generating train stutter embeddings: "):
        stutter_np = conv_audio_data(filename)
        # fluent_np --> (1, 149, 768)
        # print(np.shape(stutter_np))
        if ((np.shape(stutter_np)[0] != 1) |(np.shape(stutter_np)[1] != 149) | (np.shape(stutter_np)[2] != 768)) :
            discarded_s += 1
        else:
            x_s.append(stutter_np)
            y_s.append(1)

    discarded = 0
    for filename in tqdm(glob.glob(os.path.join(train_path_fluent, '*.wav')), desc="Generating train fluent embeddings..."):
        fluent_np = conv_audio_data(filename)
        # fluent_np --> (1, 149, 768)
        if ((np.shape(fluent_np)[0] != 1) |(np.shape(fluent_np)[1] != 149) | (np.shape(fluent_np)[2] != 768)) :
            discarded += 1
        else:
            x_f.append(fluent_np)
            y_f.append(0)

    # x_train, y_train = __shuffle_pick_quarter_data__(x_f, y_f, x_s, y_s)
    x_train, y_train = __shuffle_data__ (x_f, y_f, x_s, y_s)
    # x_train = x_f + x_s
    # y_train = y_f + y_s   
    ##################################################################################################

    x_t_f = []
    y_t_f = []
    x_t_s = []
    y_t_s = []

    discarded_t_s = 0
    # Convert to the embeddings for test data
    for filename in tqdm(glob.glob(os.path.join(test_path_stutter, '*.wav')), desc = "Generating test stutter embeddings..."):
        stutter_np = conv_audio_data(filename)
        # stutter_np --> (1, 149, 768)
        if ((np.shape(stutter_np)[0] != 1) |(np.shape(stutter_np)[1] != 149) | (np.shape(stutter_np)[2] != 768)) :
            discarded_t_s += 1
        else:
            x_t_s.append(stutter_np)
            y_t_s.append(1)

    discarded_t = 0
    for filename in tqdm(glob.glob(os.path.join(test_path_fluent, '*.wav')), desc = "Generating test fluent embeddings..."):
        fluent_np = conv_audio_data(filename)
        # fluent_np --> (1, 149, 768)
        if ((np.shape(fluent_np)[0] != 1) |(np.shape(fluent_np)[1] != 149) | (np.shape(fluent_np)[2] != 768)) :
            discarded_t += 1
        else:
            x_t_f.append(fluent_np)
            y_t_f.append(0)

    random.shuffle(x_t_f)
    random.shuffle(x_t_s)
    # we do not want balanced classes for our testing
    # x_t_f = x_t_f[0:len(x_t_s)]
    # y_t_f = y_t_f[0:len(x_t_s)]

    x_test = x_t_s + x_t_f
    y_test = y_t_s + y_t_f

    data_finish_time = datetime.now()

    print("Loaded data in " + str(data_finish_time - data_start_time) + " seconds")

    ##################################################################################################
    ## Hyper parameters
    batch_size = 256
    num_epochs = 150
    learning_rate = 0.0001

    ## DATA LOADER ##
    # split data and translate to dataloader
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.1, random_state=123, shuffle=True, stratify = y_train)

    n_samples_train = np.shape(x_train)[0]
    n_samples_valid = np.shape(x_valid)[0]
    n_samples_test = np.shape(x_test)[0]
    # n_samples_valid = np.shape(x_valid)[0]
    print('Number of samples to train = ', n_samples_train)
    print('Number of samples to validate = ', n_samples_valid)
    print('Number of samples to test = ', n_samples_test)
    # print('Number of samples for validation = ', n_samples_valid)

    train_dataset = AudioDataset(x_train,y_train,n_samples_train)
    valid_dataset = AudioDataset(x_valid, y_valid, n_samples_valid)
    test_dataset = AudioDataset(x_test,y_test,n_samples_test)
    # valid_dataset = AudioDataset(x_valid,y_valid,n_samples_valid)

    global train_loader
    global valid_loader
    global model
    train_loader = DataLoader(dataset=train_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=0)
    valid_loader = DataLoader(dataset=valid_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=0)
    model = StutterNet(batch_size).to(device)
    print(model)

    # Loss and optimizer
    global criterion
    global optimizer
    global early_stopper
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    early_stopper = EarlyStopping(patience=5, delta=0.01)

    ## MODEL TRAINING LOOP ###
    global train_accu
    global train_losses
    global eval_accu
    global eval_losses

    train_accu= []
    train_losses = []
    eval_accu=[]
    eval_losses=[]

    torch.save(model, 'models/DisfluencyNet_snd_quart_blocks.pth')

    epochs=num_epochs
    train_start_time = datetime.now()
    for epoch in range(1,epochs+1): 
        train(epoch)
        eval_loss = test(epoch)
        torch.cuda.empty_cache()
        if early_stopper(eval_loss, model):
            print('Eval performance not improving; would be early stopping now!')
            model.load_state_dict(early_stopper.best_model)
            # TODO: figure out why break makes our number of predictions go to zero :|
            # break

    train_end_time = datetime.now()
    print("\nFinished training model in " + str(train_end_time - train_start_time) + " seconds.\n")

    ##################################################################################################
    # torch.save(model, 'DisfluencyNet_wp_quart.pth')
    ##################################################################################################
    ## TEST MODEL ##
    test_loader = DataLoader(dataset=test_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=0)

    model.load_state_dict(early_stopper.best_model)
    model.eval()

    with torch.no_grad():
        total = 0
        correct = 0
        n_correct = 0
        # Compute F1 score, precision and recall
        predicted_stutter = 0
        labels_stutter = 0
        correct_stutter = 0
        i = 0
        final_label = []
        final_predicted = []
        for features, labels in test_loader:
            features = features.to(device)
            labels = labels.to(device)
            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)
            loss= criterion(outputs,labels)
            
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            #    final_predicted.append(predicted)
            #    final_label.append(labels)

            # predicted = torch.reshape(predicted,(outputs.shape[0],1))
            
        for i in range (0, len(predicted)) :
            # F1 score for stutter
            if (predicted[i] == 1) :
                predicted_stutter +=1
            if (labels[i] == 1) :
                labels_stutter +=1   
            if ((predicted[i] == 1) & (labels[i] == 1)):
                correct_stutter +=1
            if (predicted[i] == labels[i]) :
                n_correct = n_correct + 1

    print(f'Predicted stutter: {predicted_stutter}%')
    print(f'Correct stutter predictions: {correct_stutter}%')

    acc_test = 100*correct/total
    print(f'Accuracy of the network on test dataset is : {acc_test} %')
    recall = correct_stutter/ labels_stutter
    print(f'Recall of the network on test dataset is : {recall}')
    precision = correct_stutter / predicted_stutter
    print(f'Precision of the network on test dataset is : {precision}')
    f1_score = 2 * precision * recall / (precision + recall)    
    print(f'F1 Score of the network on test dataset is : {f1_score}')

    script_finish_time = datetime.now()
    print("\nEntire script took " + str(script_finish_time - script_start_time) + " seconds to run.")

if __name__ == "__main__":
    main()