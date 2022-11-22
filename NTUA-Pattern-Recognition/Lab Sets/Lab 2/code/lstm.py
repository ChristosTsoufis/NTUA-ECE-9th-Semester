
import os
import copy
from tqdm import tqdm
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# This class is used for framing the dataset
class FrameLevelDataset(Dataset):
    # feats:    list of numpy arrays that contain the sequence features (of shape seq_length * feature_dimension)
    # labels:   list that contains the label for each sequence (each label must be an integer)
    def __init__(self, feats, labels):
        self.lengths =  []
        for sample in feats:
            self.lengths.append(sample.shape[0])

        self.feats = self.zero_pad_and_stack(feats)
        self.labels = np.array(labels).astype('int64')
    
    # This function performs zero padding on a list of features and forms them into a numpy 3D array
    # returns padded: a 3D numpy array (of shape num_sequences * max_sequence_length * feature_dimension)
    def zero_pad_and_stack(self, x):
        padded = []
        max_sequence_length = max(self.lengths)
        for sample in x:
            padding = np.zeros((max_sequence_length - sample.shape[0], sample.shape[1]))
            padded.append(np.vstack((sample, padding)))
        return np.array(padded)

    def __getitem__(self, item):
        return self.feats[item], self.labels[item], self.lengths[item]

    def __len__(self):
        return len(self.feats)

# this class is used to implement the Basic LSTM
class BasicLSTM(nn.Module):
    def __init__(self, input_dim, rnn_size, output_dim, num_layers, bidirectional=False, dropout = 0):
        super(BasicLSTM, self).__init__()
        self.bidirectional = bidirectional
        self.feature_size  = rnn_size * 2 if self.bidirectional else rnn_size
        self.lstm   = nn.LSTM(input_dim, self.feature_size, num_layers, bidirectional = bidirectional, dropout = dropout, batch_first = True)
        self.linear = nn.Linear(self.feature_size, out_features = output_dim)

    # x is a 3D numpy array of dimension N * L * D = batch index * sequence index * feature index
    # lengths: N x 1
    def forward(self, x, lengths):
        batch_outputs, _ = self.lstm(x)
        last_outputs     = self.last_timestep(batch_outputs, lengths, self.bidirectional)
        predictions      = self.linear(last_outputs).squeeze(dim = -1)
        return predictions
    
    # Returns the last output of the LSTM taking into account the zero padding
    def last_timestep(self, outputs, lengths, bidirectional=False):
        if bidirectional:
            forward, backward = self.split_directions(outputs)
            last_forward  = self.last_by_index(forward, lengths)
            last_backward = backward[:, 0, :]
            # Concatenate and return
            return torch.cat((last_forward, last_backward), dim=-1)
        else:
            return self.last_by_index(outputs, lengths)

    @staticmethod
    def split_directions(outputs):
        direction_size = int(outputs.size(-1) / 2)
        forward  = outputs[:, :, :direction_size]
        backward = outputs[:, :, direction_size:]
        return forward, backward

    @staticmethod
    def last_by_index(outputs, lengths):
        # index of the last output for each sequence
        idx = (lengths - 1).view(-1, 1).expand(outputs.size(0), outputs.size(2)).unsqueeze(1)
        return outputs.gather(1, idx).squeeze()

# this function is used to evaluate the dataset
def eval_dataset(model, dataset):
    data_loader = DataLoader(dataset, batch_size = 128)
    loss = nn.CrossEntropyLoss()
    predictions = []
    true_labels = []

    with torch.no_grad():
        total_loss = 0
        for i, data in enumerate(data_loader):
            X_batch, y_batch, lengths_batch = data
            y_pred = model(X_batch.float(), lengths_batch)
            L = loss(y_pred.float(), y_batch)
            total_loss  = total_loss + L
            predictions = np.concatenate((predictions, np.argmax(y_pred, 1)))
            true_labels = np.concatenate((true_labels, y_batch))

    return total_loss/(i+1), predictions, true_labels

# this function is used to train the dataset
def train(model,
          X_train, y_train,
          X_dev, y_dev,
          batch_size = 512,
          epochs = 20,
          learning_rate = 1e-5,
          momentum = 0.73,
          weight_regularization = 0,
          early_stopping = False):

    train_data      = FrameLevelDataset(X_train, y_train)
    validation_data = FrameLevelDataset(X_dev, y_dev)

    data_loader = DataLoader(train_data, batch_size = batch_size)
    loss        = nn.CrossEntropyLoss()
    optimizer   = optim.SGD(model.parameters(),
                            lr = learning_rate,
                            momentum = momentum,
                            weight_decay = weight_regularization
                            )
    
    training_loss   = []
    validation_loss = []
    best_loss  = None
    best_model = None

    print("==========================================")

    train_loss, _, _ = eval_dataset(model, train_data)
    val_loss,   _, _ = eval_dataset(model, validation_data)

    print(f'Training   loss before training: {train_loss:.8f}')
    print(f'Validation loss before training: {val_loss:.8f}')

    training_loss.append(train_loss)
    validation_loss.append(val_loss)

    loss_increasing = 0

    for epoch in tqdm(range(epochs)):
        total_loss = 0
        for index, data in enumerate(data_loader):
            X_batch, y_batch, lengths_batch = data
            y_pred = model(X_batch.float(), lengths_batch)
            L      = loss(y_pred.float(), y_batch)
            total_loss = total_loss + L.item()

            optimizer.zero_grad()
            L.backward()
            optimizer.step()

        avg_loss = total_loss / (index+1)
        val_loss,   _, _ = eval_dataset(model, validation_data)

        validation_loss.append(val_loss)
        training_loss.append(avg_loss)

        print(f'Epoch {epoch}')
        print(f'Training   Loss: {avg_loss:.8f}')
        print(f'Validation Loss: {val_loss:.8f}')

        if(best_loss is None or val_loss < best_loss):
            best_loss  = val_loss
            best_model = copy.deepcopy(model.state_dict())
        
        # check if model is overfitting the training set or if the validation loss is climbing
        if early_stopping:
            # Overfitting
            if avg_loss * 2 < val_loss:
                break
            if validation_loss[:-1] > validation_loss[:-2]:
                loss_increasing += 1
            else:
                loss_increasing = 0
            if loss_increasing >= 5:
                break

    print("==========================================")
    return best_model, training_loss, validation_loss
