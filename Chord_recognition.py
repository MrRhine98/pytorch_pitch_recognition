import numpy as np
import matplotlib.pyplot as plt
import librosa
import torch
import os
from torch.utils.tensorboard import SummaryWriter


# ################ Utility Function #####################
def sin_wave(f, sample, secs, gain):
    t = np.arange(secs * sample)/sample
    return gain * np.sin(2 * np.pi * f * t)


def noise(shape, gain):
    return gain * np.random.randn(shape[0])


def chord_wave(secs, f0, sr, gain, major):
    """major: bool"""
    t = np.arange(sr * secs)
    sine_f0 = gain * np.sin(2 * np.pi * f0 * t / sr)
    if major:
        sine_third = gain * np.sin(2 * np.pi * f0 * 2. ** (4./12.) * t / sr)
    else:
        sine_third = gain * np.sin(2 * np.pi * f0 * 2. ** (3./12.) * t / sr)
    return sine_f0 + sine_third + noise(sine_f0.shape, gain=0.1)

class Flatten(torch.nn.Module):
    def forward(self, input):
        return torch.flatten(input, start_dim=1)
# ########################################################


# ################# Data Generator ########################
class DataGen:
    def __init__(self, sr=44100, batch_size=128):
        self.pitches = [440., 466.2, 493.8, 523.3, 554.4, 587.3,
                        622.3, 659.3, 698.5, 740., 784.0, 830.6]
        self.sr = sr
        self.batch_size = batch_size
        self.sec = 1
        self.n_class = 2    # major or minor
        self.major_cqts = []
        self.minor_cqts = []
        self.nframe_decision = 3
        self.flag = 1
        for freq in self.pitches:
            cqt = librosa.cqt(chord_wave(self.sec, freq, self.sr, gain=0.5, major=True), sr=sr,
                              fmin=220, n_bins=36, filter_scale=2)  # use three frames!
            cqt = librosa.amplitude_to_db(cqt, ref=np.min)
            cqt = cqt / np.max(cqt)  # cqt in 2d

            if self.flag == 1:
                self.major_cqts = cqt.reshape((36, 3, 29, 1)).transpose(2, 3, 0, 1)
            else:
                self.major_cqts = np.concatenate((self.major_cqts, cqt.reshape((36, 3, 29, 1)).transpose(2, 3, 0, 1)))

            # (29, 36, 3)
            cqt = librosa.cqt(chord_wave(self.sec, freq, self.sr, gain=0.5, major=False), sr=sr,
                              fmin=220, n_bins=36, filter_scale=2)   # use three frame!
            cqt = librosa.amplitude_to_db(cqt, ref=np.min)
            cqt = cqt / np.max(cqt)
            if self.flag == 1:
                self.minor_cqts = cqt.reshape((36, 3, 29, 1)).transpose(2, 3, 0, 1)
                self.flag = 0
            else:
                self.minor_cqts = np.concatenate((self.minor_cqts, cqt.reshape((36, 3, 29, 1)).transpose(2, 3, 0, 1)))

            # plt.figure()
            # plt.imshow(self.major_cqts[1, :, :], cmap=plt.get_cmap('Blues'))
            # plt.show()

            # print(self.minor_cqts.shape)
            # print(self.major_cqts.shape)
            # (348, 36, 3)

    def __next__(self):
        self.ntrain = 568
        self.nval = 64
        self.ntest = 64
        ntrain = self.ntrain
        nval = self.nval
        ntest = self.ntest
        choice = np.random.choice(696, 696)


        X = {}
        y = {}
        y_major = np.ones(self.major_cqts.shape[0])
        y_minor = np.zeros(self.minor_cqts.shape[0])
        X_data = np.concatenate((self.minor_cqts, self.major_cqts))[choice]
        y_data = np.concatenate((y_major, y_minor))[choice]

        mask = list(np.arange(ntrain))
        X['train'] = X_data[mask]
        y['train'] = y_data[mask]

        mask = list(np.arange(ntrain, ntrain+nval))
        X['val'] = X_data[mask]
        y['val'] = y_data[mask]

        mask = list(np.arange(ntrain+nval, ntrain+nval+ntest))
        X['test'] = X_data[mask]
        y['test'] = y_data[mask]

        return X, y

    next = __next__
# #########################################################################


datagen = DataGen()         # generate the data
X, y = next(datagen)
writer = SummaryWriter('Logs_2')

# ########################### Modeling #####################################
device = torch.device('cpu')
nfilter = 4
loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
learning_rate = 1e-3
model = torch.nn.Sequential(
    torch.nn.Conv2d(1, nfilter, (5, 3)),
    torch.nn.MaxPool2d((32, 1)),
    torch.nn.ReLU(),     # (32, 4, 6, 1)
    Flatten(),
    torch.nn.Linear(4, 2)
)
# ##########################################################################

# ########################### Training #####################################
X_val = torch.from_numpy(X['val'].astype(np.float32))
y_val = torch.from_numpy(y['val'].astype(np.longlong))
X_test = torch.from_numpy(X['test'].astype(np.float32))
y_test = torch.from_numpy(y['test'].astype(np.longlong))
print("Training...")
nepoch = 100
batch_size = 32
for ep in np.arange(nepoch):
    for i in np.arange(datagen.ntrain/batch_size):
        print("Epoch:", ep+1, "/", nepoch)
        batch_choice = np.random.choice(datagen.ntrain, batch_size)
        X_batch = torch.from_numpy(X['train'][batch_choice].astype(np.float32))
        y_batch = torch.from_numpy(y['train'][batch_choice].astype(np.longlong))

        score = model(X_batch)
        loss = loss_fn(score, y_batch)
        writer.add_scalar('Loss_train', loss.item(), ep * batch_size + i)
        print("Loss:", loss.item())

        y_pred = torch.argmax(score, dim=1)
        accuracy = 100 * torch.mean((y_pred == y_batch).float())
        writer.add_scalar('Accuracy', accuracy.item(), ep * batch_size + i)
        print("Train acc:", accuracy.item(), '%')


        score_val = model(X_val)
        loss_val = loss_fn(score_val, y_val)
        writer.add_scalar('Loss_val', loss_val.item(), ep * batch_size + i)

        y_val_pred = torch.argmax(score_val, dim=1)
        accuracy_val = 100 * torch.mean((y_val_pred == y_val).float())
        writer.add_scalar('Accuracy_val', accuracy_val.item(), ep * batch_size + i)


        model.zero_grad()
        loss.backward()
        with torch.no_grad():
            for param in model.parameters():
                param.data -= learning_rate * param.grad
# ####################################### Test ##########################################

test_score = model(X_test)
y_test_pred = torch.argmax(test_score, dim=1)
accuracy_test = 100 * torch.mean((y_test_pred == y_test).float())
print("Final test accuracy is:", accuracy_test.item(), "%")


