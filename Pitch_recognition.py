# A pitch recognition task is done by constructing a Fully Connected Neural network, using
# sinwav and random noise as the training set and 12 pitches as the classification result.

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
# ########################################################


# ################# Generate Data #########################
# generate a 1 second sin wave in the frequency of 'pitch'
# and in the sampling rate of 44100.
pitch = [440., 466.2, 493.8, 523.3, 554.4, 587.3, 622.3, 659.3, 698.5, 740., 784.0, 830.6]
sr = 44100
sec = 1
batch_size = 128
nclass = len(pitch)
labels = np.arange(nclass)
X = []
y = []
init = 1        # flag of initialization
count = 0       # indicate the label of X
for freq in pitch:
    wave = sin_wave(freq, sr, sec, gain=1)
    n = noise(wave.shape, 0.1 * np.random.random_sample())
    wave += n
    # implement cqt on every frame generating N * n_bins data as representation
    x = librosa.cqt(wave, sr=sr, fmin=200, n_bins=36, filter_scale=2)
    x = librosa.amplitude_to_db(x, ref=np.min)
    x = x / np.max(x)
    if init == 1:
        X = x.transpose()
    else:
        X = np.vstack((X, x.transpose()))

    z = np.zeros(x.shape[1])
    ys = z + labels[count]
    if init == 1:
        y = ys
        init = 0
    else:
        y = np.hstack((y, ys))
    count = count + 1

# print(X.shape)  # (87*12)1044 * 36
# print(y.shape)  # 1044 * 12
# #############################################################


# ################### Data Regroup ###########################
# train: 788
# validation: 128
# test: 128
# training batch size: 32
N = X.shape[0]
choice = np.random.choice(N, N, replace=False)
print(choice)
X = X[choice]
y = y[choice]
num_val = batch_size                        # 128
num_test = batch_size                       # 128
num_train = N - num_val - num_test          # 788

mask = list(np.arange(num_val))
X_val = torch.from_numpy(X[mask].astype(np.float32))
y_val = torch.from_numpy(y[mask].astype(np.longlong))
mask = list(np.arange(num_val, num_val + num_test))
X_test = torch.from_numpy(X[mask].astype(np.float32))
y_test = torch.from_numpy(y[mask].astype(np.longlong))
mask = list(np.arange(num_val + num_test, N))
X_train = X[mask]
y_train = y[mask]
# ############################################################


# #################### Modeling ##############################
D_in, H, D_out = 36, 100, 12
device = torch.device('cpu')
writer = SummaryWriter('Logs_1')
num_epoch = 100
iters = batch_size

model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out)
).to(device)

loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
train_batch_size = 32
learning_rate = 1e-3
# ###############################################################


# ######################## Training #############################
print("Training...")
for ep in np.arange(num_epoch):
    for it in np.arange(24):
        print("Epoch:"+str(ep+1)+"/"+str(num_epoch), "  Iteration:"+str(it+1)+"/"+str(24))

        batch_choice = np.random.choice(num_train, train_batch_size, replace=False)
        X_batch = torch.from_numpy(X_train[batch_choice].astype(np.float32))
        y_batch = torch.from_numpy(y_train[batch_choice].astype(np.longlong))

        score = model(X_batch)
        loss = loss_fn(score, y_batch)
        writer.add_scalar('Loss', loss.item(), ep * 24 + it)
        print("Loss:", loss.item())

        y_pred = torch.argmax(score, dim=1)
        accuracy = 100 * torch.mean((y_pred == y_batch).float())
        writer.add_scalar('Accuracy', accuracy.item(), ep * 24 + it)
        print("Train acc:", accuracy.item(), '%')

        val_score = model(X_val)
        loss_val = loss_fn(val_score, y_val)
        writer.add_scalar('Loss_val', loss_val.item(), ep * 24 + it)

        y_val_pred = torch.argmax(val_score, dim=1)
        accuracy_val = 100 * torch.mean((y_val_pred == y_val).float())
        writer.add_scalar('Accuracy_val', accuracy_val.item(), ep * 24 + it)

        model.zero_grad()
        loss.backward()

        with torch.no_grad():
            for param in model.parameters():
                param.data -= learning_rate * param.grad
# ##############################################################################


# ############################# Testing ########################################
test_score = model(X_test)
y_test_pred = torch.argmax(test_score, dim=1)
accuracy_test = 100 * torch.mean((y_test_pred == y_test).float())
print("Final test accuracy is:", accuracy_test.item(), "%")
