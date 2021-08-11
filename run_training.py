import torch.optim as optim
from torch import nn
from dataset.AudioDataset import AudioDataset
from models.models import AudioExpressionNet3
import torch

num_epochs = 10
batch_size = 64

audio_dataset = AudioDataset("/data/stars/user/rtouchen/AudioVisualGermanDataset512/", 8)

train_set, test_set = torch.utils.data.random_split(audio_dataset, [101, 30])

trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                          shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

net = AudioExpressionNet3(8)
net.cuda()

criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(num_epochs):  

    running_loss = []
    for i, (data, target) in enumerate(trainloader,0):

        audios = data.to('cuda:0')
        labels = target.to('cuda:0')

        # zero the parameter gradientss
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(audios)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # log statistics
        running_loss.append(loss.item())