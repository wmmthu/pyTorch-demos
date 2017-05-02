import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

# load CIFAR data
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=50, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=50, shuffle=True, num_workers=2)
classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# define network
class CNN(nn.Module):
	def __init__(self):
		super(CNN,self).__init__()
		
		self.conv_map = nn.Sequential(
			nn.BatchNorm2d(3),

			nn.Conv2d(3,16,3,1,1),
			nn.BatchNorm2d(16),
			nn.PReLU(),
			nn.MaxPool2d(2,2),

			nn.Conv2d(16,32,3,1,1),
			nn.BatchNorm2d(32),
			nn.PReLU(),
			nn.MaxPool2d(2,2),

			nn.Conv2d(32,64,3,1,1),
			nn.BatchNorm2d(64),
			nn.PReLU(),
			nn.MaxPool2d(2,2)
		)

		self.linear_map = nn.Sequential(
			nn.Linear(64*4*4, 100),
			nn.BatchNorm1d(100),
			nn.PReLU(),
			nn.Linear(100,10),
		)
	def forward(self, x):
		x = self.conv_map(x)
		x = x.view(-1,64*4*4) # flatten
		y = self.linear_map(x)
		return y

model     = CNN()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn   = nn.CrossEntropyLoss()

def test():
	correct = 0
	total = 0
	for data in testloader:
		imgs, labs = data
		_, pred = torch.max(model(Variable(imgs)).data, 1)
		total += labs.size(0)
		correct += torch.sum(labs == pred)
	return correct * 1. / total


for epoch in range(20):
	running_loss = 0.
	for i, data in enumerate(trainloader):
		imgs, labs = data
		imgs, labs = Variable(imgs), Variable(labs)

		optimizer.zero_grad()
		y = model(imgs)
		loss = loss_fn(y,labs)
		loss.backward()
		optimizer.step()

		running_loss += loss.data[0]
	print 'epoch %d, loss : %f, test accu : %f' % (epoch, running_loss, test())




