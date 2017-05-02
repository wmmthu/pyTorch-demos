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
class MLP(nn.Module):
	def __init__(self):
		super(MLP,self).__init__()
		self.fc1   = nn.Linear(3*32*32,100)
		self.fc2   = nn.Linear(100,10)
		self.bn    = nn.BatchNorm1d(100)
		self.dropout = nn.Dropout(0.5)

		self.main = nn.Sequential(
			nn.BatchNorm1d(3*32*32),
			nn.PReLU(),
			nn.Linear(3*32*32, 200),

			nn.BatchNorm1d(200),
			nn.PReLU(),
			nn.Linear(200,100),
			nn.BatchNorm1d(100),
			nn.PReLU(),
			nn.Linear(100,10),
		)

		self.initlayer = nn.Sequential(
			nn.BatchNorm1d(3*32*32),
			nn.PReLU(),
			nn.Linear(3*32*32, 200),
		)
		self.reslayer = nn.Sequential(
			nn.BatchNorm1d(200),
			nn.PReLU(),
			nn.Linear(200,100),
			nn.BatchNorm1d(100),
			nn.PReLU(),
			nn.Linear(100,200),	
		)  
		self.finalayer = nn.Sequential(
			nn.BatchNorm1d(200),
			nn.PReLU(),
			nn.Linear(200, 100),
			
		)
	def forward(self, x):
		x = x.view(-1, 3*32*32)
		# y = self.main(x)

		x = self.initlayer(x)
		x = self.reslayer(x) + x
		y = self.finalayer(x)
		return y

model     = MLP()
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

