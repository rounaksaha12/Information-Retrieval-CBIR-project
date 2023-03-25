import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import time

transform = transforms.Compose(
		[transforms.ToTensor(),
		 transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]
)

batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='./data',
										train=True,
										download=True,
										transform=transform)

testset = torchvision.datasets.CIFAR10(root='./data',
										train=False,
										download=True,
										transform=transform)

trainloader = torch.utils.data.DataLoader(trainset,
										batch_size=4,
										shuffle=True,
										num_workers=2)

testloader = torch.utils.data.DataLoader(testset,
										batch_size=batch_size,
										shuffle=True,
										num_workers=2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ConvNet(nn.Module):
	def __init__(self):
		super().__init__()
		self.conv1 = nn.Conv2d(3,6,5)
		self.pool = nn.MaxPool2d(2,2)
		self.conv2 = nn.Conv2d(6,16,5)
		self.fc1 = nn.Linear(16*5*5,120,120)
		self.fc2 = nn.Linear(120,84)
		self.fc3 = nn.Linear(84,10)

	def forward(self,x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = torch.flatten(x,1)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x

net = ConvNet()
net.to(device)

print(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(),lr=0.001,momentum=0.9)

start = time.perf_counter()
for epoch in tqdm(range(2)):

	running_loss = 0.0
	for i,data in enumerate(trainloader,0):
		inputs, labels = data
		inputs, labels = inputs.to(device), labels.to(device)
		optimizer.zero_grad()
		outputs = net(inputs)
		loss = criterion(outputs,labels)
		loss.backward()
		optimizer.step()

		running_loss += loss.item()
		if i % 2000 == 1999:
			print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
			running_loss = 0.0

	with torch.no_grad():
		test_acc = 0.0
		for i, data in enumerate(testloader,0):
			test_inputs, test_labels = data
			test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)
			test_outputs = net(test_inputs)
			test_labels = torch.squeeze(test_labels)
			test_outputs = torch.squeeze(torch.argmax(test_outputs, dim=1))
			test_acc += torch.sum(test_outputs == test_labels)
		test_acc /= len(testset)
		print(f'Epoch {epoch + 1}: Test accuracy = {test_acc}')

end = time.perf_counter()
print('Finished Training')

print(f'Training required {end - start} seconds')
