import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms

import numpy as np
import matplotlib.pyplot as plt 
import argparse
import time
from datetime import datetime
import os
import copy

data_dir = './'
num_classes = 10

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
	
	since = time.time()
	
	val_acc_hist = []

	best_model_wts = copy.deepcopy(model.state_dict())
	best_acc = 0.0

	for epoch in range(num_epochs):
		print(f'Epoch {epoch + 1}/{num_epochs}')
		print('-' * 10)

		for phase in ['train','val']:
			
			if phase == 'train':
				model.train()
			else:
				model.eval()

			running_loss = 0.0
			running_corrects = 0

			for inputs, labels in dataloaders[phase]:

				inputs = inputs.to(device)
				labels = labels.to(device)
				
				optimizer.zero_grad()
				with torch.set_grad_enabled(phase == 'train'):

					outputs = model(inputs)
					loss = criterion(outputs, labels)

				preds = torch.argmax(outputs, dim=1)

				if phase == 'train':
					loss.backward()
					optimizer.step()

				preds = torch.squeeze(preds)
				labels = torch.squeeze(labels)
				running_loss += loss.item() * inputs.size(0)
				running_corrects += torch.sum(preds == labels)
			
			epoch_loss = running_loss / len(dataloaders[phase].dataset)
			epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

			print(f'{phase} Loss: {epoch_loss} Acc: {epoch_acc}')

			if phase == 'val' and epoch_acc > best_acc:
				best_acc = epoch_acc
				best_model_wts = copy.deepcopy(model.state_dict())
			
			if phase == 'val':
				val_acc_hist.append(epoch_acc)
		
		print()
	
	time_elapsed = time.time() - since

	print(f'Training completed in {time_elapsed // 60} m {time_elapsed % 60} s')
	print(f'Best val Acc: {best_acc}')

	model.load_state_dict(best_model_wts)

	return model, val_acc_hist

def set_parameter_requires_grad(model, feature_extracing):
	if feature_extracing:
		for param in model.parameters():
			param.requires_grad = False

def initalize_resnet(feature_extract, use_pretrained=True):
	model_ft = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
	set_parameter_requires_grad(model_ft, feature_extract)
	num_fltrs = model_ft.fc.in_features
	model_ft.fc = nn.Linear(num_fltrs, num_classes)
	input_size = 224

	return model_ft, input_size


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--batch_size',default=8,type=int,help='Batch size to be used for both training and validation')
	parser.add_argument('--epochs',default=4,type=int,help='Number of training epochs')
	parser.add_argument('--lr',default=0.001,type=float,help='Learning rate of optimizer')
	parser.add_argument('--finetune',action='store_true',default=False,help='If set, the entire model will be trained, otherwise only the final layer')
	args = parser.parse_args()

	batch_size = args.batch_size
	num_epochs = args.epochs
	lr = args.lr
	feature_extract = not args.finetune

	now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
	print(f'Logs of run on {now}')
	print('='*40)
	print()
	print('Arguments: ')
	print('dataset = CIFAR-10')
	print(f'batch size = {batch_size}')
	print(f'epochs = {num_epochs}')
	print(f'learning rate = {lr}')
	print(f'finetune = {args.finetune}')
	print()

	model_ft, input_size = initalize_resnet(feature_extract,use_pretrained=True)

	print('Model: ')
	print(model_ft)

	data_transform = transforms.Compose([
		transforms.Resize(input_size),
		transforms.CenterCrop(input_size),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.486, 0.486], [0.229, 0.224, 0.225])
	])

	image_datasets = {
		'train': datasets.CIFAR10(
			root='./data',
			train=True,
			download=True,
			transform=data_transform
		),
		'val': datasets.CIFAR10(
			root='./data',
			train=False,
			download=True,
			transform=data_transform
		)
	}

	dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train','val']}

	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

	model_ft = model_ft.to(device)
	params_to_update = model_ft.parameters()
	print("Params to learn:")
	if feature_extract:
		params_to_update = []
		for name, param in model_ft.named_parameters():
			if param.requires_grad == True:
				params_to_update.append(param)
				print('\t',name)
	else:
		for name,param in model_ft.named_parameters():
			if param.requires_grad == True:
				print('\t',name)

	optimizer_ft = optim.SGD(params_to_update, lr=lr, momentum=0.9)

	criterion = nn.CrossEntropyLoss()

	model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs)

	PATH = f'./cifar-resnet-{'finetune' if args.finetune else 'ft'}-CIFAR10'
	torch.save(model_ft.state_dict(),PATH)

	print('--end--')
	print()


