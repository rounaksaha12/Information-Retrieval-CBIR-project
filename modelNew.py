import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
import torch.nn.functional as F

from tqdm import tqdm
import numpy as np
import time
from datetime import datetime
import os
import copy

def train_model(model, class_embeddings, device, dataloaders, optimizer, num_epochs=25):
	
	since = time.time()
	
	val_loss_hist = []

	best_model_wts = copy.deepcopy(model.state_dict())
	best_loss = float('inf')

	criterion = nn.CosineEmbeddingLoss()

	for epoch in tqdm(range(num_epochs)):
		print(f'Epoch {epoch + 1}/{num_epochs}')
		print('-' * 10)

		for phase in ['train','val']:
			
			if phase == 'train':
				model.train()
			else:
				model.eval()

			running_loss = 0.0
			# running_corrects = 0

			for inputs, labels in dataloaders[phase]:

				inputs = inputs.to(device)
				labels = labels.to(device)
				targets = torch.ones(inputs.size(0)).to(device)
				
				optimizer.zero_grad()
				with torch.set_grad_enabled(phase == 'train'):

					_, __, outputs = model(inputs)
					loss = criterion(outputs, class_embeddings[labels], targets)

				# preds = torch.argmax(outputs, dim=1)

				if phase == 'train':
					loss.backward()
					optimizer.step()

				# preds = torch.squeeze(preds)
				# labels = torch.squeeze(labels)
				running_loss += loss.item() * inputs.size(0)
				# running_corrects += torch.sum(preds == labels)
			
			epoch_loss = running_loss / len(dataloaders[phase].dataset)
			# epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

			print(f'{phase} Loss: {epoch_loss}')

			if phase == 'val' and epoch_loss < best_loss:
				best_loss = epoch_loss
				best_model_wts = copy.deepcopy(model.state_dict())
			
			if phase == 'val':
				val_loss_hist.append(epoch_loss)
		
		print()
	
	time_elapsed = time.time() - since

	print(f'Training completed in {time_elapsed // 60} m {time_elapsed % 60} s')
	print(f'Best val loss: {best_loss}')

	model.load_state_dict(best_model_wts)

	return model, val_loss_hist

def initialize_alexnet(feature_extract, num_classes, hash_size=48):
	model_ft = AlexNet_hash(num_classes, hash_size)
	for param in model_ft.alexnet.parameters():
		if feature_extract:
			param.requires_grad = False
		else:
			param.feature_extract = True
	
	
	print('Parameters to update: ')
	for name, param in model_ft.named_parameters():
		if param.requires_grad == True:
			print('\t', name)
	input_size = 224

	return model_ft, input_size

class Identity(nn.Module):
	def __init__(self):
		super(Identity, self).__init__()
		
	def forward(self, x):
		return x

class AlexNet_hash(nn.Module):
	def __init__(self, num_classes, hash_size=48):
		super().__init__()
		self.num_classes = num_classes
		self.hash_size = hash_size
		self.alexnet = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
		self.embed_dim = self.modify_alexnet()
		self.hidden = nn.Linear(self.embed_dim, hash_size)
		self.hash = nn.Sigmoid()
		self.classifier = nn.Linear(hash_size, num_classes)

	def modify_alexnet(self):
		embed_dim = self.alexnet.classifier[4].out_features
		self.alexnet.classifier[6] = Identity()
		return embed_dim

	def forward(self, x):
		x = self.alexnet(x)
		embeds = x
		x = self.hidden(x)
		x = self.hash(x)
		hashes = x
		x = self.classifier(x)
		return embeds, hashes, x
