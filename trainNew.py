import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms

import pickle
from modelNew import *
import numpy as np
import matplotlib.pyplot as plt 
import argparse
import time
from datetime import datetime
import os
import copy

data_dir = './'

if __name__ == '__main__':

	parser = argparse.ArgumentParser()

	parser.add_argument('--dataset',default='CIFAR-10',type=str,help='Dataset to be used for training and validation')
	parser.add_argument('--class_embed',default=None,type=str,help='path to pickle file containing class embeddings')

	parser.add_argument('--batch_size',default=8,type=int,help='Batch size to be used for both training and validation')
	parser.add_argument('--epochs',default=4,type=int,help='Number of training epochs')
	parser.add_argument('--lr',default=0.001,type=float,help='Learning rate of optimizer')
	parser.add_argument('--hash_size',default=48,type=int,help='Dimention of learned hashes')
	parser.add_argument('--finetune',action='store_true',default=False,help='If set, the entire model will be trained, otherwise only the final layer')

	parser.add_argument('--dump',default=3,type=int,help='dumping method')
	parser.add_argument('--save_model',action='store_true',default=True,help='If set, the model will be saved after training')
	parser.add_argument('--embedding_4096_dump',default='embeddings',type=str,help='Path to dump the learned 4690-d embeddings')
	parser.add_argument('--embedding_100_dump',default='embeddings',type=str,help='Path to dump the learned 100-d embeddings')
	parser.add_argument('--hash_dump',default='hashes',type=str,help='Path to dump the learned hashes')
	parser.add_argument('--id2imgs_dump',default='id2imgs',type=str,help='Path to dump the mapping of image ids to labels')
	args = parser.parse_args()

	dataset = args.dataset
	batch_size = args.batch_size
	num_epochs = args.epochs
	lr = args.lr
	feature_extract = not args.finetune
	hash_size = args.hash_size
	dump = args.dump
	save_model = args.save_model
	embedding_4096_dump = args.embedding_4096_dump
	embedding_100_dump = args.embedding_100_dump
	hash_dump = args.hash_dump
	id2imgs_dump = args.id2imgs_dump
	num_classes = 100 if dataset == 'CIFAR-100' else 10

	if (args.class_embed is not None):
		with open(args.class_embed,'rb') as fp:
			class_embeddings = pickle.load(fp)['embedding']
			class_embeddings = torch.from_numpy(class_embeddings)
	else:
		print('class embeddings are required')
		exit()

	now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
	print(f'Logs of run on {now}')
	print('='*40)
	print()
	print('Arguments: ')
	print(f'dataset = {dataset}')
	print(f'batch size = {batch_size}')
	print(f'epochs = {num_epochs}')
	print(f'learning rate = {lr}')
	print(f'finetune = {args.finetune}')
	print(f'class embeddings = {args.class_embed}')
	print()

	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

	model_ft, input_size = initialize_alexnet(feature_extract, num_classes, hash_size)
	model_ft = model_ft.to(device)
	class_embeddings = class_embeddings.to(device)

	data_transform = transforms.Compose([
		transforms.Resize(input_size),
		transforms.CenterCrop(input_size),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.486, 0.486], [0.229, 0.224, 0.225])
	])

	if dataset == 'CIFAR-10':
		num_classes = 10
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

	elif dataset == 'CIFAR-100':
		num_classes = 100
		image_datasets = {
			'train': datasets.CIFAR100(
				root='./data',
				train=True,
				download=True,
				transform=data_transform
			),
			'val': datasets.CIFAR100(
				root='./data',
				train=False,
				download=True,
				transform=data_transform
			)
		}

	else:
		print('Invalid dataset')
		exit()

	print('Model: ')
	print(model_ft)


	dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=2) for x in ['train','val']}

	params_to_update = []
	for name, param in model_ft.named_parameters():
		if param.requires_grad == True:
			params_to_update.append(param)

	optimizer_ft = optim.SGD(params_to_update, lr=lr, momentum=0.9)

	# criterion = nn.CrossEntropyLoss()

	model_ft, hist = train_model(model_ft, class_embeddings, device, dataloaders_dict, optimizer_ft, num_epochs=num_epochs)

	# dont confuse why train set is being used to create dalaloader called testloader
	# these are the set of images we will use for further experiments, like image retrieval
	# so basically this is the collection of images from which top k images have to be retrieved
	# we will use the images in train set as example queries
	testloader = torch.utils.data.DataLoader(
		image_datasets['train'], 
		batch_size=batch_size, 
		shuffle=False
	)

	
	model_ft.eval()

	if dump == 1:

		testloader = torch.utils.data.DataLoader(
			image_datasets['train'], 
			batch_size=1, 
			shuffle=False
		)

		id2imgs = {}
		embeds_4096 = {}
		embeds_100 = {}
		hashes = {}

		for i, data in enumerate(testloader, 0):
			
			img, label = data

			img = img.to(device)
			label = label.to(device)

			embed_4096, hashvec, embed_100 = model_ft(img)

			id2imgs[i] = torch.squeeze(label).detach().cpu().numpy()
			embeds_4096[i] = torch.squeeze(embed_4096).detach().cpu().numpy()
			embeds_100[i] = torch.squeeze(embed_100).detach().cpu().numpy()
			hashes[i] = torch.squeeze(hashvec).detach().cpu().numpy()

	elif dump == 2:

		id2imgs = {}
		embeds_4096 = {}
		embeds_100 = {}
		hashes = {}
		offset = 0

		for i, data in enumerate(testloader, 0):

			imgs, labels = data
			imgs = imgs.to(device)
			labels = labels.to(device)

			embedvec_4096, hashvec, embedvec_100 = model_ft(imgs)

			for j in range(imgs.size(0)):

				id2imgs[offset] = labels[j].detach().cpu().numpy()
				embeds_4096[offset] = embedvec_4096[j].detach().cpu().numpy()
				embeds_100[offset] = embedvec_100[j].detach().cpu().numpy()
				hashes[offset] = hashvec[j].detach().cpu().numpy()
				offset += 1

	elif dump == 3:
		id2imgs = []
		embeds_4096 = []
		embeds_100 = []
		hashes = []

		for i, data in enumerate(testloader, 0):

			imgs, labels = data
			imgs = imgs.to(device)
			labels = labels.to(device)

			embedvec_4096, hashvec, embedvec_100 = model_ft(imgs)

			id2imgs.append(labels.detach().cpu())
			embeds_4096.append(embedvec_4096.detach().cpu())
			embeds_100.append(embedvec_100.detach().cpu())
			hashes.append(hashvec.detach().cpu())

		id2imgs = torch.cat(id2imgs, dim=0).numpy()
		embeds_4096 = torch.cat(embeds_4096, dim=0).numpy()
		embeds_100 = torch.cat(embeds_100, dim=0).numpy()
		hashes = torch.cat(hashes, dim=0).numpy()

	else:
		print('Invalid dump argument')
		exit()

	print('saving id2imgs')
	with open(f'{id2imgs_dump}_{dataset}.pickle','wb') as fp:
		pickle.dump(id2imgs, fp)
	
	print('saving 4096-d embeddings')
	with open(f'{embedding_4096_dump}_{dataset}.pickle','wb') as fp:
		pickle.dump(embeds_4096, fp)

	print('saving 100-d embeddings')
	with open(f'{embedding_100_dump}_{dataset}.pickle','wb') as fp:
		pickle.dump(embeds_100, fp)
	
	print('saving hashes')
	with open(f'{hash_dump}_{dataset}.pickle','wb') as fp:
		pickle.dump(hashes, fp)

	model_ft.train()

	if save_model:
		temp = 'finetune' if args.finetune else 'ft'
		PATH = f'./{dataset}-alexnet-{temp}'
		torch.save(model_ft.state_dict(),PATH)

	print()
	print('--end--')
	print()
