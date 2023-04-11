
import torch
from torchvision import datasets, models, transforms
import modelNew
import pickle
import argparse

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset',default='train',type=str,help="If 'train',generates embeddings from training set of CIFAR-100 ,else if 'test' does same from testing set of CIFAR-100")

	args = parser.parse_args()
	dataset = args.dataset

	type = args.dataset
	model, input_size = modelNew.initialize_alexnet(0, 100, 48)
	model.load_state_dict(torch.load(
		'models/CIFAR-100-alexnet-finetune.zip', map_location=torch.device('cpu')))
	model.eval()	

	data_transform = transforms.Compose([
		transforms.Resize(input_size),
		transforms.CenterCrop(input_size),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.486, 0.486], [0.229, 0.224, 0.225])
	])

	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


	image_datasets = {
     			'train': datasets.CIFAR100(
					root='./data',
					train=True,
					download=True,
					transform=data_transform
				),
     
				'test': datasets.CIFAR100(
					root='./data',
					train=False,
					download=True,
					transform=data_transform
				)
			}

	id2imgs = {}
	embeds_4096 = {}
	embeds_100 = {}
	hashes = {}

	for i,data in enumerate(image_datasets[dataset]):
		img,label = data
		print(label)
		
		
		img_new = torch.unsqueeze(img, 0)
		embed_4096, hashvec, embed_100 = model(img_new)

		id2imgs[i] = label
		embeds_4096[i] = torch.squeeze(embed_4096).detach().cpu().numpy()
		embeds_100[i] = torch.squeeze(embed_100).detach().cpu().numpy()
		hashes[i] = torch.squeeze(hashvec).detach().cpu().numpy()

	print('saving id2imgs')
	with open(f'embeddings/{dataset}/id2imgs_CIFAR-100.pickle','wb') as fp:
		pickle.dump(id2imgs, fp)

	print('saving 4096-d embeddings')
	with open(f'embeddings/{dataset}/embeddings_4096_CIFAR-100.pickle','wb') as fp:
		pickle.dump(embeds_4096, fp)

	print('saving 100-d embeddings')
	with open(f'embeddings/{dataset}/embeddings_100_CIFAR-100.pickle','wb') as fp:
		pickle.dump(embeds_100, fp)

	print('saving hashes')
	with open(f'embeddings/{dataset}/hashes_CIFAR-100.pickle','wb') as fp:
		pickle.dump(hashes, fp)
