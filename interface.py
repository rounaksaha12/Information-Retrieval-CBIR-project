import pickle
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
import torch.nn.functional as F
from torchsummary import summary
from sklearn.metrics.pairwise import cosine_similarity

from tqdm import tqdm
from datetime import datetime
import time
import modelNew
import trainNew
import pickle
import argparse
from retrieve_images import retrieve_images
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


if __name__ == '__main__':
    with torch.no_grad():
        parser = argparse.ArgumentParser()
        parser.add_argument('--k', default=20, type=int,
                            help='Number of top similar images to retrieve')
        parser.add_argument('--dataset', default='train', type=str,
                            help="If 'train',retrieves images from training set of CIFAR-100 ,else if 'test' testing set of CIFAR-100")
        args = parser.parse_args()

        topk = args.k
        type = args.dataset
        image_path = input("Enter the path to the image file: ")
        image = Image.open(image_path)
        label = input("Enter the label for the image: ")
        
        if type == 'train':
            with open('embeddings/train/id2imgs_CIFAR-100.pickle', 'rb') as f:
                img_dataset_lables = pickle.load(f)
            with open('embeddings/train/hashes_CIFAR-100.pickle', 'rb') as f:
                img_dataset_hashes = pickle.load(f)

            img_dataset_hashes = torch.from_numpy(img_dataset_hashes)
            img_dataset_hashes = torch.where(img_dataset_hashes >= 0.5, torch.ones_like(
                img_dataset_hashes), torch.zeros_like(img_dataset_hashes))

            with open('embeddings/train/id2imgs_CIFAR-100.pickle', 'rb') as f:
                img_dataset_labels = pickle.load(f)

            with open('embeddings/train/embeddings_100_CIFAR-100.pickle', 'rb') as f:
                img_dataset_embed_100 = pickle.load(f)

            img_dataset_embed_100 = torch.from_numpy(img_dataset_embed_100)
            F.normalize(img_dataset_embed_100, p=2,
                        dim=1, out=img_dataset_embed_100)

            with open('embeddings/train/embeddings_4096_CIFAR-100.pickle', 'rb') as f:
                img_dataset_embed_4096 = pickle.load(f)

            img_dataset_embed_4096 = torch.from_numpy(img_dataset_embed_4096)
            F.normalize(img_dataset_embed_4096, p=2,
                        dim=1, out=img_dataset_embed_4096)

        elif type == 'test':
            with open('embeddings/test/id2imgs_CIFAR-100.pickle', 'rb') as f:
                img_dataset_lables = pickle.load(f)
            with open('embeddings/test/hashes_CIFAR-100.pickle', 'rb') as f:
                img_dataset_hashes = pickle.load(f)

            img_dataset_hashes = torch.from_numpy(
                np.vstack(img_dataset_hashes))
            img_dataset_hashes = torch.where(img_dataset_hashes >= 0.5, torch.ones_like(
                img_dataset_hashes), torch.zeros_like(img_dataset_hashes))

            with open('embeddings/test/id2imgs_CIFAR-100.pickle', 'rb') as f:
                img_dataset_labels = pickle.load(f)

            with open('embeddings/test/embeddings_100_CIFAR-100.pickle', 'rb') as f:
                img_dataset_embed_100 = pickle.load(f)

            img_dataset_embed_100 = torch.from_numpy(
                np.vstack(img_dataset_embed_100))
            F.normalize(img_dataset_embed_100, p=2,
                        dim=1, out=img_dataset_embed_100)

            with open('embeddings/test/embeddings_4096_CIFAR-100.pickle', 'rb') as f:
                img_dataset_embed_4096 = pickle.load(f)

            img_dataset_embed_4096 = torch.from_numpy(
                np.vstack(img_dataset_embed_4096))
            F.normalize(img_dataset_embed_4096, p=2,
                        dim=1, out=img_dataset_embed_4096)

        with open('data/cifar-100-python/meta', 'rb') as f:
            ds_label = pickle.load(f)

        with open('embeddings/cifar100.unitsphere.pickle', 'rb') as f:
            class_embeds = pickle.load(f)

        model, input_size = modelNew.initialize_alexnet(0, 100, 48)
        model.load_state_dict(torch.load(
            'models/CIFAR-100-alexnet-finetune.zip', map_location=torch.device('cpu')))
        model.eval()

        data_transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.486, 0.486], [0.229, 0.224, 0.225])])
        
        
        img_new = data_transform(image)
        start = time.process_time()
        img_new = torch.unsqueeze(img_new, 0)
        embed_4096, hash, embed_100 = model(img_new)
        hash = torch.squeeze(hash)
        embed_100 = torch.squeeze(embed_100)
        embed_4096 = torch.squeeze(embed_4096)
        hash = torch.where(hash >= 0.5, torch.ones_like(
			hash), torch.zeros_like(hash))

		
        F.normalize(embed_100, dim=0, out=embed_100)
        F.normalize(embed_4096, dim=0, out=embed_4096)
        end = time.process_time()
        print(f'\nModel inference and pre-process time : {end-start}')
        retrieved_labels_1, retrieved_labels_2, similarity_time_1, similarity_time_2 = retrieve_images(
			    embed_100, embed_4096, hash, img_dataset_embed_100, img_dataset_embed_4096, img_dataset_hashes, img_dataset_labels, ds_label, label, topk)
		
		# display retrieved images
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
        )}
        
        # img, label = image_datasets[type][]:
        # load the image at the indexes in retrieved labels 1 and display in matplotlib
        # Assuming you have 20 images in the form of numpy arrays, each with shape (height, width, channels)
        images = []
        
        print(image_datasets[type][retrieved_labels_1[0]])

        # Add your 20 images to the `images` list

        # Combine the 20 images into a single numpy array with shape (20, height, width, channels)
        images_array = np.array(images)

        # Reshape the array to (4, 5, height, width, channels), where 4 and 5 are the number of rows and columns
        images_array = images_array.reshape(4, 5, *images_array.shape[1:])

        # Combine the images in each row and column using np.concatenate
        rows = [np.concatenate(images_array[i], axis=1) for i in range(images_array.shape[0])]
        combined_images = np.concatenate(rows, axis=0)

        # Display the combined images using plt.imshow
        plt.imshow(combined_images)
        plt.show()