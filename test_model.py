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
from evaluate_retrieval import evaluate_retrieval
from retrieve_images import retrieve_images
import numpy as np
import matplotlib.pyplot as plt

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

        map1_1, map1_5, map1_10, map1_15, map1_20, map1_25, map1_50, map1_100, map1_150, map1_200, map1_k = [0.0]*11
        map2_1, map2_5, map2_10, map2_15, map2_20, map2_25, map2_50, map2_100, map2_150, map2_200, map2_k = [0.0]*11

        avg_inference_time = 0.0
        avg_similarity_time_1 = 0.0
        avg_similarity_time_2 = 0.0

        mhp1_1, mhp1_5, mhp1_10, mhp1_15, mhp1_20, mhp1_25, mhp1_50, mhp1_100, mhp1_150, mhp1_200, mhp1_k = [0.0]*11
        mhp2_1, mhp2_5, mhp2_10, mhp2_15, mhp2_20, mhp2_25, mhp2_50, mhp2_100, mhp2_150, mhp2_200, mhp2_k = [0.0]*11

        mbal_acc_1 = 0.0
        mbal_acc_2 = 0.0

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

        # construct a class similar 2-d matrix of size 100*100 where i,j the entry
        # contains the similarity of ith and jth class
        class_sim_matrix = np.zeros((100, 100))
        for i in range(100):
            for j in range(100):
                x = class_embeds['embedding'][i]
                y = class_embeds['embedding'][j]
                class_sim_matrix[i][j] = cosine_similarity(
                    x.reshape(1, -1), y.reshape(1, -1))

        # sort each row of list in place
        class_sim_matrix = np.sort(
            class_sim_matrix, axis=1, kind='mergesort')[:, ::-1]
        # print(class_sim_matrix)

        cnt = 0
        for img, label in image_datasets["test"]:
            start = time.process_time()
            img_new = torch.unsqueeze(img, 0)
            embed_4096, hash, embed_100 = model(img_new)
            hash = torch.squeeze(hash)
            embed_100 = torch.squeeze(embed_100)
            embed_4096 = torch.squeeze(embed_4096)
            hash = torch.where(hash >= 0.5, torch.ones_like(
                hash), torch.zeros_like(hash))

            # F.normalize(hash, dim=0, out=hash)
            F.normalize(embed_100, dim=0, out=embed_100)
            F.normalize(embed_4096, dim=0, out=embed_4096)
            end = time.process_time()

            print(f'\nModel inference and pre-process time : {end-start}')
            avg_inference_time += (end-start)
            retrieved_labels_1, retrieved_labels_2, similarity_time_1, similarity_time_2 = retrieve_images(
                 embed_100, embed_4096, hash, img_dataset_embed_100, img_dataset_embed_4096, img_dataset_hashes, img_dataset_labels, ds_label, label, topk)

            p1_array, hp1_array, bal_acc_1 = evaluate_retrieval(
                label, class_sim_matrix, retrieved_labels=retrieved_labels_1, k_list=[1, 5, 10, 15, 20, 25, 50, 100, 150, 200, topk])
            p2_array, hp2_array, bal_acc_2 = evaluate_retrieval(
                label, class_sim_matrix, retrieved_labels=retrieved_labels_2, k_list=[1, 5, 10, 15, 20, 25, 50, 100, 150, 200, topk])

            map1_1 += p1_array[0]
            map1_5 += p1_array[1]
            map1_10 += p1_array[2]
            map1_15 += p1_array[3]
            map1_20 += p1_array[4]
            map1_25 += p1_array[5]
            map1_50 += p1_array[6]
            map1_100 += p1_array[7]
            map1_150 += p1_array[8]
            map1_200 += p1_array[9]
            map1_k += p1_array[10]
            map2_1 += p2_array[0]
            map2_5 += p2_array[1]
            map2_10 += p2_array[2]
            map2_15 += p2_array[3]
            map2_20 += p2_array[4]
            map2_25 += p2_array[5]
            map2_50 += p2_array[6]
            map2_100 += p2_array[7]
            map2_150 += p2_array[8]
            map2_200 += p2_array[9]
            map2_k += p2_array[10]
            mhp1_1 += hp1_array[0]
            mhp1_5 += hp1_array[1]
            mhp1_10 += hp1_array[2]
            mhp1_15 += hp1_array[3]
            mhp1_20 += hp1_array[4]
            mhp1_25 += hp1_array[5]
            mhp1_50 += hp1_array[6]
            mhp1_100 += hp1_array[7]
            mhp1_150 += hp1_array[8]
            mhp1_200 += hp1_array[9]
            mhp1_k += hp1_array[10]
            mhp2_1 += hp2_array[0]
            mhp2_5 += hp2_array[1]
            mhp2_10 += hp2_array[2]
            mhp2_15 += hp2_array[3]
            mhp2_20 += hp2_array[4]
            mhp2_25 += hp2_array[5]
            mhp2_50 += hp2_array[6]
            mhp2_100 += hp2_array[7]
            mhp2_150 += hp2_array[8]
            mhp2_200 += hp2_array[9]
            mhp2_k += hp2_array[10]
            mbal_acc_1 += bal_acc_1
            mbal_acc_2 += bal_acc_2
            avg_similarity_time_1 += similarity_time_1
            avg_similarity_time_2 += similarity_time_2

            if (cnt+1) % 100 == 0:
                print(
                    f"\nMean Average precision@1 of top {topk} images using method 1 : {map1_1/(cnt+1)}")
                print(
                    f"\nMean Average precision@5 of top {topk} images using method 1 : {map1_5/(cnt+1)}")
                print(
                    f"\nMean Average precision@10 of top {topk} images using method 1 : {map1_10/(cnt+1)}")
                print(
                    f"\nMean Average precision@15 of top {topk} images using method 1 : {map1_15/(cnt+1)}")
                print(
                    f"\nMean Average precision@20 of top {topk} images using method 1 : {map1_20/(cnt+1)}")
                print(
                    f"\nMean Average precision@25 of top {topk} images using method 1 : {map1_25/(cnt+1)}")
                print(
                    f"\nMean Average precision@50 of top {topk} images using method 1 : {map1_50/(cnt+1)}")
                print(
                    f"\nMean Average precision@100 of top {topk} images using method 1 : {map1_100/(cnt+1)}")
                print(
                    f"\nMean Average precision@150 of top {topk} images using method 1 : {map1_150/(cnt+1)}")
                print(
                    f"\nMean Average precision@200 of top {topk} images using method 1 : {map1_200/(cnt+1)}")
                print(
                    f"\nMean Average precision@{topk} of top {topk} images using method 1 : {map1_k/(cnt+1)}")
                print(
                    f"\nMean Average precision@1 of top {topk} images using method 2 : {map2_1/(cnt+1)}")
                print(
                    f"\nMean Average precision@5 of top {topk} images using method 2 : {map2_5/(cnt+1)}")
                print(
                    f"\nMean Average precision@10 of top {topk} images using method 2 : {map2_10/(cnt+1)}")
                print(
                    f"\nMean Average precision@15 of top {topk} images using method 2 : {map2_15/(cnt+1)}")
                print(
                    f"\nMean Average precision@20 of top {topk} images using method 2 : {map2_20/(cnt+1)}")
                print(
                    f"\nMean Average precision@25 of top {topk} images using method 2 : {map2_25/(cnt+1)}")
                print(
                    f"\nMean Average precision@50 of top {topk} images using method 2 : {map2_50/(cnt+1)}")
                print(
                    f"\nMean Average precision@100 of top {topk} images using method 2 : {map2_100/(cnt+1)}")
                print(
                    f"\nMean Average precision@150 of top {topk} images using method 2 : {map2_150/(cnt+1)}")
                print(
                    f"\nMean Average precision@200 of top {topk} images using method 2 : {map2_200/(cnt+1)}")
                print(
                    f"\nMean Average precision@{topk} of top {topk} images using method 2 : {map2_k/(cnt+1)}")
                print(
                    f"\nMean Average hierarchical precision@1 of top {topk} images using method 1 : {mhp1_1/(cnt+1)}")
                print(
                    f"\nMean Average hierarchical precision@5 of top {topk} images using method 1 : {mhp1_5/(cnt+1)}")
                print(
                    f"\nMean Average hierarchical precision@10 of top {topk} images using method 1 : {mhp1_10/(cnt+1)}")
                print(
                    f"\nMean Average hierarchical precision@15 of top {topk} images using method 1 : {mhp1_15/(cnt+1)}")
                print(
                    f"\nMean Average hierarchical precision@20 of top {topk} images using method 1 : {mhp1_20/(cnt+1)}")
                print(
                    f"\nMean Average hierarchical precision@25 of top {topk} images using method 1 : {mhp1_25/(cnt+1)}")
                print(
                    f"\nMean Average hierarchical precision@50 of top {topk} images using method 1 : {mhp1_50/(cnt+1)}")
                print(
                    f"\nMean Average hierarchical precision@100 of top {topk} images using method 1 : {mhp1_100/(cnt+1)}")
                print(
                    f"\nMean Average hierarchical precision@150 of top {topk} images using method 1 : {mhp1_150/(cnt+1)}")
                print(
                    f"\nMean Average hierarchical precision@200 of top {topk} images using method 1 : {mhp1_200/(cnt+1)}")
                print(
                    f"\nMean Average hierarchical precision@{topk} of top {topk} images using method 1 : {mhp1_k/(cnt+1)}")
                print(
                    f"\nMean Average hierarchical precision@1 of top {topk} images using method 1 : {mhp1_1/(cnt+1)}")
                print(
                    f"\nMean Average hierarchical precision@5 of top {topk} images using method 1 : {mhp1_5/(cnt+1)}")
                print(
                    f"\nMean Average hierarchical precision@10 of top {topk} images using method 2 : {mhp2_10/(cnt+1)}")
                print(
                    f"\nMean Average hierarchical precision@15 of top {topk} images using method 2 : {mhp2_15/(cnt+1)}")
                print(
                    f"\nMean Average hierarchical precision@20 of top {topk} images using method 2 : {mhp2_20/(cnt+1)}")
                print(
                    f"\nMean Average hierarchical precision@25 of top {topk} images using method 2 : {mhp2_25/(cnt+1)}")
                print(
                    f"\nMean Average hierarchical precision@50 of top {topk} images using method 2 : {mhp2_50/(cnt+1)}")
                print(
                    f"\nMean Average hierarchical precision@100 of top {topk} images using method 2 : {mhp2_100/(cnt+1)}")
                print(
                    f"\nMean Average hierarchical precision@150 of top {topk} images using method 2 : {mhp2_150/(cnt+1)}")
                print(
                    f"\nMean Average hierarchical precision@200 of top {topk} images using method 2 : {mhp2_200/(cnt+1)}")
                print(
                    f"\nMean Average hierarchical precision@{topk} of top {topk} images using method 2 : {mhp2_k/(cnt+1)}")

                print(
                    f"\nAverage inference time: {avg_inference_time/(cnt+1)}")
                print(
                    f"\nAverage time taken to find top {topk} images by method 1 : {avg_similarity_time_1/(cnt+1)}")
                print(
                    f"\nAverage time taken to find top {topk} images by method 2 : {avg_similarity_time_2/(cnt+1)}")
                print(
                    f"Mean Balanced classification accuracy by method 1 of {topk} retrieved images : {mbal_acc_1/(cnt+1)}")
                print(
                    f"Mean Balanced classification accuracy by method 2 of {topk} retrieved images : {mbal_acc_2/(cnt+1)}")
                x = np.array([1, 5, 10, 15, 20, 25, 50, 100, 150, 200, topk])
                y1 = np.array([mhp1_1/(cnt+1), mhp1_5/(cnt+1), mhp1_10/(cnt+1), mhp1_15/(cnt+1), mhp1_20/(cnt+1), mhp1_25/(cnt+1), mhp1_50/(cnt+1), mhp1_100/(cnt+1), mhp1_150/(cnt+1), mhp1_200/(cnt+1), mhp1_k/(cnt+1)])
                y2 = np.array([map1_1/(cnt+1), map1_5/(cnt+1), map1_10/(cnt+1), map1_15/(cnt+1), map1_20/(cnt+1), map1_25/(cnt+1), map1_50/(cnt+1), map1_100/(cnt+1), map1_150/(cnt+1), map1_200/(cnt+1), map1_k/(cnt+1)])
                fig, axs = plt.subplots(nrows=1, ncols=2)

                axs[0].plot(x, y1, color='red', linestyle='--', label='My line')
                axs[0].legend()
                axs[0].set_xlabel('X Label')
                axs[0].set_ylabel('Y Label 1')
                axs[0].set_title('Graph 1')
                axs[0].legend()
                # axs[0].set_xlim(0, 6)
                # axs[0].set_ylim(0, 12)

                axs[1].plot(x, y2, color='blue', linestyle=':', label='My other line')
                axs[1].legend()
                axs[1].set_xlabel('X Label')
                axs[1].set_ylabel('Y Label 2')
                axs[1].set_title('Graph 2')
                axs[1].legend()
                # axs[0].set_xlim(8, 15)
                # axs[0].set_ylim(0, 12)
                
                plt.show(block = True)

            
            cnt += 1

        # plot graphs on pyplot and numpy
        