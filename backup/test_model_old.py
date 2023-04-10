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
import src.modelNew as modelNew
import trainNew
import pickle
import argparse


def retrieve_images(img, embed_100, embed_4096, hash, label, topk):

    with torch.no_grad():
        print("Query image label : ", ds_label['fine_label_names'][label])

        # compute cosine similarity between 100-d embeddings
        # convert trained_img_embed to a torch tensor matrix

        start_complete = time.process_time()
        embed_100_pdt = torch.matmul(trained_img_embed_100, embed_100)

        # get the topk elements of the pdt
        _, top1k_indices = torch.topk(embed_100_pdt.flatten(), k=topk)
        end_complete = time.process_time()

        # sim_100_d = []
        # for i in range(len(trained_img_labels)):
        #     sim_100_d.append(((F.cosine_similarity(
        #         embed_100, torch.from_numpy(trained_img_embed_100[i]))).numpy(), i))
        # sim_100_d.sort(key=lambda sim_100_d: sim_100_d[0], reverse=True)

        print(
            f'Time to compute similarity between 100-d embeddings of all images : {end_complete - start_complete}')
        similarity_time_1 = (end_complete - start_complete)

        print("\nTop similar images based on 100-d embeddings")

        p1_1, p1_5, p1_10, p1_15, p1_20 = [0.0, 0.0, 0.0, 0.0, 0.0]
        hp1_1, hp1_5, hp1_10, hp1_15, hp1_20 = [0.0, 0.0, 0.0, 0.0, 0.0]

        # find similarity of class embedding of image with that of query for top k images

        class_similar = []

        for i in range(topk):
            print(
                f"Similarity:{embed_100_pdt[top1k_indices[i]]} Label:{ds_label['fine_label_names'][trained_img_labels[top1k_indices[i]]]}")
            res = (trained_img_labels[top1k_indices[i]]
                   == label)
            if i < 1:
                p1_1 += res
            elif i < 5:
                p1_5 += res
            elif i < 10:
                p1_10 += res
            elif i < 15:
                p1_15 += res
            else:
                p1_20 += res

            x = class_embeds['embedding'][trained_img_labels[top1k_indices[i]]]
            y = class_embeds['embedding'][label]
            class_similar.append(cosine_similarity(
                x.reshape(1, -1), y.reshape(1, -1)))

        # sort the class_similar list and store in another list

        sorted_class_sim = sorted(class_similar, reverse=True)

        s1, s2 = [0.0, 0.0]

        for i in range(20):

            s1 += sorted_class_sim[i].item()

            s2 += class_similar[i].item()
            if s1 == 0:
                deno = 1
            else:
                deno = s1

            if i == 0:
                hp1_1 += s2/deno
            if i == 4:
                hp1_5 += s2/deno
            if i == 9:
                hp1_10 += s2/deno
            if i == 14:
                hp1_15 += s2/deno
            if i == 19:
                hp1_20 += s2/deno

        p1_5 /= 5
        p1_10 /= 10
        p1_15 /= 15
        p1_20 /= 20

        print(f'Precision@1 : {p1_1}')
        print(f'Precision@5 : {p1_5}')
        print(f'Precision@10 : {p1_10}')
        print(f'Precision@15 : {p1_15}')
        print(f'Precision@20 : {p1_20}')
        print(f'Hierarchical Precision@1 : {hp1_1}')
        print(f'Hierarchical Precision@5 : {hp1_5}')
        print(f'Hierarchical Precision@10 : {hp1_10}')
        print(f'Hierarchical Precision@15 : {hp1_15}')
        print(f'Hierarchical Precision@20 : {hp1_20}')

        # cosine similarity between hash embeddings

        start_hash = time.process_time()
        embed_hash_pdt = torch.matmul(trained_img_hashes, hash)
        _, top2k_indices = torch.topk(embed_hash_pdt.flatten(), k=2*topk)

        # whichever indices exist,except those,make all others rows in img_class_embeds to 0
        mask = torch.zeros_like(trained_img_embed_100)
        # print(top2k_indices)
        # Setting the rows corresponding to the indices to 1 in the mask tensor
        mask[top2k_indices, :] = torch.ones(
            (len(top2k_indices), mask.shape[1]))
        # print(mask)
        new_trained_img_embed_100 = trained_img_embed_100*mask
        # print(new_trained_img_embed_100)
        # F.normalize(img_class_embeds, out=img_class_embeds)
        # print(img_class_embeds)
        # print(img_class_embeds.shape)
        # print(type(img_class_embeds))
        # print(type(class_embeds['embedding'][label]))
        # print(type(torch.tensor(class_embeds['embedding'][label])))
        
        img_hash_embeds_pdt = torch.matmul(
            new_trained_img_embed_100, embed_100)

        # get topk values of img_class_embeds
        _, top3k_indices = torch.topk(img_hash_embeds_pdt.flatten(), k=topk)
        # sim_hash = []

        # for i in range(len(trained_img_labels)):
        #     sim_hash.append(((F.cosine_similarity(hash, torch.from_numpy(
        #         trained_img_hashes[i]))).numpy(), i))

        # # sort the list in descending order based on first element of tuple
        # sim_hash.sort(key=lambda sim_hash: sim_hash[0], reverse=True)

        # among the topk images take cosine similarity with their class embedding 100 dimensional value and compute top k

        # among top 2k images in hash embeddings,find topk with max cosine similarity with 100-d embeddings
        p2_1, p2_5, p2_10, p2_15, p2_20 = [0.0, 0.0, 0.0, 0.0, 0.0]
        hp2_1, hp2_5, hp2_10, hp2_15, hp2_20 = [0.0, 0.0, 0.0, 0.0, 0.0]

        # sim_100_d_hash = []

        # for i in range(2*topk):
        #     sim_100_d_hash.append(((F.cosine_similarity(
        #         trained_img_embed_100[topk_indices[i]], embed_100)).numpy(), topk_indices[i]))

        # # sort the list in descending order based on first element of tuple
        # sim_100_d_hash.sort(
        #     key=lambda sim_100_d_hash: sim_100_d_hash[0], reverse=True)

        end_hash = time.process_time()

        print("\nTop similar images based on hash embeddings")
        for i in range(topk):
            print(
                f"Similarity:{embed_hash_pdt[top2k_indices[i]]} Label:{ds_label['fine_label_names'][trained_img_labels[top2k_indices[i]]]}")

        similarity_time_2 = (end_hash - start_hash)
        class_similar = []

        print(
            f'Time to compute similarity between hash embeddings of all and 100-d embeddings of the top {2*topk} images : {end_hash - start_hash}')
        print("\nTop similar images based on 100-d embeddings and hash embeddings")
        for i in range(topk):
            print(
                f"Similarity:{img_hash_embeds_pdt[top3k_indices[i]]} Label:{ds_label['fine_label_names'][trained_img_labels[top3k_indices[i]]]}")
            res = (trained_img_labels[top3k_indices[i]] == label)
            if i < 1:
                p2_1 += res
            elif i < 5:
                p2_5 += res
            elif i < 10:
                p2_10 += res
            elif i < 15:
                p2_15 += res
            else:
                p2_20 += res
            x = class_embeds['embedding'][trained_img_labels[top3k_indices[i]]]
            y = class_embeds['embedding'][label]
            class_similar.append(cosine_similarity(
                x.reshape(1, -1), y.reshape(1, -1)))

         # sort the class_similar list and store in another list
        sorted_class_sim = sorted(class_similar, reverse=True)

        s1, s2 = [0.0, 0.0]
        for i in range(20):
            s1 += sorted_class_sim[i].item()
            s2 += class_similar[i].item()
            if s1 == 0:
                deno = 1
            else:
                deno = s1
            if i == 0:
                hp2_1 += s2/deno
            if i == 4:
                hp2_5 += s2/deno
            if i == 9:
                hp2_10 += s2/deno
            if i == 14:
                hp2_15 += s2/deno
            if i == 19:
                hp2_20 += s2/deno

        p2_5 /= 5
        p2_10 /= 10
        p2_15 /= 15
        p2_20 /= 20

        print(f'Precision@1 : {p2_1}')
        print(f'Precision@5 : {p2_5}')
        print(f'Precision@10 : {p2_10}')
        print(f'Precision@15 : {p2_15}')
        print(f'Precision@20 : {p2_20}')
        print(f'Hierarchical Precision@1 : {hp2_1}')
        print(f'Hierarchical Precision@5 : {hp2_5}')
        print(f'Hierarchical Precision@10 : {hp2_10}')
        print(f'Hierarchical Precision@15 : {hp2_15}')
        print(f'Hierarchical Precision@20 : {hp2_20}')

        # exit(1)

    return p1_1, p1_5, p1_10, p1_15, p1_20, p2_1, p2_5, p2_10, p2_15, p2_20, hp1_1, hp1_5, hp1_10, hp1_15, hp1_20, hp2_1, hp2_5, hp2_10, hp2_15, hp2_20, similarity_time_1, similarity_time_2


if __name__ == '__main__':
    with torch.no_grad():
        parser = argparse.ArgumentParser()
        parser.add_argument('--k', default=20, type=int,
                            help='Number of top similar images to retrieve')
        args = parser.parse_args()

        topk = args.k

        map1_1, map1_5, map1_10, map1_15, map1_20 = [0.0, 0.0, 0.0, 0.0, 0.0]
        map2_1, map2_5, map2_10, map2_15, map2_20 = [0.0, 0.0, 0.0, 0.0, 0.0]

        avg_inference_time = 0.0
        avg_similarity_time_1 = 0.0
        avg_similarity_time_2 = 0.0

        mhp1_1, mhp1_5, mhp1_10, mhp1_15, mhp1_20 = [0.0, 0.0, 0.0, 0.0, 0.0]
        mhp2_1, mhp2_5, mhp2_10, mhp2_15, mhp2_20 = [0.0, 0.0, 0.0, 0.0, 0.0]

        # extract hashes.pickle file
        with open('embeddings/hashes_CIFAR-100.pickle', 'rb') as f:
            trained_img_hashes = pickle.load(f)

        # make trained images data binary ,if entry >= 0.5 then 1 else 0
        # look for way to fasten up this ,should exist
        # for i, entry in enumerate(trained_img_hashes):
        #     for j, ele in enumerate(trained_img_hashes[i]):
        #         if ele >= 0.5:
        #             trained_img_hashes[i][j] = 1
        #         else:
        #             trained_img_hashes[i][j] = 0

        # print(trained_img_hashes)
        trained_img_hashes = torch.from_numpy(trained_img_hashes)
        trained_img_hashes = torch.where(trained_img_hashes >= 0.5, torch.ones_like(
            trained_img_hashes), torch.zeros_like(trained_img_hashes))

        F.normalize(trained_img_hashes, p=2, dim=1, out=trained_img_hashes)
        # print(trained_img_hashes)

        # extract id2imgs.pickle file
        with open('embeddings/id2imgs_CIFAR-100.pickle', 'rb') as f:
            trained_img_labels = pickle.load(f)

        # extract 100 d embeddings
        with open('embeddings/embeddings_100_CIFAR-100.pickle', 'rb') as f:
            trained_img_embed_100 = pickle.load(f)

        trained_img_embed_100 = torch.from_numpy(trained_img_embed_100)
        F.normalize(trained_img_embed_100, p=2,
                    dim=1, out=trained_img_embed_100)
        # extract 4096 dim img embeddings
        with open('embeddings/embeddings_4096_CIFAR-100.pickle', 'rb') as f:
            trained_img_embed_4096 = pickle.load(f)

        trained_img_embed_4096 = torch.from_numpy(trained_img_embed_4096)
        F.normalize(trained_img_embed_4096, p=2,
                    dim=1, out=trained_img_embed_4096)
        # extract the dataset label
        with open('data/cifar-100-python/meta', 'rb') as f:
            ds_label = pickle.load(f)

        # extract the class embeddings

        with open('embeddings/cifar100.unitsphere.pickle', 'rb') as f:
            class_embeds = pickle.load(f)
        # print(class_embeds)
        # make a 50000*100 tensor where each row is the class embedding of the ith image
        # img_class_embeds = torch.zeros((50000, 100))

        # Filling the tensor with the class embeddings for each image
        # for i in range(50000):
        #     img_class_embeds[i, :] = torch.tensor(
        #         class_embeds['embedding'][trained_img_labels[i]])

        # print(img_class_embeds)

        model, input_size = modelNew.initialize_alexnet(0, 100, 48)
        model.load_state_dict(torch.load(
            'embeddings/CIFAR-100-alexnet-finetune.zip', map_location=torch.device('cpu')))
        model.eval()
        # summary(model, (3, 224, 224))

        data_transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.486, 0.486], [0.229, 0.224, 0.225])])

        image_datasets = {
            'test': datasets.CIFAR100(
                root='./data',
                train=False,
                download=True,
                transform=data_transform
            )}

        cnt = 0
        for img, label in image_datasets["test"]:
            # start timer here
            start = time.process_time()
            img_new = torch.unsqueeze(img, 0)
            end = time.process_time()
            embed_4096, hash, embed_100 = model(img_new)
            hash = torch.squeeze(hash)
            embed_100 = torch.squeeze(embed_100)
            embed_4096 = torch.squeeze(embed_4096)
            hash = torch.where(hash >= 0.5, torch.ones_like(
                hash), torch.zeros_like(hash))
            # print(hash)
            F.normalize(hash, dim=0, out=hash)
            F.normalize(embed_100, dim=0, out=embed_100)
            F.normalize(embed_4096, dim=0, out=embed_4096)
            # print(embed_100)
            # print(hash)

            # hash = hash[0].numpy()
            # # binarise entries of hash
            # for i in range(len(hash)):

            #     if hash[i] >= 0.5:
            #         hash[i] = 1
            #     else:
            #         hash[i] = 0

            print(f'\nModel inference time : {end-start}')
            avg_inference_time += (end-start)
            p1_1, p1_5, p1_10, p1_15, p1_20, p2_1, p2_5, p2_10, p2_15, p2_20, hp1_1, hp1_5, hp1_10, hp1_15, hp1_20, hp2_1, hp2_5, hp2_10, hp2_15, hp2_20, similarity_time_1, similarity_time_2 = retrieve_images(
                img, embed_100, embed_4096, hash, label, topk)
            map1_1 += p1_1
            map1_5 += p1_5
            map1_10 += p1_10
            map1_15 += p1_15
            map1_20 += p1_20
            map2_1 += p2_1
            map2_5 += p2_5
            map2_10 += p2_10
            map2_15 += p2_15
            map2_20 += p2_20
            mhp1_1 += hp1_1
            mhp1_5 += hp1_5
            mhp1_10 += hp1_10
            mhp1_15 += hp1_15
            mhp1_20 += hp1_20
            mhp2_1 += hp2_1
            mhp2_5 += hp2_5
            mhp2_10 += hp2_10
            mhp2_15 += hp2_15
            mhp2_20 += hp2_20
            avg_similarity_time_1 += similarity_time_1
            avg_similarity_time_2 += similarity_time_2

            if (cnt+1) % 10 == 0:
                print(
                    f"\nAverage precision@1 of top {topk} images using method 1 : {map1_1/(cnt+1)}")
                print(
                    f"\nAverage precision@5 of top {topk} images using method 1 : {map1_5/(cnt+1)}")
                print(
                    f"\nAverage precision@10 of top {topk} images using method 1 : {map1_10/(cnt+1)}")
                print(
                    f"\nAverage precision@15 of top {topk} images using method 1 : {map1_15/(cnt+1)}")
                print(
                    f"\nAverage precision@20 of top {topk} images using method 1 : {map1_20/(cnt+1)}")
                print(
                    f"\nAverage precision@1 of top {topk} images using method 2 : {map2_1/(cnt+1)}")
                print(
                    f"\nAverage precision@5 of top {topk} images using method 2 : {map2_5/(cnt+1)}")
                print(
                    f"\nAverage precision@10 of top {topk} images using method 2 : {map2_10/(cnt+1)}")
                print(
                    f"\nAverage precision@15 of top {topk} images using method 2 : {map2_15/(cnt+1)}")
                print(
                    f"\nAverage precision@20 of top {topk} images using method 2 : {map2_20/(cnt+1)}")

                print(
                    f"\nAverage hierarchichal precision@1 of top {topk} images using method 1 : {mhp1_1/(cnt+1)}")
                print(
                    f"\nAverage hierarchichal precision@5 of top {topk} images using method 1 : {mhp1_5/(cnt+1)}")
                print(
                    f"\nAverage hierarchichal precision@10 of top {topk} images using method 1 : {mhp1_10/(cnt+1)}")
                print(
                    f"\nAverage hierarchichal precision@15 of top {topk} images using method 1 : {mhp1_15/(cnt+1)}")
                print(
                    f"\nAverage hierarchichal precision@20 of top {topk} images using method 1 : {mhp1_20/(cnt+1)}")
                print(
                    f"\nAverage hierarchichal precision@1 of top {topk} images using method 2 : {mhp2_1/(cnt+1)}")
                print(
                    f"\nAverage hierarchichal precision@5 of top {topk} images using method 2 : {mhp2_5/(cnt+1)}")
                print(
                    f"\nAverage hierarchichal precision@10 of top {topk} images using method 2 : {mhp2_10/(cnt+1)}")
                print(
                    f"\nAverage hierarchichal precision@15 of top {topk} images using method 2 : {mhp2_15/(cnt+1)}")
                print(
                    f"\nAverage hierarchichal precision@20 of top {topk} images using method 2 : {mhp2_20/(cnt+1)}")

                print(
                    f"\nAverage inference time: {avg_inference_time/(cnt+1)}")
                print(
                    f"\nAverage time taken to find top {topk} images by method 1 : {avg_similarity_time_1/(cnt+1)}")
                print(
                    f"\nAverage time taken to find top {topk} images by method 2 : {avg_similarity_time_2/(cnt+1)}")
            cnt += 1

        # hash embeddings made 0-1
        # matrix multiplication not performed yet
