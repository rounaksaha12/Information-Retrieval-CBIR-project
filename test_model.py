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

        start_complete = time.process_time()
        embed_100_pdt = torch.matmul(trained_img_embed_100, embed_100)

        _, top1k_indices = torch.topk(embed_100_pdt.flatten(), k=topk)
        end_complete = time.process_time()

        print(
            f'Time to compute similarity between 100-d embeddings of all images : {end_complete - start_complete}')
        similarity_time_1 = (end_complete - start_complete)

        print("\nTop similar images based on 100-d embeddings")

        p1_1, p1_5, p1_10, p1_15, p1_20 = [0.0, 0.0, 0.0, 0.0, 0.0]
        hp1_1, hp1_5, hp1_10, hp1_15, hp1_20 = [0.0, 0.0, 0.0, 0.0, 0.0]

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

        sorted_class_sim = sorted(class_similar, reverse=True)

        s1, s2 = [0.0, 0.0]

        for i in range(20):

            # s1 += sorted_class_sim[i].item()

            s2 += class_similar[i].item()
            # if s1 == 0:
            #     deno = 1
            # else:
            #     deno = s1

            if i == 0:
                hp1_1 += s2
            if i == 4:
                hp1_5 += s2/5
            if i == 9:
                hp1_10 += s2/10
            if i == 14:
                hp1_15 += s2/15
            if i == 19:
                hp1_20 += s2/20

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

        start_hash = time.process_time()
        # embed_hash_pdt = torch.mm(torch.bitwise_and(trained_img_hashes.unsqueeze(2), hash.T.unsqueeze(0)), torch.ones(50000, 1, 1)).squeeze()
        # print(trained_img_hashes)
        # print(hash)
        # embed_hash_pdt = torch.squeeze(embed_hash_pdt)

        # print(-embed_hash_pdt)
        _, top2k_indices = torch.topk(-torch.cdist(trained_img_hashes,
                                      torch.unsqueeze(hash, 0), p=0).flatten(), k=topk)

        mask = torch.zeros_like(trained_img_embed_100)

        mask[top2k_indices, :] = torch.ones(
            (len(top2k_indices), mask.shape[1]))

        new_trained_img_embed_100 = trained_img_embed_100*mask

        img_hash_embeds_pdt = torch.matmul(
            new_trained_img_embed_100, embed_100)

        _, top3k_indices = torch.topk(img_hash_embeds_pdt.flatten(), k=topk)

        end_hash = time.process_time()
        similarity_time_2 = (end_hash - start_hash)

        p2_1, p2_5, p2_10, p2_15, p2_20 = [0.0, 0.0, 0.0, 0.0, 0.0]
        hp2_1, hp2_5, hp2_10, hp2_15, hp2_20 = [0.0, 0.0, 0.0, 0.0, 0.0]

        embed_hash_pdt = torch.cdist(
            trained_img_hashes, torch.unsqueeze(hash, 0), p=0)

        print(
            f'Time to compute similarity between hash embeddings of all and 100-d embeddings of the top {2*topk} images : {end_hash - start_hash}')

        print("\nTop similar images based on 100-d embeddings and hash embeddings")
        for i in range(topk):
            print(
                f"Similarity:{img_hash_embeds_pdt[top3k_indices[i]]} Label:{ds_label['fine_label_names'][trained_img_labels[top3k_indices[i]]]}")
        class_similar = []
        print(
            "\nTop similar images based on hash embeddings [least hamming distance at top]")
        for i in range(topk):
            print(
                f"Hamming distance:{embed_hash_pdt[top2k_indices[i]].item()} Label:{ds_label['fine_label_names'][trained_img_labels[top2k_indices[i]]]}")

            res = (trained_img_labels[top2k_indices[i]] == label)
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
            x = class_embeds['embedding'][trained_img_labels[top2k_indices[i]]]
            y = class_embeds['embedding'][label]
            class_similar.append(cosine_similarity(
                x.reshape(1, -1), y.reshape(1, -1)))

        sorted_class_sim = sorted(class_similar, reverse=True)

        s1, s2 = [0.0, 0.0]
        for i in range(20):
            # s1 += sorted_class_sim[i].item()
            s2 += class_similar[i].item()
            # if s1 == 0:
            #     deno = 1
            # else:
            #     deno = s1
            if i == 0:
                hp2_1 += s2
            if i == 4:
                hp2_5 += s2/5
            if i == 9:
                hp2_10 += s2/10
            if i == 14:
                hp2_15 += s2/15
            if i == 19:
                hp2_20 += s2/20

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

        with open('embeddings/hashes_CIFAR-100.pickle', 'rb') as f:
            trained_img_hashes = pickle.load(f)

        trained_img_hashes = torch.from_numpy(trained_img_hashes)
        trained_img_hashes = torch.where(trained_img_hashes >= 0.5, torch.ones_like(
            trained_img_hashes), torch.zeros_like(trained_img_hashes))

        # F.normalize(trained_img_hashes, p=2, dim=1, out=trained_img_hashes)

        with open('embeddings/id2imgs_CIFAR-100.pickle', 'rb') as f:
            trained_img_labels = pickle.load(f)

        with open('embeddings/embeddings_100_CIFAR-100.pickle', 'rb') as f:
            trained_img_embed_100 = pickle.load(f)

        trained_img_embed_100 = torch.from_numpy(trained_img_embed_100)
        F.normalize(trained_img_embed_100, p=2,
                    dim=1, out=trained_img_embed_100)

        with open('embeddings/embeddings_4096_CIFAR-100.pickle', 'rb') as f:
            trained_img_embed_4096 = pickle.load(f)

        trained_img_embed_4096 = torch.from_numpy(trained_img_embed_4096)
        F.normalize(trained_img_embed_4096, p=2,
                    dim=1, out=trained_img_embed_4096)

        with open('data/cifar-100-python/meta', 'rb') as f:
            ds_label = pickle.load(f)

        with open('embeddings/cifar100.unitsphere.pickle', 'rb') as f:
            class_embeds = pickle.load(f)

        model, input_size = modelNew.initialize_alexnet(0, 100, 48)
        model.load_state_dict(torch.load(
            'embeddings/CIFAR-100-alexnet-finetune.zip', map_location=torch.device('cpu')))
        model.eval()

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
