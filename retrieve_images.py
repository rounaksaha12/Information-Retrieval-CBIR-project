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


def retrieve_images(embed_100, embed_4096, hash, img_dataset_embed_100, img_dataset_embed_4096, img_dataset_hashes, img_dataset_labels, ds_label=None, label=None, topk=20):

    with torch.no_grad():
        if label is not None:
            print("Query image label : ", ds_label['fine_label_names'][label])

        start_complete = time.process_time()
        embed_100_pdt = torch.matmul(img_dataset_embed_100, embed_100)

        _, top1k_indices = torch.topk(embed_100_pdt.flatten(), k=topk)
        end_complete = time.process_time()

        print(
            f'Time to compute similarity between 100-d embeddings of all images : {end_complete - start_complete}')
        similarity_time_1 = (end_complete - start_complete)

        print("\nTop similar images based on 100-d embeddings")

        for i in range(topk):
            print(
                f"Similarity:{embed_100_pdt[top1k_indices[i]]} Label:{ds_label['fine_label_names'][img_dataset_labels[top1k_indices[i]]]}")

        mask = torch.zeros_like(img_dataset_embed_100)

        start_hash = time.process_time()

        _, top2k_indices = torch.topk(-torch.cdist(img_dataset_hashes,
                                      torch.unsqueeze(hash, 0), p=0).flatten(), k=topk)

        mask[top2k_indices, :] = torch.ones(
            (len(top2k_indices), mask.shape[1]))

        new_img_dataset_embed_100 = img_dataset_embed_100*mask

        img_hash_embeds_pdt = torch.matmul(
            new_img_dataset_embed_100, embed_100)

        _, top3k_indices = torch.topk(img_hash_embeds_pdt.flatten(), k=topk)

        end_hash = time.process_time()
        similarity_time_2 = (end_hash - start_hash)

        embed_hash_pdt = torch.cdist(
            img_dataset_hashes, torch.unsqueeze(hash, 0), p=0)

        print(
            f'Time to compute similarity between hash embeddings of all and 100-d embeddings of the top {2*topk} images : {end_hash - start_hash}')

        print(
            "\nTop similar images based on hash embeddings [least hamming distance at top]")
        for i in range(topk):
            print(
                f"Hamming distance:{embed_hash_pdt[top2k_indices[i]].item()} Label:{ds_label['fine_label_names'][img_dataset_labels[top2k_indices[i]]]}")

        print("\nTop similar images based on 100-d embeddings and hash embeddings")
        for i in range(topk):
            print(
                f"Similarity:{img_hash_embeds_pdt[top3k_indices[i]]} Label:{ds_label['fine_label_names'][img_dataset_labels[top3k_indices[i]]]}")

        img_dataset_labels_1 = [img_dataset_labels[i] for i in top1k_indices]
        img_dataset_labels_2 = [img_dataset_labels[i] for i in top3k_indices]

    return img_dataset_labels_1, img_dataset_labels_2, similarity_time_1, similarity_time_2
