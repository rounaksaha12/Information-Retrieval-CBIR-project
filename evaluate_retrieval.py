import torch
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import balanced_accuracy_score

def evaluate_retrieval(query_label, class_sim_matrix, retrieved_labels, img_dataset_labels,k_list=[1, 5, 10, 15, 20]):
    with torch.no_grad():
        with open('embeddings/cifar100.unitsphere.pickle', 'rb') as f:
            class_embeds = pickle.load(f)

        precision_array = [0]*(len(k_list))

        hp_array = [0]*(len(k_list))

        class_similar = []
        for i in range(len(retrieved_labels)):
            res = (retrieved_labels[i] == query_label)
            for j in range(len(k_list)):
                if i < k_list[j]:
                    precision_array[j] += res

            x = class_embeds['embedding'][retrieved_labels[i]]
            y = class_embeds['embedding'][query_label]
            class_similar.append(cosine_similarity(
                x.reshape(1, -1), y.reshape(1, -1)))
            

        for i in range(len(k_list)):
            precision_array[i] /= k_list[i]
            print(f"Precision@{k_list[i]}: {precision_array[i]}")

        # now go through entire corpus of image class embeddings and find cosine similarity of this class embeddings
        # so need id2imgs in argument
        # all_dataset_class_sim = []
        # for label in img_dataset_labels:
        #     x = class_embeds['embedding'][label]
        #     y = class_embeds['embedding'][query_label]
        #     all_dataset_class_sim.append(cosine_similarity(
        #         x.reshape(1, -1), y.reshape(1, -1))) 
        
        # sort the all_dataset_class_sim
        # all_dataset_class_sim.sort(reverse=True)
        
        s2 = 0.0
        # deno = 0.0
        for i in range(len(retrieved_labels)):
            s2 = class_similar[i].item()
            # deno += all_dataset_class_sim[i].item()
            for j in range(len(k_list)):
                if i < k_list[j]:
                    hp_array[j] += s2
                # if i == k_list[j]-1:
                #     hp_array[j] /= deno
                #     print(f"Hierarchical Precision@{k_list[j]}: {hp_array[j]}")
                             
        for i in range(len(k_list)):
            hp_array[i] /= k_list[i]
            
        
        # bal_acc = balanced_accuracy_score(query_label,retrieved_labels)
        bal_acc = 0


        return precision_array, hp_array, bal_acc
