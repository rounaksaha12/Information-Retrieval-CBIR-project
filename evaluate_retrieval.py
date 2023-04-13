import torch
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import balanced_accuracy_score


def evaluate_retrieval(query_label, class_sim_matrix, retrieved_labels, k_list=[1, 5, 10, 15, 20, 25, 50, 100, 150, 200]):
    with torch.no_grad():
        with open('embeddings/cifar100.unitsphere.pickle', 'rb') as f:
            class_embeds = pickle.load(f)

        precision_array = [0]*(len(k_list))

        hp_array = [0]*(len(k_list))

        class_similar = []
        cor = 0
        for i in range(len(retrieved_labels)):
            res = (retrieved_labels[i] == query_label)
            cor += res
            if res == 1:
                for j in range(len(k_list)):
                    if i < k_list[j]:
                        precision_array[j] += cor/(i+1)

            x = class_embeds['embedding'][retrieved_labels[i]]
            y = class_embeds['embedding'][query_label]
            class_similar.append(cosine_similarity(
                x.reshape(1, -1), y.reshape(1, -1)))

        for i in range(len(k_list)):
            precision_array[i] /= k_list[i]
            print(f"Average Precision@{k_list[i]}: {precision_array[i]}")

        s2 = 0.0
        deno = 0.0
        y_pred = []
        y_true = []
        csum = 0.0
        for i in range(len(retrieved_labels)):
            y_pred.append(retrieved_labels[i])
            csum += class_similar[i].item()
            deno += class_sim_matrix[query_label][i//100].item()
            if class_sim_matrix[query_label][i//100].item() == class_similar[i].item():
                y_true.append(retrieved_labels[i])
            else:
                y_true.append(100)
            for j in range(len(k_list)):
                if i < k_list[j]:
                    hp_array[j] += csum/deno
                if i == k_list[j]-1:
                    hp_array[j] /= k_list[j]
                    print(
                        f"Average Hierarchical Precision@{k_list[j]}: {hp_array[j]}")

        # compute balanced accuracy for the classes
        bal_acc = balanced_accuracy_score(y_true, y_pred)

        return precision_array, hp_array, bal_acc
