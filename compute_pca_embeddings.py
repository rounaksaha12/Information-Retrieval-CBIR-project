# take the image embeddings from the embeddings_CIFAR-100.pickle file and perform pca upon them

# first load the embeddings_CIFAR-100.pickle file

# then perform pca on the embeddings
from sklearn.decomposition import PCA
import pickle
import numpy as np


def load_pickle_diplay():
    with open("rounak_embeddings/embeddings_CIFAR-100.pickle", "rb") as f:
        embeddings = pickle.load(f)
    print(embeddings)
    return embeddings


def perform_pca(embeddings, n_components=2):
    pca = PCA(n_components)
    pca.fit(embeddings)
    pca_embeddings = pca.transform(embeddings)
    return pca_embeddings


embeddings = load_pickle_diplay()
# print dimensions of embeddings
print(embeddings.shape)
pca_embeddings = perform_pca(embeddings)
print(pca_embeddings.shape)
print(pca_embeddings)
# store the pca_embeddings in a pickle file
with open("rounak_embeddings/pca_embeddings.pickle", "wb") as f:
    pickle.dump(pca_embeddings, f)