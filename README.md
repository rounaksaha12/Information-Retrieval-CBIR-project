# Content-Based Image Retrieval using Binary Hashes

This repository contains the implementation of **Algorithm 2** and **Algorithm 2 + hashes**.  

**Algorithm 2** is retrieval using the multiplication of 100-dimensional embeddings of corpus images and query image [ matrix multiplication with a vector ] and retrieving the *top k* images.  

**Algorithm 2 + hashes** is shortlisting *top 2k* images using *hashes* followed by performing Algorithm 2 on the shortlisted pool of images.  

The details of both the algorithms are described in the **Report**.  

The training and testing was carried out using the [**The CIFAR-100 dataset**][1] :

## 0. Setup

After cloning this repository,create **embeddings** and **models** folder, download the **embeddings** and **models** folder from [**here**][2], and extract their contents into the **embeddings** and **models** folder in the directory respectively.

## 1. Quickrun

Before running any scripts ensure that you have the required dependencies. To install the dependencies, use:

```text
    virtualenv venv
    source venv/bin/activate
    pip install -r requirements.txt
```

### 1.1. All Test Results

For computing the test results for all 10000 test images in the CIFAR-100 dataset, use:

```python
    python test_model.py \
    --k <value of top k> \
    --dataset <test/train >
```

The dataset argument can be used to specify whether to retrieve from **test/train** images of the **CIFAR-100** dataset. The results include the performance metrics obtained and the predictions of *top k* images along with similarity scores both by **Algorithm 2** and **Algorithm 2 +  hashes**. A graph named 'plot.png' is created displaying variation of mAP and mAHP versus k.
<!-- 
### 1.2. Interactive Demo

For an interactive demo, use:

```python
    python interface.py
``` -->

## 2. Longrun

Before running any scripts ensure that you have the required dependencies. To install the dependencies, use:

```text
    virtualenv venv
    source venv/bin/activate
    pip install -r requirements.txt
```

### 2.1. Train Model

To finetune the model, use:

```
make finetune
```


### 2.2. Compute Image Embeddings

To compute the image embeddings of the <test/train> images of the CIFAR-100 dataset, use:

```python
    python get_embeddings.py --dataset <test/train>    
```

The embeddings are stored in folder .\embeddings\\<test/train>

## 3. Model Performance

### 3.1. All Test Images

|Method               | Inference time | Retrieval time | mAP@1 | mAP@5 | mAP@100 | map@250 | mAHP@1 | mAHP@5 | mAHP@100 | mAHP@250 |
|---------------------| ---------------|----------------| ------|-------|---------|---------|--------|--------|----------|----------|
|Algo 2               | 0.22s        | 4.67ms          | 1.0      | 0.5646      | 0.2664        | 0.1321        | 1.0       | 0.8865       | 0.7916         | 0.7942         |
|Algo 2 + hashes      | 0.22s               | 4.48ms          | 1.0      | 0.5617      | 0.2559        | 0.1236        | 1.0       | 0.8865       | 0.7838         | 0.7786         |  

[1]:https://www.cs.toronto.edu/~kriz/cifar.html  

[2]:https://drive.google.com/drive/folders/1E7P3qUCizB3ENI4y-cQul9xO3CWetgFI?usp=sharing
