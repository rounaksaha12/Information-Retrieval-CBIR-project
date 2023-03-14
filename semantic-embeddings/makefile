class_embed:	*.*
		python3 compute_class_embedding.py --hierarchy Cifar-Hierarchy/cifar.parent-child.txt --out embeddings/cifar10.unitsphere.pickle
sem_embed:	*.*
		export TF_CPP_MIN_LOG_LEVEL=2
		python3 learn_image_embeddings.py --dataset CIFAR-10 --data_root ./cifar-10-batches-py --embedding embeddings/cifar10.unitsphere.pickle --architecture simple --max_decay 0.1 --epochs 5 --model_dump cifar10-embedding.model.h5 --feature_dump cifar10-features.pickle

evaluate:	*.*
		python evaluate_retrieval.py --dataset CIFAR-10 --data_root ./cifar-10-batches-py --hierarchy Cifar-Hierarchy/cifar.parent-child.txt --classes_from embeddings/cifar10.unitsphere.pickle --feat cifar10-features.pickle  --label "Semantic Embeddings"
