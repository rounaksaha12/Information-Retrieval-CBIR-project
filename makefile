DATASET=CIFAR-100
BATCH_SIZE=8
EPOCHS=10
LR=0.001
HASH_SIZE=48
FINETUNE=True
DUMP=3
EMBED_4096_DUMP=./embeddings/train/embeddings_4096
EMBED_100_DUMP=./embeddings/train/embeddings_100
HASH_DUMP=hashes
IDMAP_DUMP=id2imgs
CLASS_EMBED=./embeddings/cifar100.unitsphere.pickle
K=250
TYPE=test
EMBED_TYPE=test

# still have to activate the environment before running this command
finetune:	
	python3 trainNew.py --dataset ${DATASET} --class_embed ${CLASS_EMBED} --batch_size ${BATCH_SIZE} --epochs ${EPOCHS} --lr ${LR} --finetune --save_model --dump ${DUMP} --embedding_4096_dump ${EMBED_4096_DUMP} --embedding_100_dump ${EMBED_100_DUMP} --hash_dump ${HASH_DUMP} --id2imgs_dump ${IDMAP_DUMP} | tee -a out-alexnet-${DATASET}-transfer-learn.log

feat_extract:	
	python3 trainNew.py --dataset ${DATASET} --class_embed ${CLASS_EMBED} --batch_size ${BATCH_SIZE} --epochs ${EPOCHS} --lr ${LR} --save_model --dump ${DUMP} --embedding_4096_dump ${EMBED_4096_DUMP} --embedding_100_dump ${EMBED_100_DUMP} --hash_dump ${HASH_DUMP} --id2imgs_dump ${IDMAP_DUMP} | tee -a out-alexnet-${DATASET}-transfer-learn.log

run_alexnet:	
	python3 trainNew.py --epochs 1 | tee -a out-alexnet-cifar10-transfer-learn.log

run_test:
	python3 test_model.py --k ${K} --dataset ${TYPE} | tee retrieval_metrics_calc.log

get_embeddings:
	python3 get_embeddings.py --dataset ${EMBED_TYPE}


