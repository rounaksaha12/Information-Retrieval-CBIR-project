BATCH_SIZE=8
EPOCHS=10
LR=0.001
FINETUNE=False

# still have to activate the environment before running this command
run:	python3 resnet-finetune.py --batch_size ${BATCH_SIZE} --epochs ${EPOCHS} --lr ${LR} --finetune ${FINETUNE} | tee -a out-resnet-cifar10-transfer-learn.log