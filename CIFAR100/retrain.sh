export CUDA_VISIBLE_DEVICES="0"

python -u main.py --arch resnet \
					   --depth 18 \
					   --batch-size 256 \
					   --masked-retrain \
					   --sparsity-type irregular \
					   --epoch 300 \
					   --optmzr sgd \
					   --lr 0.1 \
					   --lr-scheduler cosine \
					   --warmup \
					   --config-file config_resnet18v1\
					   --warmup-epochs 8  &&
echo "Congratus! Finished rew training!"
