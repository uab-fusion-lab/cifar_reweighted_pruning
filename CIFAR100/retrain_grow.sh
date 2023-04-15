export CUDA_VISIBLE_DEVICES="0"

python -u main.py --arch resnet \
					   --depth 18 \
					   --batch-size 256 \
					   --masked-retrain \
					   --masked-grow \
					   --sparsity-type irregular \
					   --epoch 300 \
					   --optmzr sgd \
					   --lr 0.1 \
					   --lr-scheduler cosine \
					   --warmup \
					   --warmup-epochs 8  &&
echo "Congratus! Finished rew training!"
