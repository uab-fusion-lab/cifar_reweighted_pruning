export CUDA_VISIBLE_DEVICES="0"

python -u adv_main.py --arch vgg \
					   --depth 16 \
					   --batch-size 256 \
					   --no-tricks \
					   --check-model \
					   --sparsity-type filter \
					   --epoch 50 \
					   --optmzr sgd \
					   --lr 0.0001 \
					   --lr-scheduler default \
					   --combine-progressive \
					   --warmup-epochs 5 \
					   --alpha 0.0 \
					   --smooth-eps 0.1 &&
echo "Congratus! Finished rew training!"