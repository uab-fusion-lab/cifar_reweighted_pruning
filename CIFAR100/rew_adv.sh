export CUDA_VISIBLE_DEVICES="0"

python -u adv_main.py --arch resnet \
					   --depth 18 \
					   --batch-size 256 \
					   --no-tricks \
					   --rew \
					   --sparsity-type filter \
					   --epoch 50 \
					   --optmzr sgd \
					   --adv \
					   --lr 0.001 \
					   --lr-scheduler default \
					   --combine-progressive\
					   --warmup-epochs 0 \
					   --alpha 0.0 \
					   --smooth-eps 0.1 &&
echo "Congratus! Finished rew training!"