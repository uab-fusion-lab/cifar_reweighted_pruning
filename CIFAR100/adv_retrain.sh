export CUDA_VISIBLE_DEVICES="0"

python -u adv_main.py --arch vgg \
					   --depth 16 \
					   --batch-size 256 \
					   --masked-retrain \
					   --no-tricks \
					   --sparsity-type filter \
					   --epoch 600 \
					   --optmzr sgd \
					   --lr 0.1 \
					   --adv \
					   --lr-scheduler default \
					   --config-file config_vgg16v1 \
					   --warmup-epochs 0  &&
echo "Congratus! Finished retraining!"