export CUDA_VISIBLE_DEVICES="0"

python -u adv_pre_train.py --arch resnet \
					   --depth 18 \
					   --batch-size 256 \
					   --epoch 200 \
					   --optmzr sgd \
					   --lr 0.1 \
					   --lr-scheduler cosine \
					   --warmup-epochs 0 \
					   --alpha 0.0 \
					   --smooth-eps 0.0 &&
echo "Congratus! Finished pre training!"