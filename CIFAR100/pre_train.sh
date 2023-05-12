export CUDA_VISIBLE_DEVICES="0"

python -u adv_main.py --arch resnet \
             --pre-train \
					   --depth 18 \
					   --batch-size 256 \
					   --epoch 400 \
					   --optmzr sgd \
					   --lr 0.1 \
					   --adv \
					   --lr-scheduler default \
					   --warmup-epochs 0 \
					   --alpha 0.0 \
					   --smooth-eps 0.0 &&
echo "Congratus! Finished pre training!"