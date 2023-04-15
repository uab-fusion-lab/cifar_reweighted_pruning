export CUDA_VISIBLE_DEVICES="0"

python -u main.py --arch resnet \
					   --depth 18 \
					   --batch-size 256 \
					   --no-tricks \
					   --rew \
					   --masked-grow \
					   --sparsity-type irregular \
					   --epoch 200 \
					   --optmzr adam \
					   --lr 0.001 \
					   --lr-scheduler cosine \
					   --warmup \
					   --warmup-epochs 5 \
					   --mixup \
					   --alpha 0.3 \
					   --smooth \
					   --smooth-eps 0.1 &&
echo "Congratus! Finished rew training!"
