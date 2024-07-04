CUDA_VISIBLE_DEVICES=1 python ../train_finetune.py \
    --test_envs 0 1 3\
    --dataset DomainNet \
    --batch_size 32 \
    --lr 0.05 \
    --val_fraction 1.00 \
    --holdout_fraction 0.10 \
    --epochs 51 \
    --resume /data/home/limengze/WH/OOD/NewAblaExp/shared_models/unsuperPretrain/dmae_leave013_depth1_1/checkpoint-500.pth \
    --augment