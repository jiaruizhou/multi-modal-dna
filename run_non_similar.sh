set -e
unset RANK
export CUDA_VISIBLE_DEVICES="0,1,2,3"


MODEL=train_vit_freeze_bert_debug
DATA="non_similar"
CKPT_NUM=0
mkdir -p ./output/$MODEL/$DATA 

python -m torch.distributed.launch --nproc_per_node 4 fuse_main_finetune.py \
    --lr 1e-4 \
    --warmup_epochs 1 \
    --epochs 10 \
    --batch_size 600 \
    --accum_iter 1 \
    --data $DATA \
    --kmer 5 \
    --vit_model vit_base_patch4_5mer \
    --vit_resume ./output/output_b_p4_5mer/checkpoint-540.pth \
    --global_pooling_vit_bert \
    --global_pool_vit \
    --model $MODEL \
    --drop_path 0.2 \
    --smoothing 0.1 \
    --train_vit \
    --pooling_method cls_output \
    --amp --loss_scale \
    2>&1 | tee ./output/$MODEL/$DATA/training.log
