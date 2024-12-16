# set -e
# # train and test the fuse model
# unset RANK
# CUDA_VISIBLE_DEVICES="6" python fuse_main_finetune.py \
#     --batch_size 256 --model fuse --vit_model vit_base_patch4_5mer --kmer 5 \
#     --vit_resume ./output/output_b_p4_5mer/checkpoint-540.pth --smoothing 0.0 \
#     --epochs 20 --blr 1e-3 --layer_decay 0.75 --weight_decay 0.05 \
#     --drop_path 0.2 --reprob 0.25 --mixup 0 --cutmix 0 \
#     --data non_similar --clip_grad 2.0
# exit

# train and test the fuse model
unset RANK
export CUDA_VISIBLE_DEVICES="0,1,2,3"


# MODEL=freeze_vit_freeze_bert_proj_lr1e-4
# for DATA in "non_similar" "similar" "final"; do
#     mkdir -p ./output/$MODEL/$DATA 
#     python -m torch.distributed.launch --nproc_per_node 4 fuse_main_finetune.py \
#         --batch_size 600 --model $MODEL --accum_iter 1 --pooling_method cls_output \
#         --global_pooling_vit_bert --global_pool_vit \
#         --vit_model vit_base_patch4_5mer \
#         --vit_resume ./output/output_b_p4_5mer/checkpoint-540.pth --smoothing 0.1 \
#         --epochs 10 --blr 1e-3 --layer_decay 0.75 --weight_decay 0.05 --warmup_epochs 2\
#         --drop_path 0.2 --reprob 0.25 --mixup 0 --cutmix 0 \
#         --data $DATA 2>&1 | tee ./output/$MODEL/$DATA/training.log

    # python fuse_main_finetune.py \
    #     --batch_size 600 --model $MODEL \
    #     --eval --resume ./output/$MODEL/$DATA/checkpoint-$CKPT_NUM.pth \
    #     --global_pooling_vit_bert --global_pool_vit\
    #     --vit_model vit_base_patch4_5mer --kmer 5 \
    #     --vit_resume ./output/output_b_p4_5mer/checkpoint-540.pth --smoothing 0.1 \
    #     --epochs 8 --blr 1e-3 --layer_decay 0.75 --weight_decay 0.05 --warmup_epochs 2\
    #     --drop_path 0.2 --reprob 0.25 --mixup 0 --cutmix 0 \
    #     --data $DATA 
# done


MODEL=freeze_vit_freeze_bert_proj_lr1e-4
for DATA in "non_similar" "similar" "final"; do
    mkdir -p ./output/$MODEL/$DATA 
    python -m torch.distributed.launch --nproc_per_node 4 fuse_main_finetune.py \
        --batch_size 600 --model $MODEL --accum_iter 1 --pooling_method cls_output \
        --global_pooling_vit_bert --global_pool_vit --vit2bert_proj \
        --smoothing 0.1 \
        --epochs 10 --blr 1e-3 --layer_decay 0.75 --weight_decay 0.05 --warmup_epochs 2\
        --drop_path 0.2 --reprob 0.25 --mixup 0 --cutmix 0 \
        --data $DATA 2>&1 | tee ./output/$MODEL/$DATA/training.log
done



# MODEL=freeze_vit_train_bert_online
# for DATA in "non_similar" "similar" "final"; do
#     if [ $DATA == "non_similar" ]; then
#         CKPT_NUM=0
#     else
#         CKPT_NUM=7
#     fi
#     mkdir -p ./output/$MODEL/$DATA 
#     python -m torch.distributed.launch --nproc_per_node 8 fuse_main_finetune.py \
#         --batch_size 300 --model $MODEL --accum_iter 1 --train_bert --pooling_method cls_output \
#         --global_pooling_vit_bert --global_pool_vit \
#         --vit_model vit_base_patch4_5mer --kmer 5 \
#         --vit_resume ./output/output_b_p4_5mer/checkpoint-540.pth --smoothing 0.1 \
#         --epochs 8 --blr 1e-3 --layer_decay 0.75 --weight_decay 0.05 --warmup_epochs 2\
#         --drop_path 0.2 --reprob 0.25 --mixup 0 --cutmix 0 \
#         --data $DATA --correct_data --benchmark  2>&1 | tee ./output/$MODEL/$DATA/training.log

    # python fuse_main_finetune.py \
    #     --batch_size 600 --model $MODEL \
    #     --eval --resume ./output/$MODEL/$DATA/checkpoint-$CKPT_NUM.pth \
    #     --global_pooling_vit_bert --global_pool_vit\
    #     --vit_model vit_base_patch4_5mer --kmer 5 \
    #     --vit_resume ./output/output_b_p4_5mer/checkpoint-540.pth --smoothing 0.1 \
    #     --epochs 8 --blr 1e-3 --layer_decay 0.75 --weight_decay 0.05 --warmup_epochs 2\
    #     --drop_path 0.2 --reprob 0.25 --mixup 0 --cutmix 0 \
    #     --data $DATA 
# done

# python -m torch.distributed.launch --nproc_per_node 4 fuse_main_finetune.py \
#     --batch_size 300 --model fuse11 --train_vit --accum_iter 2\
#     --global_pooling_vit_bert --global_pool_vit\
#     --vit_model vit_base_patch4_5mer --kmer 5 \
#     --vit_resume ./output/output_b_p4_5mer/checkpoint-540.pth --smoothing 0.1 \
#     --epochs 8 --blr 1e-3 --layer_decay 0.75 --weight_decay 0.05 --warmup_epochs 2\
#     --drop_path 0.2 --reprob 0.25 --mixup 0 --cutmix 0 \
#     --data non_similar 2>&1 | tee ./output/fuse11/non_similar/training.log

# python fuse_main_finetune.py \
#     --batch_size 600 --model fuse11 --train_vit --accum_iter 2\
#     --eval --resume ./output/fuse11/non_similar/checkpoint-0.pth \
#     --global_pooling_vit_bert --global_pool_vit\
#     --vit_model vit_base_patch4_5mer --kmer 5 \
#     --vit_resume ./output/output_b_p4_5mer/checkpoint-540.pth --smoothing 0.1 \
#     --epochs 8 --blr 1e-3 --layer_decay 0.75 --weight_decay 0.05 --warmup_epochs 2\
#     --drop_path 0.2 --reprob 0.25 --mixup 0 --cutmix 0 \
#     --data non_similar 

# python -m torch.distributed.launch --nproc_per_node 4 fuse_main_finetune.py \
#     --batch_size 300 --model fuse11 --train_vit --accum_iter 2\
#     --global_pooling_vit_bert --global_pool_vit\
#     --vit_model vit_base_patch4_5mer --kmer 5 \
#     --vit_resume ./output/output_b_p4_5mer/checkpoint-540.pth --smoothing 0.1 \
#     --epochs 8 --blr 1e-3 --layer_decay 0.75 --weight_decay 0.05 --warmup_epochs 2\
#     --drop_path 0.2 --reprob 0.25 --mixup 0 --cutmix 0 \
#     --data similar 2>&1 | tee ./output/fuse11/similar/training.log

# python fuse_main_finetune.py \
#     --batch_size 600 --model train_vit_freeze_bert_online --train_vit --accum_iter 1\
#     --eval --resume ./output/train_vit_freeze_bert_online/non_similar/checkpoint-7.pth \
#     --global_pooling_vit_bert --global_pool_vit\
#     --vit_model vit_base_patch4_5mer --kmer 5 \
#     --vit_resume ./output/output_b_p4_5mer/checkpoint-540.pth --smoothing 0.1 \
#     --epochs 8 --blr 1e-3 --layer_decay 0.75 --weight_decay 0.05 --warmup_epochs 2\
#     --drop_path 0.2 --reprob 0.25 --mixup 0 --cutmix 0 \
#     --data non_similar 

# python -m torch.distributed.launch --nproc_per_node 4 fuse_main_finetune.py \
#     --batch_size 300 --model fuse11 --train_vit --accum_iter 2\
#     --global_pooling_vit_bert  --global_pool_vit \
#     --vit_model vit_base_patch4_5mer --kmer 5 \
#     --vit_resume ./output/output_b_p4_5mer/checkpoint-540.pth --smoothing 0.1 \
#     --epochs 8 --blr 1e-3 --layer_decay 0.75 --weight_decay 0.05 --warmup_epochs 2\
#     --drop_path 0.2 --reprob 0.25 --mixup 0 --cutmix 0 \
#     --data final 2>&1 | tee ./output/fuse11/final/training.log

# python fuse_main_finetune.py \
#     --batch_size 600 --model fuse11 --train_vit \
#     --eval --resume ./output/fuse11/final/checkpoint-7.pth \
#     --global_pooling_vit_bert --global_pool_vit \
#     --vit_model vit_base_patch4_5mer --kmer 5 \
#     --vit_resume ./output/output_b_p4_5mer/checkpoint-540.pth --smoothing 0.1 \
#     --epochs 8 --blr 1e-3 --layer_decay 0.75 --weight_decay 0.05 --warmup_epochs 2\
#     --drop_path 0.2 --reprob 0.25 --mixup 0 --cutmix 0 \
#     --data final 

# # python fuse_main_finetune.py \
# #     --eval --resume ./output/fuse5/non_similar/checkpoint-49.pth \
# #     --batch_size 600 --model fuse5 --vit_model vit_base_patch4_5mer --kmer 5\
# #     --vit_resume ./output/output_b_p4_5mer/checkpoint-540.pth \
# #     --drop_path 0.2 --reprob 0.25 --mixup 0 --cutmix 0 \
# #     --data non_similar

