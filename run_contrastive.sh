unset RANK
export CUDA_VISIBLE_DEVICES="0,1,2,3"

LABEL_RANK=1
NAME=bert
CKP=4
# PRETRAIN_MODEL=$NAME-bce-$LABEL_RANK
# MODEL=$PRETRAIN_MODEL
MODEL=$NAME-bce-$LABEL_RANK-10epoch

for DATA in "non_similar" "similar" "final"; do

    mkdir -p ./contrastive_output/$MODEL/$DATA 

    # python -m torch.distributed.launch --nproc_per_node 4 contrastive_main.py --batch_size 96 --accum_iter 1 --pooling_method cls_output --epochs 10 \
    #     --output_dir ./contrastive_output/ --name $NAME --cross_process_negatives --label_rank $LABEL_RANK --loss_fn clip_loss \
    #     --global_pooling_vit_bert --global_pool_vit --model $MODEL --lr 1e-4 --blr 1e-3 --layer_decay 0.75 --weight_decay 0.05 --warmup_epochs 1 --drop_path 0.2 --reprob 0.25 \
    #     --data $DATA 2>&1 | tee ./contrastive_output/$MODEL/$DATA/training.log

    python -m torch.distributed.launch --nproc_per_node 4 retrieval.py --get_encode --cross_gpu_sample --model $MODEL --name $NAME --test_ckp $CKP --retrieval_resume ./contrastive_output/$MODEL/$DATA/checkpoint-$CKP.pth \
        --batch_size 1000 --global_pooling_vit_bert --global_pool_vit --vit2bert_proj --data $DATA  2>&1 | tee ./contrastive_output/$MODEL/$DATA/retrieval_get_encode.log

    python retrieval.py --single_gpu --model $MODEL --name $NAME --test_ckp $CKP --global_pooling_vit_bert --global_pool_vit --vit2bert_proj \
        --data $DATA  2>&1 | tee ./contrastive_output/$MODEL/$DATA/retrieval.log

done

MODEL=$NAME-bce-$LABEL_RANK-10epoch-mix
for DATA in "non_similar" "similar" "final"; do

    mkdir -p ./contrastive_output/$MODEL/$DATA 

    # python -m torch.distributed.launch --nproc_per_node 4 contrastive_main.py --batch_size 96 --accum_iter 1 --pooling_method cls_output --epochs 10 \
    #     --output_dir ./contrastive_output/ --name $NAME --cross_process_negatives --label_rank $LABEL_RANK --loss_fn clip_loss \
    #     --global_pooling_vit_bert --global_pool_vit --model $MODEL --lr 1e-4 --blr 1e-3 --layer_decay 0.75 --weight_decay 0.05 --warmup_epochs 1 --drop_path 0.2 --reprob 0.25 \
    #     --data $DATA 2>&1 | tee ./contrastive_output/$MODEL/$DATA/training.log

    python -m torch.distributed.launch --nproc_per_node 4 retrieval.py --get_encode --cross_gpu_sample --model $MODEL --name $NAME --test_ckp $CKP --retrieval_resume ./contrastive_output/$MODEL/$DATA/checkpoint-$CKP.pth \
        --batch_size 1000 --global_pooling_vit_bert --global_pool_vit --vit2bert_proj --data $DATA  2>&1 | tee ./contrastive_output/$MODEL/$DATA/retrieval_get_encode.log

    python retrieval.py --single_gpu --model $MODEL --name $NAME --test_ckp $CKP --global_pooling_vit_bert --global_pool_vit --vit2bert_proj \
        --data $DATA  2>&1 | tee ./contrastive_output/$MODEL/$DATA/retrieval.log

done
# python retrieval.py --single_gpu \
#         --model $MODEL --name vit \
#         --batch_size 800 --global_pooling_vit_bert --global_pool_vit --vit2bert_proj \
#         --data non_similar  2>&1 | tee ./contrastive_output/$MODEL/$DATA/retrieval.log

# PRETRAIN_MODEL=train_vit_freeze_bert_lr1e-4
# MODEL=$PRETRAIN_MODEL-cls
# for DATA in "similar" "non_similar" "final"; do
#     if [ $DATA == "non_similar" ]; then
#         CKPT_NUM=0
#     else
#         CKPT_NUM=7
#     fi
#     mkdir -p ./contrastive_output/$MODEL/$DATA 
#     python -m torch.distributed.launch --nproc_per_node 4 contrastive_cls_main.py \
#         --batch_size 300 --accum_iter 1 --vit2bert_proj  --lr 1e-4 --pooling_method cls_output --epochs 20 \
#         --output_dir ./contrastive_output/ \
#         --contrastive_resume ./contrastive_output/$PRETRAIN_MODEL/$DATA/checkpoint-2.pth \
#         --global_pooling_vit_bert --global_pool_vit \
#         --model $MODEL \
#         --blr 1e-3 --layer_decay 0.75 --weight_decay 0.05 --warmup_epochs 1 \
#         --drop_path 0.2 --reprob 0.25 \
#         --data $DATA 2>&1 | tee ./contrastive_output/$MODEL/$DATA/training.log
# done

# MODEL=CONTRAST_train_vit_freeze_bert_lr1e-3
# for DATA in "similar" "non_similar" "final"; do
#     if [ $DATA == "non_similar" ]; then
#         CKPT_NUM=0
#     else
#         CKPT_NUM=7
#     fi
#     mkdir -p ./contrastive_output/$MODEL/$DATA 
#     python -m torch.distributed.launch --nproc_per_node 4 contrastive_main.py \
#         --batch_size 300 --accum_iter 1 --vit2bert_proj  --lr 1e-3 --pooling_method cls_output --epochs 20 \
#         --global_pooling_vit_bert --global_pool_vit \
#         --model $MODEL \
#         --blr 1e-3 --layer_decay 0.75 --weight_decay 0.05 --warmup_epochs 1 \
#         --drop_path 0.2 --reprob 0.25 \
#         --data $DATA 2>&1 | tee ./contrastive_output/$MODEL/$DATA/training.log
# done