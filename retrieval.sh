unset RANK
export CUDA_VISIBLE_DEVICES="0"

# PRETRAIN_MODEL=train_vit_freeze_bert_lr1e-4
# MODEL=$PRETRAIN_MODEL
# for DATA in "non_similar" "similar" "final"; do
#     mkdir -p ./contrastive_output/$MODEL/$DATA 
#     python -m torch.distributed.launch --nproc_per_node 3 retrieval.py \
#         --model $MODEL --name vit --contrastive_resume ./contrastive_output/$PRETRAIN_MODEL/$DATA/checkpoint-2.pth \
#         --batch_size 1000 --global_pooling_vit_bert --global_pool_vit --vit2bert_proj \
#         --data $DATA 
# done

PRETRAIN_MODEL=train_vit_freeze_bert_lr1e-4
MODEL=$PRETRAIN_MODEL
for DATA in "non_similar" "similar" "final" ; do
    mkdir -p ./contrastive_output/$MODEL/$DATA 
    # python -m torch.distributed.launch --nproc_per_node 4 retrieval.py \
    #     --get_encode --cross_gpu_sample --model $MODEL --name vit --contrastive_resume ./contrastive_output/$PRETRAIN_MODEL/$DATA/checkpoint-2.pth \
    #     --batch_size 1000 --global_pooling_vit_bert --global_pool_vit --vit2bert_proj \
    #     --data $DATA  2>&1 | tee ./contrastive_output/$MODEL/$DATA/retrieval_get_encode.log

    python retrieval.py --single_gpu \
        --model $MODEL --name vit --contrastive_resume ./contrastive_output/$PRETRAIN_MODEL/$DATA/checkpoint-2.pth \
        --batch_size 1000 --global_pooling_vit_bert --global_pool_vit --vit2bert_proj \
        --data $DATA  2>&1 | tee ./contrastive_output/$MODEL/$DATA/retrieval.log

done

for DATA in "non_similar" "similar" "final" ; do
    mkdir -p ./contrastive_output/$MODEL/$DATA 
    # python -m torch.distributed.launch --nproc_per_node 4 retrieval.py \
    #     --get_encode --cross_gpu_sample --model $MODEL --name mae --contrastive_resume ./contrastive_output/$PRETRAIN_MODEL/$DATA/checkpoint-2.pth \
    #     --batch_size 1000 --global_pooling_vit_bert --global_pool_vit --vit2bert_proj \
    #     --data $DATA  2>&1 | tee ./contrastive_output/$MODEL/$DATA/retrieval_get_encode.log

    python retrieval.py --single_gpu \
        --model $MODEL --name mae --contrastive_resume ./contrastive_output/$PRETRAIN_MODEL/$DATA/checkpoint-2.pth \
        --batch_size 1000 --global_pooling_vit_bert --global_pool_vit --vit2bert_proj \
        --data $DATA  2>&1 | tee ./contrastive_output/$MODEL/$DATA/retrieval.log

done

for DATA in "non_similar" "similar" "final" ; do
    mkdir -p ./contrastive_output/$MODEL/$DATA 
    # python -m torch.distributed.launch --nproc_per_node 4 retrieval.py \
    #     --get_encode --cross_gpu_sample --model $MODEL --name fuse --contrastive_resume ./contrastive_output/$PRETRAIN_MODEL/$DATA/checkpoint-2.pth \
    #     --batch_size 1000 --global_pooling_vit_bert --global_pool_vit --vit2bert_proj \
    #     --data $DATA  2>&1 | tee ./contrastive_output/$MODEL/$DATA/retrieval_get_encode.log

    python retrieval.py --single_gpu \
        --model $MODEL --name fuse --contrastive_resume ./contrastive_output/$PRETRAIN_MODEL/$DATA/checkpoint-2.pth \
        --batch_size 1000 --global_pooling_vit_bert --global_pool_vit --vit2bert_proj \
        --data $DATA  2>&1 | tee ./contrastive_output/$MODEL/$DATA/retrieval.log

done

for DATA in "non_similar" "similar" "final" ; do
    mkdir -p ./contrastive_output/$MODEL/$DATA 
    # python -m torch.distributed.launch --nproc_per_node 4 retrieval.py \
    #     --get_encode --cross_gpu_sample --model $MODEL --name bert --contrastive_resume ./contrastive_output/$PRETRAIN_MODEL/$DATA/checkpoint-2.pth \
    #     --batch_size 1000 --global_pooling_vit_bert --global_pool_vit --vit2bert_proj \
    #     --data $DATA  2>&1 | tee ./contrastive_output/$MODEL/$DATA/retrieval_get_encode.log

    python retrieval.py --single_gpu \
        --model $MODEL --name bert --contrastive_resume ./contrastive_output/$PRETRAIN_MODEL/$DATA/checkpoint-2.pth \
        --batch_size 1000 --global_pooling_vit_bert --global_pool_vit --vit2bert_proj \
        --data $DATA  2>&1 | tee ./contrastive_output/$MODEL/$DATA/retrieval.log

done
# PRETRAIN_MODEL=train_vit_freeze_bert_lr1e-4
# MODEL=$PRETRAIN_MODEL
# for DATA in "non_similar" "similar" "final"; do
#     mkdir -p ./contrastive_output/$MODEL/$DATA 
#     python -m torch.distributed.launch --nproc_per_node 3 retrieval.py \
#         --model $MODEL --name fuse --contrastive_resume ./contrastive_output/$PRETRAIN_MODEL/$DATA/checkpoint-2.pth \
#         --batch_size 1000 --global_pooling_vit_bert --global_pool_vit --vit2bert_proj \
#         --data $DATA 
# done

# PRETRAIN_MODEL=train_vit_train_bert_lr1e-4
# MODEL=$PRETRAIN_MODEL
# for DATA in "similar" "non_similar" "final"; do
#     mkdir -p ./contrastive_output/$MODEL/$DATA 
#     python -m torch.distributed.launch --nproc_per_node 4 retrieval.py \
#         --model $MODEL --contrastive_resume ./contrastive_output/$PRETRAIN_MODEL/$DATA/checkpoint-2.pth \
#         --batch_size 1000 --global_pooling_vit_bert --global_pool_vit --vit2bert_proj \
#         --data $DATA 
# done