# 0
CUDA_VISIBLE_DEVICES=7 python retrieval.py --resume ./output_dir2/checkpoint-700.pth \
   --batch_size 2048 --model mae_vit_large_patch4 --norm_pix_loss --mask_ratio 0.75  --blr 1.5e-4 --weight_decay \
 0.05 --data_path ./data/non_similar  --data non_similar
# 4
 CUDA_VISIBLE_DEVICES=7 python retrieval.py --resume ./output_dir2/checkpoint-700.pth \
   --batch_size 2048 --model mae_vit_large_patch4 --norm_pix_loss --mask_ratio 0.75  --blr 1.5e-4 --weight_decay \
 0.05 --data_path ./data/similar  --data similar
# # 5
CUDA_VISIBLE_DEVICES=7 python retrieval.py --resume ./output_dir2/checkpoint-700.pth \
   --batch_size 2048 --model mae_vit_large_patch4 --norm_pix_loss --mask_ratio 0.75  --blr 1.5e-4 --weight_decay \
 0.05 --data_path ./data/final  --data final