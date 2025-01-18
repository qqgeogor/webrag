python train_ebm_cifar10_gan_r3gan_ctrl_vit_dec_imagenet.py \
    --data_path /root/autodl-tmp/imagenet100/ \
    --output_dir /root/autodl-tmp/output/imagenet100_ctrl \
    --img_size 128 \
    --patch_size 16 \
    --epochs 1200 \
    --batch_size 128 \
    --embed_dim 192 \
    --decoder_embed_dim 192 \
    --encoder_depth 12 \
    --decoder_depth 4 \
    --num_heads 3 


python visualize_attention.py     --checkpoint_path ./output/cifar10-ebm-gan-r3gan-ctrl-vit-dec/ebm_gan_checkpoint_160.pth     --output_dir ./attention_vis     --num_samples 10     --layer_idx 5  # Visualize the third layer (0-based indexing)
 

python visualize_attention.py    --checkpoint_path F:/output/cifar10-ebm-cl-r-ema-dino/ebm_gan_checkpoint_10.pth     --output_dir ./attention_vis     --num_samples 10     --layer_idx 5  # Visualize the third layer (0-based indexing)
 