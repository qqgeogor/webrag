conda activate F:\condaenv\llma
# python train_ebm_cifar10_gan_r3gan_ctrl_convnext_imagenet.py `
#     --data_path C:/dataset/tiny-imagenet/train/ `
#     --output_dir F:/output/tiny_imagenet_ctrl `
#     --img_size 64 `
#     --patch_size 8 `
#     --epochs 1200 `
#     --batch_size 128 `
#     --embed_dim 192 `
#     --decoder_embed_dim 192 `
#     --encoder_depth 6 `
#     --decoder_depth 6 `
#     --save_freq 1 `
#     --num_heads 3 


# python train_ebm_cifar10_gan_r3gan_ctrl_imagenet.py `
#     --data_path C:/dataset/tiny-imagenet/train/ `
#     --output_dir F:/output/tiny_imagenet_ctrl `
#     --img_size 64 `
#     --epochs 1200 `
#     --batch_size 128 `
#     --latent_dim 64 `

# conda activate F:\condaenv\llma
# python train_ebm_cifar10_gan_r3gan_ctrl_vit_imagenet.py `
#     --data_path C:/dataset/tiny-imagenet/train/ `
#     --output_dir F:/output/tiny_imagenet_ctrl `
#     --img_size 64 `
#     --patch_size 8 `
#     --epochs 1200 `
#     --batch_size 128 `
#     --embed_dim 192 `
#     --decoder_embed_dim 192 `
#     --encoder_depth 6 `
#     --decoder_depth 6 `
#     --num_heads 3 `
#     --save_freq 1 

# # python train_ebm_cifar10_gan_r3gan_ctrl_imagenet.py `
# #     --data_path C:/dataset/tiny-imagenet/train/ `
# #     --output_dir F:/output/tiny_imagenet_ctrl `
# #     --img_size 64 `
# #     --epochs 1200 `
# #     --batch_size 128 `
# #     --latent_dim 64 `


# python train_ebm_cifar10_gan_r3gan_ctrl_imagenet_dino.py `
#     --data_path C:/dataset/tiny-imagenet/train/ `
#     --output_dir F:/output/tiny_imagenet_ctrl `
#     --img_size 32 `
#     --epochs 1200 `
#     --batch_size 128 `
#     --latent_dim 64 `

# python .\train_ebm_cifar10_gan_r3gan_ctrl_dino.py --resume F:/output/tiny_imagenet_ctrl_dino/ebm_gan_checkpoint_91.pth


python .\train_ebm_cifar10_gan_r3gan_class.py #--resume F:/output/tiny_imagenet_ctrl_dino_class/ebm_gan_checkpoint_174.pth --batch_size 1024 --save_freq 10

