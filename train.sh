CUDA_VISIBLE_DEVICES=3 python -m domainbed.scripts.train \
    --data_dir=./datasets/ \
    --output_dir=./checkpoint/Glaucoma/resnet50-BN/ \
    --algorithm ERM \
    --dataset Glaucoma \
    --hparams '''{"backbone": "resnet50-BN", "lr": 1e-4}''' \
    --trial_seed 0 \
    --train_envs 0