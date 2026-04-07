LD_PRELOAD=/misc/envs/bonsai/lib/libjpeg.so.8 \
    /misc/envs/bonsai/bin/python scripts/convert_imagenet_ffcv.py \
    --data /media/data/datasets/imagenet \
    --output /media/data/datasets/imagenet_ffcv \
    --train-resolution 800 --val-resolution 256 \
    --jpeg-quality 95 --workers 16
