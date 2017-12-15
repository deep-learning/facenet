#!/usr/bin/env bash

python3 src/train_softmax.py --logs_base_dir ~/logs/facenet/ --models_base_dir ~/models/facenet/ --data_dir /media/zhenglai/data/align_160 --image_size 160 --model_def models.inception_resnet_v1 --lfw_dir /home/zhenglai/datasets/lfw/lfw_mtcnnpy_160_margin_0 --optimizer RMSPROP --learning_rate -1 --max_nrof_epochs 160 --keep_probability 0.8 --random_flip --learning_rate_schedule_file data/learning_rate_schedule_classifier_msceleb.txt --weight_decay 5e-5 --center_loss_factor 1e-2 --center_loss_alfa 0.9 --batch_size 30 --epoch_size 3000 --gpu_memory_fraction 0.8 --embedding_size 1024 --pretrained_model /home/zhenglai/models/facenet/20171214-094447/model-20171214-094447.ckpt-21000 --global_step 216