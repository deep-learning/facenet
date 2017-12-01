
```bash 
cd ~/data/zihui_align_160
rename 's/^/zihui_/' *
cp -rf * /media/zhenglai/data/align_160
```

```bash 
python3 src/train_softmax.py --logs_base_dir ~/logs/facenet/ --models_base_dir ~/models/facenet/ --data_dir /media/zhenglai/data/align_160 --image_size 160 --model_def models.inception_resnet_v1 --lfw_dir /home/zhenglai/datasets/lfw/lfw_mtcnnpy_160_margin_0 --optimizer RMSPROP --learning_rate -1 --max_nrof_epochs 80 --keep_probability 0.8 --random_crop --random_flip --learning_rate_schedule_file data/learning_rate_schedule_classifier_msceleb.txt --weight_decay 5e-5 --center_loss_factor 1e-2 --center_loss_alfa 0.9 --batch_size 50
```

## TODO

- The parameter `epoch_size` should rather be named `evaluate_every_n_steps` or similar since it has nothing to do with the number of examples in the dataset. Hopefully this will be fixed in the future...
- For the MS-Celeb-1M results I cleaned the dataset by selecting a subset of the training images based on the distance for each image to the class center. I plan to write a wiki page for how I did it when I get some time. I didn't use a clean list as such, but instead had a file containing the distance between that image embedding to its class center. And when running training I can specify decide to use only the 75% of the images that are closest to its class center.
Learning parameters are different as well. In #48 you can see the learning curves with some hyper parameters in the legend (wd=weight decay, cl=center loss, Kp=keep percentile, 75 => keep only the 75% of the images closest to its class centers). For all runs I removed classes with less than 60 images.
The number of images and classes depends on the settings, but for example for Kp=75 gave me 4 213 410 images over 51 261 classes. Training took ~43 hours (250 000 steps with batch size 90), but the learning rate schedule can probably be optimized quite a bit). Final accuracy has varied between 0.993 and 0.995.

```
python src/calculate_filtering_metrics.py ~/datasets/msceleb/msceleb_mtcnnalign_182_160 ~/models/export/20170511-185253/20170511-185253.pb ~/filtering_metrics_msceleb_20170511-185253.hdf
``` 