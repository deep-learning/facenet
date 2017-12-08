export PYTHONPATH=$PYTHONPATH:$HOME/repo/facenet/src
export CUDA_VISIBLE_DEVICES=''
python3 src/align/align_dataset_mtcnn.py ~/data/msceleb/raw ~/data/msceleb/msceleb/s182.m44 --image_size 182 --margin 44 --gpu_memory_fraction 0.2 --resize --warn_multiple_faces --parallelism 40