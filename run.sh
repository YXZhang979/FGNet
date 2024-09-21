CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
  --dataset pascal --data-root /data1/zhangyuxuan/VOC2012 \
  --backbone resnet101 --fold 0 --shot 5
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
  --dataset pascal --data-root /data1/zhangyuxuan/VOC2012 \
  --backbone resnet101 --fold 1 --shot 5
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
  --dataset pascal --data-root /data1/zhangyuxuan/VOC2012 \
  --backbone resnet101 --fold 2 --shot 5
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
  --dataset pascal --data-root /data1/zhangyuxuan/VOC2012 \
  --backbone resnet101 --fold 3 --shot 5
