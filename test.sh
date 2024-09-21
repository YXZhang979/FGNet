python test.py \
  --dataset pascal --data-root /data1/zhangyuxuan/VOC2012 \
  --backbone resnet101 --fold 0 --shot 5
python -W ignore test.py \
  --dataset pascal --data-root /data1/zhangyuxuan/VOC2012 \
  --backbone resnet101 --fold 1 --shot 5
python -W ignore test.py \
  --dataset pascal --data-root /data1/zhangyuxuan/VOC2012 \
  --backbone resnet101 --fold 2 --shot 5
python -W ignore test.py \
  --dataset pascal --data-root /data1/zhangyuxuan/VOC2012 \
  --backbone resnet101 --fold 3 --shot 5

