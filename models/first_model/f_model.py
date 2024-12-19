# python train.py --img 640 --batch 16 --epochs 20 --data data/mrt.yaml --cfg models/yolov5s.yaml --weights yolov5s.pt --name mrt_yolov5s_results
# python val.py --data data/mrt.yaml --weights runs/train/mrt_yolov5s_results/weights/best.pt
# python detect.py --source path/to/your/image --weights runs/train/mrt_yolov5s_results/weights/best.pt --conf 0.5