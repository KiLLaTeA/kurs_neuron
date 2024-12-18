# python train.py --img 640 --batch 16 --epochs 50 --data data/mrt.yaml --cfg models/yolov5s.yaml --weights yolov5s.pt --name mrt_yolov5s_results

# python val.py --data data/mrt.yaml --weights runs/train/mrt_yolov5s_results/weights/best.pt

# python detect.py --source path/to/your/image --weights runs/train/mrt_yolov5s_results/weights/best.pt --conf 0.5

# import torch
# # Загрузка модели
# model = torch.hub.load('ultralytics/yolov5', 'custom',
# path='runs/train/daisy_yolov5s_results/weights/best.pt')
# # Загрузка изображения
# img ='1s.jpg'
# # Детектирование объектов
# results = model(img)
# # Вывод результатов
# results.show() # Показывает изображение с обнаруженными объектами
# results.print() # Выводит информацию о детектированных объектах консоль
# results.save() # Сохраняет изображение с обнаруженными объектами