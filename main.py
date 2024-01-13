from ultralytics import YOLO

if __name__ == '__main__':

    # model = YOLO("yolov8n.yaml")

    # results = model.train(data="data/data.yaml", epochs=100, batch=4, imgsz=640)

    model = YOLO("runs/detect/train2/weights/best.pt")
    metrics = model.val()
    metrics.box.map    # map50-95
    metrics.box.map50  # map50
    metrics.box.map75  # map75
    metrics.box.maps   # a list contains map50-95 of each category