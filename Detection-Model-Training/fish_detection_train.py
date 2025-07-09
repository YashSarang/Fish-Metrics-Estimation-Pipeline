from ultralytics import YOLO


def main():
    # Load the model
    # Make sure the model is compatible with your training code
    model = YOLO("yolo12s.pt")

    # Train the model on GPU (CUDA)
    model.train(
        data="UnderWaterfish.v4i.yolov12/data.yaml",  # Path to dataset config
        epochs=150,
        imgsz=640,     # Image size
        batch=16,      # Adjust based on your GPU's VRAM
        device='cuda',  # Force CUDA usage
        project="/runs/train",
        name="underwater_fish_yolo12s",
        resume=False
    )


if __name__ == "__main__":
    main()
