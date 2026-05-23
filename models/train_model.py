from ultralytics import YOLO
import shutil
from pathlib import Path

def train_and_deploy():
    print(" Optimized Model Training...")
    
    # 1. Load the factory-default model
    model = YOLO("yolo11n.pt")

    # 2. Train the model using your exact Mac-optimized settings
    results = model.train(
        data="datasets/weed_crop_aerial/data.yaml",
        epochs=25,
        imgsz=640,
        device="mps",
        batch=16,
        workers=2,
        cache=True,
        patience=10
    )

    # 3. Locate where YOLO saved the new "best.pt"
    # results.save_dir automatically knows if it was train-4, train-5, etc.
    best_model_path = Path(results.save_dir) / "weights" / "best.pt"
    
    # 4. Define where we want it to go
    destination = Path("models/aerial_weeds.pt")
    destination.parent.mkdir(parents=True, exist_ok=True)

    # 5. Automatically copy and rename the file!
    shutil.copy(best_model_path, destination)
    
    print(f"\n✅ TRAINING COMPLETE!")
    print(f" New AI brain automatically installed at: {destination}")

if __name__ == "__main__":
    train_and_deploy()