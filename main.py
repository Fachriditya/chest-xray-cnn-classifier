import os
import sys
import time

# Redam log C++ INFO/WARNING TensorFlow agar output terminal bersih untuk laporan
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing import load_and_preprocess_data
from src.train import train_cnn_model
from src.evaluate import evaluate_cnn_model


def main():
    start_time = time.time()
    print("==========================================================")
    print("   KLASIFIKASI CHEST X-RAY (PNEUMONIA) - CUSTOM CNN 2D    ")
    print("==========================================================\n")

    # 1. Preprocessing & Streaming Pipeline
    train_ds, val_ds, test_ds, class_weights, class_names = load_and_preprocess_data(
        base_dir='data/raw/chest_xray',
        img_size=(150, 150),
        batch_size=32
    )

    # 2. Training Custom CNN 2D
    model, history = train_cnn_model(
        train_ds=train_ds,
        val_ds=val_ds,
        class_weights=class_weights,
        epochs=30,
        model_output_path='assets/cnn_pneumonia_model.keras'
    )

    # 3. Evaluasi & Ekspor Visualisasi
    evaluate_cnn_model(
        model=model,
        history=history,
        test_ds=test_ds,
        class_names=class_names,
        cm_output_path='assets/confusion_matrix_cnn.png'
    )

    elapsed = time.time() - start_time
    print(f"[INFO] Total waktu eksekusi pipeline: {elapsed / 60:.2f} menit.")


if __name__ == '__main__':
    main()