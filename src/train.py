import os
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from src.model import build_custom_cnn


def train_cnn_model(
    train_ds,
    val_ds,
    class_weights,
    epochs=30,
    model_output_path='assets/cnn_pneumonia_model.keras'
):
    print("==========================================================")
    print("           STEP 2: TRAINING CUSTOM CNN 2D MODEL           ")
    print("==========================================================\n")

    model = build_custom_cnn(input_shape=(150, 150, 3), learning_rate=0.0005)

    print("\n[INFO] --- Ringkasan Arsitektur CNN ---")
    model.summary()

    # Konfigurasi Callbacks untuk efisiensi komputasi & mencegah overfitting
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=6,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
    ]

    print(f"\n[INFO] Memulai proses training (Maksimal {epochs} Epochs)...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )

    # Simpan model terbaik ke folder assets
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    model.save(model_output_path)
    print(f"\n[INFO] Training selesai! Model terbaik disimpan di: '{model_output_path}'\n")

    return model, history


if __name__ == '__main__':
    pass