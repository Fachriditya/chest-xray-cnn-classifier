import tensorflow as tf
from tensorflow.keras import layers, models


def build_custom_cnn(input_shape=(150, 150, 3), learning_rate=0.0005):
    print("[INFO] Membangun arsitektur Custom CNN 2D (from scratch)...")

    # Layer Data Augmentation (hanya aktif saat proses training)
    data_augmentation = models.Sequential([
        layers.RandomFlip('horizontal'),
        layers.RandomRotation(0.1),
    ], name='data_augmentation')

    model = models.Sequential([
        layers.Input(shape=input_shape),
        data_augmentation,
        layers.Rescaling(1.0 / 255.0),

        # --- Convolutional Block 1 (32 Filters) ---
        layers.Conv2D(32, (3, 3), padding='same'),
        layers.LeakyReLU(negative_slope=0.1),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.SpatialDropout2D(0.2),

        # --- Convolutional Block 2 (64 Filters) ---
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.LeakyReLU(negative_slope=0.1),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.SpatialDropout2D(0.2),

        # --- Convolutional Block 3 (128 Filters) ---
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.LeakyReLU(negative_slope=0.1),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.SpatialDropout2D(0.2),

        # --- Classification Head ---
        layers.GlobalAveragePooling2D(),
        layers.Dense(256),
        layers.LeakyReLU(negative_slope=0.1),
        layers.Dropout(0.4),
        layers.Dense(1, activation='sigmoid')
    ], name='Custom_ChestXRay_CNN')

    print(f"[INFO] Mengompilasi model (Optimizer: Adam lr={learning_rate}, Loss: binary_crossentropy)...")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


if __name__ == '__main__':
    m = build_custom_cnn()
    m.summary()