import os
import tensorflow as tf


def _resolve_dataset_dir(base_dir='data/raw/chest_xray'):
    """Mendeteksi path folder train & test secara otomatis (antisipasi nested folder dari zip Kaggle)."""
    nested_dir = os.path.join(base_dir, 'chest_xray')
    if os.path.exists(os.path.join(nested_dir, 'train')):
        return nested_dir
    if os.path.exists(os.path.join(base_dir, 'train')):
        return base_dir
    raise FileNotFoundError(
        f"Folder 'train' tidak ditemukan di '{base_dir}' maupun '{nested_dir}'. "
        "Pastikan dataset Kaggle sudah diekstrak ke 'data/raw/chest_xray/'."
    )


def _compute_class_weights(train_dir):
    """Menghitung class_weight untuk mengatasi ketimpangan jumlah data NORMAL vs PNEUMONIA."""
    normal_dir = os.path.join(train_dir, 'NORMAL')
    pneumonia_dir = os.path.join(train_dir, 'PNEUMONIA')

    count_normal = len([f for f in os.listdir(normal_dir) if not f.startswith('.')])
    count_pneumonia = len([f for f in os.listdir(pneumonia_dir) if not f.startswith('.')])
    total = count_normal + count_pneumonia

    weight_for_0 = (1.0 / count_normal) * (total / 2.0)
    weight_for_1 = (1.0 / count_pneumonia) * (total / 2.0)

    class_weights = {0: round(weight_for_0, 4), 1: round(weight_for_1, 4)}
    print(f"[INFO] Distribusi Train -> NORMAL (0): {count_normal} citra | PNEUMONIA (1): {count_pneumonia} citra")
    print(f"[INFO] Class Weights terhitung -> {class_weights}")
    return class_weights


def load_and_preprocess_data(base_dir='data/raw/chest_xray', img_size=(150, 150), batch_size=32):
    print("==========================================================")
    print("       STEP 1: DATA LOADING & IMAGE PREPROCESSING         ")
    print("==========================================================\n")

    dataset_root = _resolve_dataset_dir(base_dir)
    train_dir = os.path.join(dataset_root, 'train')
    test_dir = os.path.join(dataset_root, 'test')

    print(f"[INFO] Direktori dataset terdeteksi di: '{dataset_root}'")
    print(f"[INFO] Konfigurasi Citra: Resize {img_size} RGB | Batch Size: {batch_size}")

    # 1. Hitung Class Weights dari folder train
    class_weights = _compute_class_weights(train_dir)

    # 2. Load Training & Validation Dataset (Split 80% Train, 20% Val dari folder train)
    print("[INFO] Memuat dataset Training (80%) dan Validation (20%)...")
    train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=0.2,
        subset='both',
        seed=42,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='binary',
        color_mode='rgb'
    )

    # 3. Load Testing Dataset (shuffle=False wajib agar urutan label sinkron saat evaluasi)
    print("[INFO] Memuat dataset Testing (Unseen Data)...")
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='binary',
        color_mode='rgb',
        shuffle=False
    )

    class_names = train_ds.class_names
    print(f"[INFO] Kelas Target terdeteksi: {class_names} (0={class_names[0]}, 1={class_names[1]})")

    # 4. Optimasi Pipeline Streaming (Prefetch untuk mencegah bottleneck CPU/RAM)
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

    print("[INFO] Pipeline preprocessing dan streaming tensor siap digunakan!\n")
    return train_ds, val_ds, test_ds, class_weights, class_names


if __name__ == '__main__':
    load_and_preprocess_data()