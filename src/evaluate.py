import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


def plot_training_history(history, output_path='assets/training_history.png'):
    """Menyimpan grafik kurva Accuracy & Loss selama proses training."""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(1, len(acc) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    # Subplot 1: Accuracy
    axes[0].plot(epochs_range, acc, label='Training Accuracy', marker='o', markersize=3)
    axes[0].plot(epochs_range, val_acc, label='Validation Accuracy', marker='o', markersize=3)
    axes[0].set_title('Training vs Validation Accuracy', fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend(loc='lower right')
    axes[0].grid(True, linestyle='--', alpha=0.5)

    # Subplot 2: Loss
    axes[1].plot(epochs_range, loss, label='Training Loss', marker='o', markersize=3)
    axes[1].plot(epochs_range, val_loss, label='Validation Loss', marker='o', markersize=3)
    axes[1].set_title('Training vs Validation Loss', fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"[INFO] Grafik Training History disimpan di: '{output_path}'")


def evaluate_cnn_model(
    model,
    history,
    test_ds,
    class_names,
    cm_output_path='assets/confusion_matrix_cnn.png'
):
    print("==========================================================")
    print("       STEP 3: EVALUASI MODEL & VISUALISASI METRIK        ")
    print("==========================================================\n")

    # 1. Simpan grafik history training
    plot_training_history(history, output_path='assets/training_history.png')

    # 2. Evaluasi Loss & Akurasi pada Data Uji
    print("[INFO] Mengevaluasi model pada dataset Testing...")
    loss, acc = model.evaluate(test_ds, verbose=0)

    print(f"\n[INFO] Akurasi Data Uji (Test Accuracy) : {acc * 100:.2f}%")
    print(f"[INFO] Loss Data Uji (Test Loss)        : {loss:.4f}\n")

    # 3. Ekstraksi Label Aktual & Prediksi
    print("[INFO] Menghitung prediksi kelas pada dataset Testing...")
    y_true = np.concatenate([y.numpy() for _, y in test_ds], axis=0).flatten().astype(int)
    y_pred_prob = model.predict(test_ds, verbose=0).flatten()
    y_pred = (y_pred_prob > 0.5).astype(int)

    # 4. Cetak Classification Report
    print(" CLASSIFICATION REPORT (CUSTOM CNN 2D) ".center(60, '='))
    print(classification_report(y_true, y_pred, target_names=class_names))
    print("=" * 60 + "\n")

    # 5. Plot & Simpan Confusion Matrix
    print("[INFO] Membuat visualisasi Heatmap Confusion Matrix...")
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6.5, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        annot_kws={'fontsize': 13, 'fontweight': 'bold'}
    )
    plt.xlabel('Predicted Label', fontweight='bold')
    plt.ylabel('Actual Label', fontweight='bold')
    plt.title('Confusion Matrix - Custom CNN 2D (Chest X-Ray)', fontweight='bold', pad=12)
    plt.tight_layout()

    os.makedirs(os.path.dirname(cm_output_path), exist_ok=True)
    plt.savefig(cm_output_path, dpi=200)
    plt.close()

    print(f"[INFO] Grafik Confusion Matrix disimpan di: '{cm_output_path}'\n")
    print("==========================================================")
    print("          PIPELINE CUSTOM CNN 2D TELAH SELESAI!           ")
    print("==========================================================\n")


if __name__ == '__main__':
    pass