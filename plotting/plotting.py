import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix_f(confusion_matrix, loader, num_classes):
    fig, ax = plt.subplots(figsize=(8, 6))

    class_labels = getattr(loader.dataset, 'class_names', None)
    if class_labels is None:
        class_labels = [str(i) for i in range(num_classes)]

    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=True,
        xticklabels=class_labels,
        yticklabels=class_labels,
        ax=ax
    )

    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix')
    plt.tight_layout()
    plt.show()