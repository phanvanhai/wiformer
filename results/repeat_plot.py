import matplotlib.pyplot as plt

SPLIT_BY_REPEAT = 1

if SPLIT_BY_REPEAT:
    # split by repeat number
    # models = ["ResNet1D", "CNN_GRU", "BiMamba"]
    # datasets = {
    #     "XRF55-subset": [99.27, 98.13, 96.35],
    #     "ESP":   [99.69, 97.5, 95.78],
    #     "ASUS":  [99.14, 97.73, 97.03],
    # }
    models = ["CNN_GRU", "BiMamba"]
    datasets = {
        "XRF55-subset": [98.13, 96.35],
        "CMR-HAR-ESP32":   [97.5, 95.78],
        "CMR-HAR-Nexmon":  [97.73, 97.03],
    }
else:
    pass

plt.figure(figsize=(8, 5))

for name, values in datasets.items():
    plt.plot(models, values, marker="o", linewidth=2, label=name)

    for x, y in zip(models, values):
        plt.annotate(
            f"{y:.2f}",
            (x, y),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center"
        )

plt.xlabel("Model")
plt.ylabel("Accuracy (%)")

# Gộp tất cả các list lại rồi tìm min
min_value = min(val for sublist in datasets.values() for val in sublist)
max_value = max(val for sublist in datasets.values() for val in sublist)


plt.ylim(int(min_value), min(int(max_value) + 5, 99))
plt.grid(True, linestyle="--", alpha=0.35)
plt.legend()

plt.tight_layout()
plt.savefig("model_accuracy_comparison.png", dpi=300, bbox_inches="tight")
plt.show()