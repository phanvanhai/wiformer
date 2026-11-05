import matplotlib.pyplot as plt

SPLIT_BY_USER = 1

if not SPLIT_BY_USER:
    # split by repeat number
    models = ["ResNet1D", "CNN_GRU", "BiMamba"]
    datasets = {
        "XRF55": [98.44, 97.92, 95.21],
        "ESP":   [96.80, 96.17, 95.78],
        "ASUS":  [99.22, 97.58, 96.72],
        # "ESP_30sub":   [97.42, 98.13 , 96.25],
        # "ASUS_30sub":  [99.22, 98.05, 96.48],
        # "ASUS_1anten":  [98.67, 96.09, 95.78],
        # "ASUS_2anten":  [99.38, 96.80, 96.72],
        # "ASUS_3anten":  [99.14, 97.27, 97.50],
    }
else:
    # split by user number
    models = ["ResNet1D", "CNN_GRU", "BiMamba"]
    datasets = {
        "XRF55": [44.48, 31.67, 34.79],
        "ESP":   [58.05, 31.41, 37.58],
        "ASUS":  [47.66, 37.97, 60.86],
    }

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


plt.ylim(int(min_value) - 5, min(int(max_value) + 5, 100))
plt.grid(True, linestyle="--", alpha=0.35)
plt.legend()

plt.tight_layout()
plt.savefig("model_accuracy_comparison.png", dpi=300, bbox_inches="tight")
plt.show()