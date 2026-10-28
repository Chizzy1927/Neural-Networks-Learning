import os

import matplotlib.pyplot as plt


# Results from the three SST-2 transfer-learning experiments
models = ["Frozen", "LoRA", "Full Fine-Tuning"]

validation_accuracy = [83.37, 89.45, 90.37]
test_accuracy = [85.70, 90.54, 94.40]
f1_scores = [86.78, 91.30, 94.85]

trainable_parameters = [
    592_130,
    739_586,
    66_955_010,
]

trainable_percentage = [
    0.88,
    1.09,
    100.00,
]

training_time = [
    44.37,
    103.91,
    150.43,
]


# Create the results folder if it does not already exist.
results_folder = os.path.join(
    os.path.dirname(__file__),
    "results",
)

os.makedirs(results_folder, exist_ok=True)


# Create one figure containing the main experiment comparisons.
fig, axes = plt.subplots(
    2,
    2,
    figsize=(13, 9),
)

fig.suptitle(
    "DistilBERT Fine-Tuning Strategy Comparison on SST-2",
    fontsize=16,
    fontweight="bold",
)


# 1. Validation and test accuracy
x = range(len(models))
width = 0.35

axes[0, 0].bar(
    [position - width / 2 for position in x],
    validation_accuracy,
    width,
    label="Validation Accuracy",
)

axes[0, 0].bar(
    [position + width / 2 for position in x],
    test_accuracy,
    width,
    label="Comparison-Set Accuracy",
)

axes[0, 0].set_title("Model Performance")
axes[0, 0].set_ylabel("Accuracy (%)")
axes[0, 0].set_xticks(list(x))
axes[0, 0].set_xticklabels(models)
axes[0, 0].set_ylim(75, 100)
axes[0, 0].legend()

for index, value in enumerate(validation_accuracy):
    axes[0, 0].text(
        index - width / 2,
        value + 0.4,
        f"{value:.2f}%",
        ha="center",
        fontsize=9,
    )

for index, value in enumerate(test_accuracy):
    axes[0, 0].text(
        index + width / 2,
        value + 0.4,
        f"{value:.2f}%",
        ha="center",
        fontsize=9,
    )


# 2. Trainable parameter percentage
axes[0, 1].bar(
    models,
    trainable_percentage,
)

axes[0, 1].set_title("Trainable Parameters")
axes[0, 1].set_ylabel("Trainable Parameters (%)")

for index, value in enumerate(trainable_percentage):
    axes[0, 1].text(
        index,
        value + 1,
        f"{value:.2f}%",
        ha="center",
        fontsize=10,
    )


# 3. Training runtime
axes[1, 0].bar(
    models,
    training_time,
)

axes[1, 0].set_title("Training Runtime")
axes[1, 0].set_ylabel("Minutes")

for index, value in enumerate(training_time):
    axes[1, 0].text(
        index,
        value + 2,
        f"{value:.2f} min",
        ha="center",
        fontsize=10,
    )


# 4. F1 score
axes[1, 1].bar(
    models,
    f1_scores,
)

axes[1, 1].set_title("Comparison-Set F1 Score")
axes[1, 1].set_ylabel("F1 Score (%)")
axes[1, 1].set_ylim(80, 100)

for index, value in enumerate(f1_scores):
    axes[1, 1].text(
        index,
        value + 0.3,
        f"{value:.2f}%",
        ha="center",
        fontsize=10,
    )


# Add the main takeaway beneath the charts.
fig.text(
    0.5,
    0.01,
    (
        "LoRA reached 89.45% validation accuracy while training "
        "only 1.09% of the model parameters."
    ),
    ha="center",
    fontsize=11,
)


plt.tight_layout(
    rect=[0, 0.04, 1, 0.95]
)


output_path = os.path.join(
    results_folder,
    "model_comparison.png",
)

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight",
)

print(f"Saved comparison chart to: {output_path}")

plt.show()