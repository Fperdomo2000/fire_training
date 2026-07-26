"""Default training configuration for TorchvisionEfficientNetForClassification.

Edit the parameters below to customize training, then run train.py.
The get_training_args() function returns a fully-configured TrainingArguments instance.
"""

from pathlib import Path

from transformers import TrainingArguments

# Image preprocessing
image_size = 384

# Training hyperparameters
num_epochs = 10
per_device_train_batch_size = 16
per_device_eval_batch_size = 32
learning_rate = 3e-5
weight_decay = 1e-4
warmup_ratio = 0.05
seed = 42

# Dataset (TIFF only): recompute band_stats.json even if a cached one exists
recompute_band_stats = False

# Mixed precision training
fp16 = False

# Checkpointing
save_total_limit = 2
logging_steps = 20


def get_training_args(output_dir: Path) -> TrainingArguments:
    """Returns a TrainingArguments instance configured with the parameters above.

    Args:
        output_dir: The output directory for checkpoints. The final model is
            saved to <output_dir>/final/ by train.py.
    """
    return TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=logging_steps,
        seed=seed,
        fp16=fp16,
        report_to=[],
    )
