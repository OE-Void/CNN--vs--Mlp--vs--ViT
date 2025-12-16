import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')

def load_mnist():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Normalize
    x_train = x_train.astype("float32") / 255.0
    x_test  = x_test.astype("float32") / 255.0

    # Add channel dimension
    x_train = np.expand_dims(x_train, -1)
    x_test  = np.expand_dims(x_test, -1)

    # One-hot encoding
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test  = tf.keras.utils.to_categorical(y_test, 10)

    return (x_train, y_train), (x_test, y_test)

class PatchEmbedding(layers.Layer):
    def __init__(self, patch_size=7, embed_dim=64, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.proj = layers.Dense(embed_dim)

    def call(self, x):
        # x shape: (batch, H, W, C)
        patches = tf.image.extract_patches(
            images=x,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID"
        )
        # patches shape: (batch, grid_h, grid_w, patch_h*patch_w*C)
        patch_dim = patches.shape[-1]

        # Reshape to (batch, num_patches, flattened_patch_dim)
        # We use tf.shape(x)[0] for dynamic batch size
        patches = tf.reshape(patches, [tf.shape(x)[0], -1, patch_dim])
        return self.proj(patches)

    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size, "embed_dim": self.embed_dim})
        return config

class ClassToken(layers.Layer):
    def __init__(self, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim

    def build(self, input_shape):
        self.cls_token = self.add_weight(
            name="cls_token",
            shape=(1, 1, self.embed_dim),
            initializer="zeros",
            trainable=True
        )
        super().build(input_shape)

    def call(self, x):
        batch_size = tf.shape(x)[0]
        # Broadcast cls_token to match batch size
        cls_tokens = tf.tile(self.cls_token, [batch_size, 1, 1])
        return tf.concat([cls_tokens, x], axis=1)

    def get_config(self):
        config = super().get_config()
        config.update({"embed_dim": self.embed_dim})
        return config

class PositionalEmbedding(layers.Layer):
    def __init__(self, num_tokens, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_tokens = num_tokens
        self.embed_dim = embed_dim

    def build(self, input_shape):
        self.pos_embed = self.add_weight(
            name="pos_embed",
            shape=(1, self.num_tokens, self.embed_dim),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
            trainable=True
        )
        super().build(input_shape)

    def call(self, x):
        return x + self.pos_embed

    def get_config(self):
        config = super().get_config()
        config.update({"num_tokens": self.num_tokens, "embed_dim": self.embed_dim})
        return config

def build_mlp(input_shape=(28, 28, 1), num_classes=10):
    inputs = layers.Input(shape=input_shape)
    x = layers.Flatten()(inputs)
    for _ in range(3):
        x = layers.Dense(512, activation="relu")(x)
        x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inputs, outputs, name="MLP")

def build_cnn(input_shape=(28, 28, 1), num_classes=10):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPool2D()(x)

    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPool2D()(x)

    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inputs, outputs, name="CNN")

def apply_transformer_block(x, embed_dim, num_heads, mlp_dim, dropout=0.1):
    """
    Applies a transformer block to tensor x.
    Refactored from being a separate Model to a functional application
    to ensure correct graph tracing and model saving.
    """
    # Attention Block
    x1 = layers.LayerNormalization(epsilon=1e-6)(x)
    attn_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(x1, x1)
    attn_output = layers.Dropout(dropout)(attn_output)
    x2 = layers.Add()([x, attn_output])

    # MLP Block
    x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
    x3 = layers.Dense(mlp_dim, activation="gelu")(x3)
    x3 = layers.Dropout(dropout)(x3)
    x3 = layers.Dense(embed_dim)(x3)
    x3 = layers.Dropout(dropout)(x3)

    return layers.Add()([x2, x3])

def build_vit(img_size=28, patch_size=7, embed_dim=64, depth=6, num_heads=4, mlp_dim=128, num_classes=10):
    inputs = layers.Input(shape=(img_size, img_size, 1))

    # 1. Patch Embeddings
    x = PatchEmbedding(patch_size, embed_dim)(inputs)

    # 2. Append Class Token
    x = ClassToken(embed_dim)(x)

    # 3. Add Positional Embeddings
    # Calculate number of patches + 1 class token
    num_patches = (img_size // patch_size) ** 2
    num_tokens = num_patches + 1
    x = PositionalEmbedding(num_tokens, embed_dim)(x)

    # 4. Transformer Encoder Blocks
    for _ in range(depth):
        x = apply_transformer_block(x, embed_dim, num_heads, mlp_dim)

    # 5. Output Head
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    # Take the [CLS] token (index 0)
    cls_out = x[:, 0, :]
    outputs = layers.Dense(num_classes, activation="softmax")(cls_out)

    return models.Model(inputs, outputs, name="ViT")


def train_model(model, x_train, y_train, x_test, y_test, epochs=10, batch_size=128):
    # Use AdamW if available (TF 2.5+) for better ViT convergence, else standard Adam
    try:
        optimizer = optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4)
    except AttributeError:
        optimizer = optimizers.Adam(learning_rate=1e-3)

    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    start = time.perf_counter()
    history = model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    elapsed = time.perf_counter() - start
    return history, elapsed

def run_benchmark(epochs=10, batch_size=128):
    (x_train, y_train), (x_test, y_test) = load_mnist()

    mlp = build_mlp()
    cnn = build_cnn()
    vit = build_vit(depth=6)

    results = {}
    for model in [mlp, cnn, vit]:
        print(f"\n{'='*10} Training {model.name} {'='*10}")
        # Print summary to check architecture
        # model.summary()

        history, elapsed = train_model(model, x_train, y_train, x_test, y_test, epochs, batch_size)

        params = model.count_params()
        final_acc = history.history["val_accuracy"][-1]

        results[model.name] = {
            "history": history.history,
            "time_sec": elapsed,
            "params": params,
            "final_acc": final_acc,
        }
    return results

def plot_curves(results, outdir="plots"):
    os.makedirs(outdir, exist_ok=True)

    # Validation Accuracy
    plt.figure(figsize=(10,6))
    for name, r in results.items():
        plt.plot(r["history"]["val_accuracy"], label=f"{name}")
    plt.title("Validation accuracy vs epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Val Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(outdir, "val_accuracy.png"))
    plt.close()

    # Validation Loss
    plt.figure(figsize=(10,6))
    for name, r in results.items():
        plt.plot(r["history"]["val_loss"], label=f"{name}")
    plt.title("Validation loss vs epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Val Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(outdir, "val_loss.png"))
    plt.close()

if __name__ == "__main__":
    # Ensure TF doesn't eat all GPU memory if running locally
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    results = run_benchmark(epochs=10, batch_size=128)
    plot_curves(results)

    print("\nBenchmark Summary:")
    print("-" * 50)
    for name, r in results.items():
        print(f"{name}: Params={r['params']:,} | Time={r['time_sec']:.1f}s | Final Val Acc={r['final_acc']:.4f}")