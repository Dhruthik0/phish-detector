# # # src/train_cnn.py
# # from __future__ import annotations
# # import argparse, os
# # import torch
# # import torch.nn as nn
# # from torch.utils.data import DataLoader, Dataset
# # import pandas as pd
# # import numpy as np
# # from sklearn.utils.class_weight import compute_class_weight
# # from sklearn.model_selection import train_test_split
# # from tqdm import tqdm

# # from src.dataset import prepare_sequences, CHAR2IDX
# # from src.cnn_model_torch import CharCNN


# # # -------------------------
# # # Dataset Wrapper
# # # -------------------------
# # class URLDataset(Dataset):
# #     def __init__(self, urls: list[str], labels: list[int], max_len: int = 200):
# #         self.X = prepare_sequences(urls, max_len=max_len)
# #         self.y = np.array(labels, dtype=np.float32)

# #     def __len__(self):
# #         return len(self.y)

# #     def __getitem__(self, idx):
# #         return torch.tensor(self.X[idx], dtype=torch.long), torch.tensor(self.y[idx], dtype=torch.float32)


# # # -------------------------
# # # Training Function
# # # -------------------------
# # def train_cnn_model(csv_path: str, model_out: str = "models/cnn_best_torch.pt",
# #                     max_len: int = 200, batch_size: int = 64, epochs: int = 10, lr: float = 1e-3):
# #     # Load data
# #     df = pd.read_csv(csv_path)
# #     label_col = next((c for c in ["label", "is_phishing", "phishing"] if c in df.columns), None)
# #     if label_col is None:
# #         raise ValueError("No label column found in dataset")

# #     urls = df["url"].astype(str).tolist()
# #     y = df[label_col].astype(int).tolist()

# #     # Train/val split
# #     X_train, X_val, y_train, y_val = train_test_split(urls, y, test_size=0.2, stratify=y, random_state=42)

# #     train_ds = URLDataset(X_train, y_train, max_len=max_len)
# #     val_ds = URLDataset(X_val, y_val, max_len=max_len)
# #     train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
# #     val_loader = DataLoader(val_ds, batch_size=batch_size)

# #     # Device
# #     device = torch.device("mps" if torch.backends.mps.is_available() else
# #                           "cuda" if torch.cuda.is_available() else "cpu")
# #     print(f"⚡ Training on: {device}")

# #     # Model
# #     vocab_size = len(CHAR2IDX) + 2  # PAD + UNK
# #     print("Using vocab size:", vocab_size)
# #     model = CharCNN(vocab_size=vocab_size).to(device)

# #     # Class weights (for imbalance)
# #     class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
# #     class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
# #     print("Class weights:", class_weights)

# #     # Loss + Optimizer
# #     criterion = nn.BCELoss()
# #     optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# #     best_val_loss = float("inf")

# #     for epoch in range(epochs):
# #         # ---- Training ----
# #         model.train()
# #         total_loss = 0
# #         loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
# #         for Xb, yb in loop:
# #             Xb, yb = Xb.to(device), yb.to(device)

# #             optimizer.zero_grad()
# #             preds = model(Xb)            # shape: [batch]
# #             loss = criterion(preds, yb)
# #             loss.backward()
# #             optimizer.step()

# #             total_loss += loss.item() * len(yb)
# #             loop.set_postfix(loss=loss.item())

# #         train_loss = total_loss / len(train_ds)

# #         # ---- Validation ----
# #         model.eval()
# #         val_loss = 0
# #         with torch.no_grad():
# #             for Xb, yb in val_loader:
# #                 Xb, yb = Xb.to(device), yb.to(device)
# #                 preds = model(Xb)        # shape: [batch]
# #                 loss = criterion(preds, yb)
# #                 val_loss += loss.item() * len(yb)

# #         val_loss /= len(val_ds)

# #         print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

# #         # Save best
# #         if val_loss < best_val_loss:
# #             best_val_loss = val_loss
# #             os.makedirs(os.path.dirname(model_out), exist_ok=True)
# #             torch.save(model.state_dict(), model_out)
# #             print(f"✅ Saved best model at epoch {epoch+1}")

# #     print("Training complete.")


# # # -------------------------
# # # CLI Entry
# # # -------------------------
# # if __name__ == "__main__":
# #     p = argparse.ArgumentParser()
# #     p.add_argument("--data", default="data/raw/phishing.csv")
# #     p.add_argument("--epochs", type=int, default=10)
# #     p.add_argument("--batch_size", type=int, default=64)
# #     args = p.parse_args()

# #     train_cnn_model(args.data, epochs=args.epochs, batch_size=args.batch_size)
# # src/cnn_model_torch.py
# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class CharCNN(nn.Module):
#     def __init__(self, vocab_size: int, embed_dim: int = 128, num_filters: int = 256, dropout: float = 0.5):
#         super(CharCNN, self).__init__()

#         # Character embedding
#         self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

#         # Parallel convolution layers with different kernel sizes
#         kernel_sizes = [3, 5, 7]
#         self.convs = nn.ModuleList([
#             nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=k)
#             for k in kernel_sizes
#         ])

#         # BatchNorm for stability
#         self.batch_norms = nn.ModuleList([
#             nn.BatchNorm1d(num_filters) for _ in kernel_sizes
#         ])

#         # Dropout
#         self.dropout = nn.Dropout(dropout)

#         # Fully connected layers
#         self.fc1 = nn.Linear(len(kernel_sizes) * num_filters, 256)
#         self.fc2 = nn.Linear(256, 1)

#     def forward(self, x):
#         # x: (batch, seq_len)
#         x = self.embedding(x)  # (batch, seq_len, embed_dim)
#         x = x.transpose(1, 2)  # (batch, embed_dim, seq_len) for Conv1d

#         # Apply convs with different kernel sizes
#         conv_outputs = []
#         for conv, bn in zip(self.convs, self.batch_norms):
#             c = conv(x)                      # (batch, num_filters, L_out)
#             c = bn(c)
#             c = F.relu(c)
#             c = F.max_pool1d(c, c.shape[2])  # global max pool → (batch, num_filters, 1)
#             conv_outputs.append(c.squeeze(2))  # → (batch, num_filters)

#         # Concatenate conv outputs
#         out = torch.cat(conv_outputs, dim=1)  # (batch, num_filters * len(kernel_sizes))

#         # Fully connected layers
#         out = self.dropout(F.relu(self.fc1(out)))
#         out = torch.sigmoid(self.fc2(out)).squeeze(1)  # (batch,)

#         return out
# src/train_cnn.py
from __future__ import annotations
import argparse, os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.dataset import prepare_sequences, CHAR2IDX
from src.cnn_model_torch import CharCNN


# -------------------------
# Dataset Wrapper
# -------------------------
class URLDataset(Dataset):
    def __init__(self, urls: list[str], labels: list[int], max_len: int = 200):
        self.X = prepare_sequences(urls, max_len=max_len)
        self.y = np.array(labels, dtype=np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.long), torch.tensor(self.y[idx], dtype=torch.float32)


# -------------------------
# Training Function
# -------------------------
def train_cnn_model(
    csv_path: str,
    model_out: str = "models/cnn_best_torch.pt",
    max_len: int = 200,
    batch_size: int = 64,
    epochs: int = 10,
    lr: float = 1e-3,
    patience: int = 3,
):
    # Load dataset
    df = pd.read_csv(csv_path)
    label_col = next((c for c in ["label", "is_phishing", "phishing"] if c in df.columns), None)
    if label_col is None:
        raise ValueError("No label column found in dataset")

    urls = df["url"].astype(str).tolist()
    y = df[label_col].astype(int).tolist()

    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(urls, y, test_size=0.2, stratify=y, random_state=42)

    train_ds = URLDataset(X_train, y_train, max_len=max_len)
    val_ds = URLDataset(X_val, y_val, max_len=max_len)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    # Device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚡ Training on: {device}")

    # Model
    vocab_size = len(CHAR2IDX) + 2  # PAD + UNK
    print("Using vocab size:", vocab_size)
    model = CharCNN(vocab_size=vocab_size).to(device)

    # Class weights for imbalance
    class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
    print("Class weights:", class_weights)

    # Loss, optimizer, scheduler
    criterion = nn.BCELoss(reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        # ---- Training ----
        model.train()
        total_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for Xb, yb in loop:
            Xb, yb = Xb.to(device), yb.to(device)

            optimizer.zero_grad()
            preds = model(Xb)

            # Apply class weights per sample
            sample_weights = torch.tensor([class_weights[int(label.item())] for label in yb], device=device)
            loss = criterion(preds, yb)
            loss = (loss * sample_weights).mean()

            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(yb)
            loop.set_postfix(loss=loss.item())

        train_loss = total_loss / len(train_ds)

        # ---- Validation ----
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                preds = model(Xb)
                loss = nn.BCELoss()(preds, yb)
                val_loss += loss.item() * len(yb)
        val_loss /= len(val_ds)

        # Step scheduler
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # ---- Early Stopping ----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            os.makedirs(os.path.dirname(model_out), exist_ok=True)
            torch.save(model.state_dict(), model_out)
            print(f"✅ Saved best model at epoch {epoch+1}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("⏹ Early stopping triggered.")
                break

    print("🎉 Training complete.")


# -------------------------
# CLI Entry
# -------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/raw/phishing.csv")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=64)
    args = p.parse_args()

    train_cnn_model(args.data, epochs=args.epochs, batch_size=args.batch_size)
