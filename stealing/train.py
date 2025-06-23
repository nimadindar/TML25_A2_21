import torch
import torch.optim as optim
import torch.nn.functional as F
import os
import csv
from tqdm import tqdm
from pathlib import Path

class StolenEncoder:
    def __init__(self, encoder, lr, num_epochs, lambda_kd, lambda_siam):
        self.encoder = encoder
        self.lr = lr
        self.num_epochs = num_epochs
        self.lambda_kd = lambda_kd
        self.lambda_siam = lambda_siam
        self.csv_file = "./results/saved_models/stolen_encoder_metrics.csv"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_dir = "./results/saved_models"

        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

    def train(self, train_loader, val_loader, model_idx):
        self.encoder = self.encoder.to(self.device)
        optimizer = optim.Adam(self.encoder.parameters(), lr=self.lr)

        file_exists = os.path.exists(self.csv_file)
        with open(self.csv_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['model_id', 'epoch', 'loss'])

        for epoch in range(self.num_epochs):
            self.encoder.train()
            running_loss = 0.0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}", unit="batch")

            for image, views, target in progress_bar:
                image = image.to(self.device)
                views = [v.to(self.device) for v in views]
                target = target.to(self.device).detach()

                # Student outputs
                image_rep = self.encoder(image)  # [B, 1024]
                view_reps = [self.encoder(v) for v in views]  # list of [B, 1024]
                view_reps = torch.stack(view_reps, dim=0).permute(1, 0, 2)  # [B, K, 1024]

                # KD loss: MSE on original image
                kd_loss = F.mse_loss(image_rep, target)

                # Siamese loss: MSE on all augmented views
                target_exp = target.unsqueeze(1).expand(-1, len(views), -1)  # [B, K, 1024]
                siam_loss = F.mse_loss(view_reps, target_exp)

                loss = self.lambda_kd * kd_loss + self.lambda_siam * siam_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                progress_bar.set_postfix(loss=running_loss / (progress_bar.n + 1))

            epoch_loss = running_loss / len(train_loader)
            print(f"Epoch [{epoch+1}/{self.num_epochs}], Loss: {epoch_loss:.4f}")

            # Validation
            self.encoder.eval()
            val_errs = []
            with torch.no_grad():
                for image, views, target in val_loader:
                    image = image.to(self.device)
                    target = target.to(self.device)
                    pred = self.encoder(image)
                    err = torch.sqrt(((pred - target) ** 2).sum(1))  # L2 distance
                    val_errs.append(err)
            val_l2 = torch.cat(val_errs).mean().item()
            print(f"Validation L2: {val_l2:.4f}")

            self._write_to_csv(model_idx, epoch, epoch_loss)

        torch.save(self.encoder.state_dict(), Path(self.save_dir) / f"stolen_model_{model_idx}.pth")

    def _write_to_csv(self, index, epoch, epoch_loss):
        with open(self.csv_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([index+1, epoch+1, f"{epoch_loss:.4f}"])