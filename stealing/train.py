import torch
import torch.nn.functional as F
import torch.optim as optim
import os
import csv
from tqdm import tqdm
from pathlib import Path

class StolenEncoder:
    def __init__(self, student_model, optimizer_rate, training_iterations, distill_coeff, invariance_coeff):
        self.student_model = student_model
        self.optimizer_rate = optimizer_rate
        self.training_iterations = training_iterations
        self.distill_coeff = distill_coeff
        self.invariance_coeff = invariance_coeff
        self.csv_file = "./data_models/model_metrics.csv"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_dir = "./results"

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)

    def train(self, train_loader, val_loader, experiment_id):
        self.student_model = self.student_model.to(self.device)
        optimizer = optim.Adam(self.student_model.parameters(), lr=self.optimizer_rate)

        file_exists = os.path.exists(self.csv_file)
        with open(self.csv_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['experiment_id', 'epoch', 'train_loss'])

        for epoch in range(self.training_iterations):
            self.student_model.train()
            running_loss = 0.0
            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.training_iterations}', unit='batch')

            for image, views, label in train_loader:
                image = image.to(self.device)
                views = [v.to(self.device) for v in views]
                target = label.to(self.device).detach()

                # Student outputs
                image_rep = self.student_model(image)  # [B, 1024]
                view_reps = [self.student_model(v) for v in views]  # list of [B, 1024]
                view_reps = torch.stack(view_reps, dim=0).permute(1, 0, 2)  # [B, K, 1024]

                # KD loss: MSE on original image
                kd_loss = F.mse_loss(image_rep, target)

                # Siamese loss: MSE on all augmented views
                target_exp = target.unsqueeze(1).expand(-1, len(views), -1)  # [B, K, 1024]
                siam_loss = F.mse_loss(view_reps, target_exp)

                loss = self.distill_coeff * kd_loss + self.invariance_coeff * siam_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                progress_bar.set_postfix(loss=running_loss / (progress_bar.n + 1))

            epoch_loss = running_loss / len(train_loader)
            print(f'Epoch [{epoch+1}/{self.training_iterations}], Loss: {epoch_loss:.4f}')

            # Validation
            self.student_model.eval()
            val_errs = []
            with torch.no_grad():
                for image, views, target in val_loader:
                    image = image.to(self.device)
                    target = target.to(self.device)
                    pred = self.student_model(image)
                    err = torch.sqrt(((pred - target) ** 2).sum(1))  # L2 distance
                    val_errs.append(err)
            val_l2 = torch.cat(val_errs).mean().item()
            print(f'Validation L2: {val_l2:.4f}')

            self._write_to_csv(experiment_id, epoch, epoch_loss)

        torch.save(self.student_model.state_dict(), Path(self.save_dir) / f'stolen_model_{experiment_id}.pth')

    def _write_to_csv(self, index, epoch, epoch_loss):
        with open(self.csv_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([index+1, epoch+1, f'{epoch_loss:.4f}'])