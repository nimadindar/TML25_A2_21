import torch
import torch.optim as optim
import torch.nn.functional as F

import os
import csv
from tqdm import tqdm
from pathlib import Path


from stealing.loss import contrastive_loss

class StolenEncoder:
    def __init__(self, encoder, lr, num_epochs, lambda_value):
        self.encoder = encoder
        self.lr = lr
        self.num_epochs = num_epochs
        self.lambda_value = lambda_value
        # self.data_loader = data_loader
        self.csv_file = "./results/saved_models/stolen_encoder_metrics.csv"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_dir = "./results/saved_models"

        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

    def train_l2(self, dataloader, model_idx):

        self.encoder = self.encoder.to(self.device)
        # criterion = L2Loss()
        optimizer = optim.Adam(self.encoder.parameters(), lr = self.lr)

        file_exists = os.path.exists(self.csv_file)
        with open(self.csv_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['model_id', 'epoch', 'loss'])

        self.encoder.train()
        
        for epoch in range(self.num_epochs):
            running_loss = 0.0
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.num_epochs}", unit="batch")

            for image, views, target in progress_bar:
                image = image.to(self.device)
                target = target.to(self.device).detach()                    # [B, D]

                image_rep = self.encoder(image)
                preds = [self.encoder(v.to(self.device)) for v in views]    # list of [B, D]
                preds = torch.stack(preds, dim=0)                           # [4, B, D]

                target_exp = target.unsqueeze(0).expand_as(preds)           # expand target to [4, B, D] for broadcast

                loss_l1 = F.mse_loss(image_rep, target)
                loss_l2 = F.mse_loss(preds, target_exp)
                loss = loss_l1 + self.lambda_value * loss_l2

                optimizer.zero_grad()
                loss.backward()

                if getattr(self.encoder, "use_centering", True): 
                    with torch.no_grad():
                        self.encoder.update_center(image_rep)

                optimizer.step()

                running_loss += loss.item()

                progress_bar.set_postfix(loss=running_loss / (progress_bar.n + 1))

            epoch_loss = running_loss / len(dataloader)

            print(f"Epoch [{epoch+1}/{self.num_epochs}], Loss: {epoch_loss:.4f}")
            self._write_to_csv(model_idx, epoch, epoch_loss)

        torch.save(self.encoder.state_dict(), Path(self.save_dir) / f"stolen_model_{model_idx}.pth")

    def train_contrastive(self, dataloader, model_idx):
        self.encoder = self.encoder.to(self.device)
        optimizer = optim.Adam(self.encoder.parameters(), lr = self.lr)

        file_exists = os.path.exists(self.csv_file)
        with open(self.csv_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['model_id', 'epoch', 'loss'])

        self.encoder.train()
        
        for epoch in range(self.num_epochs):
            running_loss = 0.0
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.num_epochs}", unit="batch")

            for image, views, target in progress_bar:
                image = image.to(self.device)
                target = target.to(self.device).detach()

                B = image.shape[0]
                
                image_rep = self.encoder(image)  # [B, 1024]
                view_reps = [self.encoder(v.to(self.device)) for v in views]  # list of [B, 1024]
                view_reps = torch.stack(view_reps, dim=0).permute(1, 0, 2)  # [B, 4, 1024]

                all_queries = torch.cat([image_rep.unsqueeze(1), view_reps], dim=1)  # [B, 5, 1024]
                all_queries = all_queries.view(-1, image_rep.size(1))  # [5*B, 1024]
                targets_expanded = target.unsqueeze(1).expand(-1, 5, -1).reshape(-1, target.size(1))  # [5*B, 1024]

                # Construct negatives: for each image, exclude its own target
                neg_indices = [torch.tensor([j for j in range(B) if j != i], device=target.device) for i in range(B)]
                neg_indices = torch.stack(neg_indices).repeat(5, 1)  # [5*B, B-1]
                batch_negatives = target[neg_indices]  # [5*B, B-1, 1024]

                loss = contrastive_loss(all_queries, targets_expanded, batch_negatives)

                optimizer.zero_grad()
                loss.backward()
                if getattr(self.encoder, "use_centering", True):
                    with torch.no_grad():
                        self.encoder.update_center(image_rep)
                optimizer.step()

                running_loss += loss.item()
                progress_bar.set_postfix(loss=running_loss / (progress_bar.n + 1))

            epoch_loss = running_loss / len(dataloader)

            print(f"Epoch [{epoch+1}/{self.num_epochs}], Loss: {epoch_loss:.4f}")
            self._write_to_csv(model_idx, epoch, epoch_loss)

        torch.save(self.encoder.state_dict(), Path(self.save_dir) / f"stolen_model_{model_idx}.pth")

    def _softmax_with_temperature(self, logits, T=0.07):
        return F.softmax(logits / T, dim=1)
    
    def _write_to_csv(self, index, epoch, epoch_loss):
        with open(self.csv_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([index+1, epoch+1, f"{epoch_loss:.4f}"])