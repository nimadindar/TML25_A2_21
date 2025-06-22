import torch
import torch.nn.functional as F

class L2Loss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return torch.norm(x - y, p=2, dim=1).mean()



def contrastive_loss(query, positive, negatives=None, temperature=0.07):
    """
    query: [B, D] - encoder(image)
    positive: [B, D] - target (API rep)
    negatives: Optional [N, D] - other negatives
    """
    query = F.normalize(query, dim=1, eps=1e-8)
    positive = F.normalize(positive, dim=1, eps=1e-8)

    pos_sim = torch.sum(query * positive, dim=1, keepdim=True)  # [B, 1]

    if negatives is not None:
        negatives = F.normalize(negatives, dim=1)
        neg_sim = torch.matmul(query, negatives.T)              # [B, N]
        logits = torch.cat([pos_sim, neg_sim], dim=1)           # [B, 1+N]
    else:
        raise ValueError("To correctly calculate contrastive loss, negative samples should be provided.")

    logits /= temperature

    labels = torch.zeros(query.size(0), dtype=torch.long).to(query.device)  # positives are at index 0

    return F.cross_entropy(logits, labels)