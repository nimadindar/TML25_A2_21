import torch
import torch.nn.functional as F

class L2Loss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return torch.norm(x - y, p=2, dim=1).mean()


def contrastive_loss(query, positive, negatives, temperature=0.07):
    """
    query: [M, D] - encoder outputs (M = 5*B for original + 4 views)
    positive: [M, D] - target (API representations)
    negatives: [M, N, D] - negative samples for each query (N = B-1)
    temperature: scalar
    """
    if negatives is None:
        raise ValueError("Contrastive loss requires negative samples")

    query = F.normalize(query, dim=1, eps=1e-8)  # [M, D]
    positive = F.normalize(positive, dim=1, eps=1e-8)  # [M, D]
    negatives = F.normalize(negatives, dim=2, eps=1e-8)  # [M, N, D]

    pos_sim = torch.sum(query * positive, dim=1, keepdim=True)  # [M, 1]
    neg_sim = torch.bmm(query.unsqueeze(1), negatives.transpose(1, 2)).squeeze(1)  # [M, N]
    logits = torch.cat([pos_sim, neg_sim], dim=1)  # [M, 1+N]
    logits /= temperature

    labels = torch.zeros(query.size(0), dtype=torch.long).to(query.device)  # Positives at index 0
    return F.cross_entropy(logits, labels)