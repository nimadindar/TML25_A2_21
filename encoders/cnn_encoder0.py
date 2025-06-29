import torch
import torch.nn as nn

class CNNencoder(nn.Module):
    """Class for loading CNN-based pretrained encoders"""
    def __init__(self, model_name, output_dim=1024, pretrained=True):
        super().__init__()
        self.model_name = model_name
        self.output_dim = output_dim
        self.pretrained = pretrained
        self.avgpool_output = None

        available_models = torch.hub.list("chenyaofo/pytorch-cifar-models", force_reload=False)
        if self.model_name not in available_models:
            raise ValueError(f"Invalid model_name: '{model_name}'. Available models are: {available_models}")

        try:
            self.encoder = torch.hub.load("chenyaofo/pytorch-cifar-models", model_name, pretrained=pretrained)
        except Exception as e:
            raise RuntimeError(f"Failed to load model '{model_name}' from torch.hub: {str(e)}")

        self.encoder.avgpool.register_forward_hook(self._hook_fn)

        self.register_buffer("center", torch.zeros(output_dim))
        self.center_momentum = 0.9
        self.use_centering = True  

        input_features = 64  # Adjust based on model architecture
        self.projection = nn.Linear(input_features, output_dim)

    def _hook_fn(self, module, input, output):
        self.avgpool_output = output

    def forward(self, x):
        # if x.dim() != 4 or x.shape[1:] != (3, 32, 32):
        #     raise ValueError(f"Expected input shape [batch_size, 3, 32, 32], got {x.shape}")

        _ = self.encoder(x)

        if self.avgpool_output is None:
            raise RuntimeError("Avgpool output not captured. Check hook registration.")

        features = self.avgpool_output.squeeze(-1).squeeze(-1)

        output = self.projection(features)

        if self.use_centering:
            output = output - self.center  
        return output
        
    
    @torch.no_grad()
    def update_center(self, output):
        """
        output: Tensor of shape [batch_size, output_dim]
        """
        batch_center = output.mean(dim=0)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

    

