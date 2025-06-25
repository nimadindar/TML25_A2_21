# TML25_A2_21: Model Stealing Attack

## Project Overview

This project implements a Model Stealing Attack to replicate a black-box ResNet-based encoder protected by B4B defense, which adds noise to API output representations. The goal is to minimize the L2 distance between a student model’s 1024-dimensional outputs and the target encoder’s on a private test set, evaluated via a remote server. Using `ModelStealingPub.pt` and API-queried representations (`out0.pickle`,...), five methods were tested, achieving a best L2 score of 5.46226692199707.

## Implemented Methods

Five methods were tested to steal the B4B-protected encoder:

1. **Pretrained ResNet20**:
   - **Rationale**: Used pretrained ResNet20 to leverage CIFAR-10 features, aiming to align with noisy target outputs.
   - **Implementation**: Set `BACKBONE_TYPE="resnet20"` in `cnn_encoder.py`, fine-tuned with MSE loss on 4000 images (4 API queries).
   - **Outcome**: L2 score ~6.47, limited by pretrained weight misalignment with B4B noise.
   - **Parameters**: `LEARNING_RATE=1e-3`, `epochs=30`, `TRAIN_BATCH_SIZE=32`.

2. **ResNet20 Tuning**:
   - **Rationale**: Tuned parameters and augmentations to improve ResNet20’s robustness to B4B noise.
   - **Implementation**: Varied `LEARNING_RATE=[1e-3, 5e-4, 1e-4]`, `epochs=[10, 15, 20]`, tried different augmentation methods.
   - **Outcome**: No improvement beyond 6.47, due to ResNet20’s limited capacity.
   - **Parameters**: Tested multiple learning rates, augmentations.

3. **ResNet18 with Modified Inputs**:
   - **Rationale**: Adopted ResNet18, modified for 3x32x32 inputs, to counter B4B noise with higher capacity.
   - **Implementation**: Used `BACKBONE_TYPE="resnet18"`, adjusted `conv1` to 3x3 kernel. Trained on 1000 images (750/250 split) with (`DISTILLATION_WEIGHT=1.0`) and (`INVARIANCE_WEIGHT=0.5`).
   - **Outcome**: Best L2 score ~5.46.
   - **Parameters**: `LEARNING_RATE=3e-4`, `epochs=15`, `TRAIN_BATCH_SIZE=64`.


4. **L2 Normalization in Outputs**:
   - **Rationale**: Normalized outputs to match target encoder scaling, reducing B4B noise impact.
   - **Implementation**: Added `F.normalize(out, dim=1)` in `cnn_encoder.py`, retrained ResNet18.
   - **Outcome**: Worse L2 score.
   - **Parameters**: Same as ResNet18 setup.


## Folder Structure

### Folder Structure

```
model_stealing_project/
├── 📁 data/                       # Directory for public dataset
│   ├── 📄 ModelStealingPub.pt     # Public dataset with images
├── 📁 dataset/                    # Directory for dataset-related scripts
│   ├── 📄 dataset.py              # Loads dataset and applies augmentations
│   ├── 📄 create_subset.py        # Creates random dataset subsets
├── 📁 encoders/                   # Student models directory
│   ├── 📄 cnn_encoder.py          # Defines student model
├── 📁 results/                    # Directory for model checkpoints and metrics
│   ├── 📁 saved_models/           # Stores trained models and ONNX files
│   ├── 📄 model_metrics.csv       # Logs training and validation metrics
│   ├── 📄 out0.pickle,...         # API-queried representation files
├── 📁 stealing/                   # Directory for dstealing-related scripts
│   ├── 📄 loss.py                 # Defines loss functions
│   ├── 📄 query_api.py            # Handles API queries and model submission
│   ├── 📄 train.py                # Implements training
├── 📄 main.py                     # Orchestrates API querying, training, and ONNX model submission
├── 📄 config.py                   # Defines API, training, and augmentation configurations
├── 📄 requirements.txt            # Lists Python dependencies
├── 📄 .env                        # Stores environment variables



```

## Results

| **Method**                     | **Output File**            | **Description**                                           | **Key Features**                               | **L2 Score** |
|--------------------------------|----------------------------|-----------------------------------------------------------|------------------------------------------------|--------------|
| Pretrained ResNet20            | `stolen_model_0.pth`       | Pretrained ResNet20, 4000 images.                | `LEARNING_RATE=1e-3`, 4 queries                | * ~6.47        |
| ResNet18 with Modified Inputs  | `stolen_model_1.pth`       | ResNet18, 3x32x32 inputs, 1000 images.     | `DISTILLATION_WEIGHT=1.0`, `INVARIANCE_WEIGHT=0.5` | ~5.46        |
| Changed Mean and Std           | `stolen_model_2.pth`       | CIFAR-10 normalization with ResNet18.                     | `MEAN=[0.4914, ...]` | ~5.55        |
| L2 Normalization in Outputs    | `stolen_model_3.pth`       | L2-normalized ResNet18 outputs.                           | `F.normalize(out, dim=1)`                      | ~25.46        |

** tried different parameters, best score is reported

**Note**: Best score (~5.46) used ResNet18, stored in `stolen_model_2.pth` 

## Dependencies

`requirements.txt`:
```
torch
torchvision
numpy
onnxruntime
requests
python-dotenv
tqdm
pickle
```


## Usage

1. **Request API**: Set `REQUEST_NEW_API=True` in `main.py`, run `python main.py`. Update `APIConfig.SEED`, `APIConfig.PORT`.
2. **Query API**: Set `QUERY=True`, run `python main.py`. Outputs `out{IDX}.pickle` (4 queries, 4000 images).
3. **Train**: Set `STEAL=True`, run `python main.py`. Saves `stolen_model_{EXPERIMENT_ID}.pth`.
4. **Submit**: Set `SUBMIT=True`, run `python main.py`. Exports ONNX, submits with `APIConfig.SEED`. Ensure 3x32x32 input, 1024 output.

## Conclusion

Achieved the best L2 ~5.46 against B4B-protected encoder using ResNet18 student model and several augmentations.

## References

1. Dubiński, J., Pawlak, S., Boenisch, F., Trzcinski, T., & Dziedzic, A. (2023). Bucks for Buckets (B4B): Active Defenses Against Stealing Encoders. In Advances in Neural Information Processing Systems, 36 (NeurIPS 2023).
https://proceedings.neurips.cc/paper_files/paper/2023/hash/ad1efab57a04d93f097e7fbb2d4fc054-Abstract-Conference.html

2. Liu, Y., Jia, J., Liu, H., & Gong, N. Z. (2023). StolenEncoder: Stealing Pre-trained Encoders in Self-supervised Learning. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).
https://arxiv.org/abs/2201.05889

3. Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the Knowledge in a Neural Network. arXiv preprint arXiv:1503.02531.
https://arxiv.org/abs/1503.02531