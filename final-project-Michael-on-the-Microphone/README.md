## Problem
Multi-class image classification (9 waste categories) using **transfer learning** with frozen backbones and a custom classification head.

## Data & splits
- Per class/folder, **first 80%** images → train; **last 20%** → test.
- Within train, **random 20% per class** as validation.
- Preprocess: resize or zero-pad to backbone input; one-hot encoded labels.

## Models
Pretrained CNN backbones with **all conv blocks frozen**; train only the head:
- **ResNet50**, **ResNet100**, **EfficientNetB0**, **VGG16** (as specified).
- Head: Global pooling → Dense(s) with **ReLU**, **batch norm**, **L2 reg**, **dropout=0.2** → **Softmax**.

## Optimization
- **Adam**, categorical cross-entropy.
- **Augmentation** (train-time): random crops, zoom, rotations, flips, contrast/translation.
- Train **≥50 epochs** (up to 100) with **early stopping** on validation loss; checkpoint best weights.

## Evaluation
- Report **Precision/Recall/F1/AUC** on **train/val/test**; classwise confusion matrix; training curves (loss/error vs epochs).
- Compare backbones: capacity vs generalization; note any overfitting mitigated by augmentation/dropout/L2.

## Artifacts
- Saved heads (and optionally full models) per backbone.
- Plots: history curves; confusion matrices; sample predictions.

## Tech used
`tensorflow`/`keras`, `opencv-python` (I/O/resize), `numpy`, `pandas`, `matplotlib`.

## ML tasks
**Transfer learning**, **regularization via augmentation/L2/BN/dropout**, **model selection** among backbones, **early stopping** & checkpointing.

## Notes
- Keep preprocessing **identical** across backbones for fair comparison.
- If class imbalance exists, consider **class weights** or **balanced sampling** (document if applied).
