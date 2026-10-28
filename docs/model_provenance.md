# Model Provenance — Turbofan RUL Prediction

## 1. Dataset
- Dataset: NASA C-MAPSS FD001
- Training data: `data/train_FD001.txt`
- Test data: `data/test_FD001.txt`
- Ground-truth RUL: `data/RUL_FD001.txt`
- Raw data contains 26 columns: engine identifier, time cycles, three operational settings, and 21 sensor measurements.

## 2. Model Architecture
- Model type: CNN–LSTM
- Framework: TensorFlow/Keras 2.21.0
- Input shape: `(30, 17)`
- Output shape: `(1,)`
- Architecture: Two Conv1D layers, MaxPooling1D, two LSTM layers, Dropout layers, Dense layers, and a linear regression output.
- Prediction target: Remaining Useful Life (RUL), measured in engine cycles.

## 3. Preprocessing
- Normalization: MinMaxScaler
- Normalization parameters: Derived from FD001 training data
- Window size: 30 cycles
- Window step: 1
- RUL cap: 130 cycles
- Selected model features: 17
- Preprocessing specification: `models/preprocess_spec.json`
- The original training script fits the scaler before splitting the generated windows into training and validation sets. This is a limitation of the existing validation methodology.
- The experiment sketch specifies 25 raw input columns. Their mapping to the model's 17 features must be verified before running experiments.

## 4. Training Configuration
- Optimizer: RMSprop
- Learning rate: 0.001
- Loss function: Mean Squared Error (MSE)
- Evaluation metric: Mean Absolute Error (MAE)
- Maximum epochs: 100
- Batch size: 200
- Early stopping: Validation loss, patience 10
- Checkpoint selection: Best validation loss

## 5. Saved Model
- Model file: `models/best_model.keras`
- Model format: Keras
- Model input: `(None, 30, 17)`
- Model output: `(None, 1)`
- Checkpoint is stored locally and excluded from Git.

## 6. Experiment Deployment
The experiment compares three placements:

1. Cloud-only: Cloud preprocessing and FP32 inference.
2. Hybrid: Edge preprocessing and cloud FP32 inference.
3. Edge-only: Edge preprocessing and INT8 inference.

The current saved checkpoint is the Keras model. INT8 conversion and validation have not yet been completed.

## 7. Outstanding Documentation
- Original model source and licensing information must be confirmed.
- Git commit identifying the trained implementation must be recorded.
- INT8 conversion method and accuracy comparison must be documented after implementation.
- Resource-limit failures identified by preflight must be resolved before collecting valid performance measurements.
