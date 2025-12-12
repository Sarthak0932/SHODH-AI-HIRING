# Task 2: Deep Learning Model for Loan Default Prediction

## Overview
Built a Multi-Layer Perceptron (MLP) using PyTorch to predict loan default probability as a binary classification problem. The model was trained on 373K+ samples and evaluated using AUC and F1-Score metrics.

---

## Our Approach

### 1. **Problem Framing**
- **Task Type**: Binary classification
- **Target Variable**: `is_default` (0 = Fully Paid, 1 = Defaulted/Late)
- **Objective**: Predict the probability that a loan will default
- **Evaluation Metrics**: AUC-ROC and F1-Score (as required by assignment)

### 2. **Feature Selection**
Selected 19 features across multiple categories:

**Numerical Features:**
- `loan_amnt`, `int_rate`, `installment`, `annual_inc`, `dti`
- `open_acc`, `total_acc`, `revol_bal`, `revol_util`
- `fico_score_avg`, `income_to_loan_ratio`, `credit_history_years`
- `emp_length_years`, `delinq_2yrs`, `inq_last_6mths`, `pub_rec`

**Encoded Categorical Features:**
- `grade_encoded`: Risk grade (A-G) as integers
- `home_encoded`: Home ownership status (RENT, OWN, MORTGAGE, etc.)
- `purpose_encoded`: Loan purpose (debt consolidation, credit card, etc.)

### 3. **Data Preprocessing**
- **Handling Missing Values**: Dropped rows with missing values in selected features
- **Train-Val-Test Split**: 70% train / 15% validation / 15% test (stratified)
- **Standardization**: Applied StandardScaler to normalize features (mean=0, std=1)
- **Tensor Conversion**: Converted to PyTorch tensors for model training
- **DataLoaders**: Created batched data loaders with batch_size=256

### 4. **Model Architecture**

**Multi-Layer Perceptron (MLP) Design:**
```
Input Layer (19 features)
    ↓
Linear(19 → 128) → BatchNorm1D → ReLU → Dropout(0.3)
    ↓
Linear(128 → 64) → BatchNorm1D → ReLU → Dropout(0.3)
    ↓
Linear(64 → 32) → BatchNorm1D → ReLU → Dropout(0.3)
    ↓
Linear(32 → 1) → Sigmoid
    ↓
Output (default probability)
```

**Key Design Choices:**
- **Hidden Layers**: 3 layers with decreasing dimensions (128→64→32)
- **Activation**: ReLU for non-linearity
- **Batch Normalization**: Stabilizes training and speeds convergence
- **Dropout (0.3)**: Prevents overfitting by randomly dropping 30% of neurons
- **Output Activation**: Sigmoid to produce probabilities [0, 1]

**Total Parameters**: ~20,000 trainable parameters

### 5. **Training Configuration**
- **Loss Function**: Binary Cross-Entropy (BCE) Loss
- **Optimizer**: Adam with learning_rate=0.001, weight_decay=1e-5
- **Learning Rate Scheduler**: ReduceLROnPlateau (reduces LR when validation loss plateaus)
- **Epochs**: 30
- **Device**: CUDA (GPU) if available, else CPU

### 6. **Training Process**
- Forward pass: Compute predictions
- Loss calculation: BCE between predictions and true labels
- Backward pass: Compute gradients
- Optimizer step: Update weights
- Validation after each epoch: Monitor overfitting
- Learning rate adjustment: Automatic reduction on plateau

---

## Results

### Test Set Performance

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **AUC** | 0.6842 | Model has moderate ability to distinguish defaulters from non-defaulters |
| **F1-Score** | 0.3316 | Balanced measure considering precision and recall |
| **Accuracy** | 0.8015 | Overall correctness (but less meaningful due to class imbalance) |
| **Test Loss** | ~0.45 | Binary cross-entropy loss on test set |

### Classification Report
```
              precision    recall  f1-score
Fully Paid       0.83      0.93      0.88
Defaulted        0.48      0.26      0.33
```

**Key Observations:**
- High recall for Fully Paid (93%): Good at identifying safe loans
- Low recall for Defaulted (26%): Misses many actual defaults
- Precision-recall tradeoff: Conservative model (fewer false positives)

### Learning Curves
- Training and validation loss decreased steadily
- No significant overfitting (val loss tracked train loss)
- Learning rate reduction improved convergence

---

## Common Questions & Answers

### Q1: Why did you choose AUC and F1-Score as evaluation metrics?
**A:** 
- **AUC (Area Under ROC Curve)**: Threshold-independent metric that measures the model's ability to rank defaulters higher than non-defaulters. Particularly useful for imbalanced datasets. An AUC of 0.68 means the model has a 68% chance of ranking a random defaulter higher than a random non-defaulter.

- **F1-Score**: Harmonic mean of precision and recall, providing a single metric that balances false positives and false negatives. Critical in lending where both types of errors have costs (denying good customers vs. approving bad ones).

### Q2: Why use PyTorch instead of simpler models like logistic regression?
**A:** The assignment specifically required a deep learning model using PyTorch or TensorFlow. Deep learning advantages:
- **Non-linear patterns**: Can capture complex interactions between features
- **Feature learning**: Learns representations automatically through hidden layers
- **Scalability**: Easily extends to larger datasets and more features
- **Flexibility**: Can incorporate advanced techniques (attention, embeddings, etc.)

However, for this particular problem, a well-tuned gradient boosting model (XGBoost) might achieve similar or better performance.

### Q3: What does an AUC of 0.68 mean in business terms?
**A:** 
- **Interpretation**: The model performs moderately better than random guessing (AUC=0.50)
- **Business Impact**: If you present the model with a defaulted loan and a non-defaulted loan, there's a 68% chance it will correctly rank the defaulted loan as higher risk
- **Room for Improvement**: An AUC of 0.68 suggests the model has predictive power but isn't highly discriminative. Financial institutions typically aim for AUC > 0.75 for production models

### Q4: Why is the F1-Score relatively low (0.33) despite 80% accuracy?
**A:** This is due to **class imbalance**:
- ~80% of loans are Fully Paid, so predicting "Fully Paid" for everything gives 80% accuracy
- F1-Score focuses on the minority class (Defaulted), where the model struggles
- The model has low recall (26%) for defaults, meaning it misses many actual defaulters
- **Conclusion**: Accuracy is misleading; F1-Score reveals the true challenge

### Q5: Why use Batch Normalization in the architecture?
**A:** Benefits of Batch Normalization:
1. **Stabilizes training**: Normalizes inputs to each layer, reducing internal covariate shift
2. **Faster convergence**: Allows higher learning rates
3. **Regularization effect**: Adds slight noise, helping prevent overfitting
4. **Improved gradient flow**: Prevents vanishing/exploding gradients

### Q6: How did you prevent overfitting?
**A:** Multiple techniques:
1. **Dropout (0.3)**: Randomly deactivates 30% of neurons during training
2. **Weight Decay (1e-5)**: L2 regularization penalty on large weights
3. **Early Stopping**: Monitor validation loss (could stop if it increases)
4. **Batch Normalization**: Provides mild regularization
5. **Train-Val-Test Split**: Separate validation set for monitoring

### Q7: Why use ReduceLROnPlateau scheduler?
**A:** When training loss stops improving (plateaus), reducing the learning rate helps:
- Fine-tune weights with smaller updates
- Escape local minima
- Achieve better convergence
- Automatically adapts without manual tuning

Configuration: Reduces LR by 50% if validation loss doesn't improve for 3 epochs.

### Q8: What is the threshold for converting probabilities to binary predictions?
**A:** We used **threshold = 0.5** (default):
- Probability < 0.5 → Predict "Fully Paid" (class 0)
- Probability ≥ 0.5 → Predict "Defaulted" (class 1)

**Business Consideration**: In production, you might adjust this threshold based on business costs:
- Lower threshold (e.g., 0.3): More conservative, deny more loans (reduce false negatives)
- Higher threshold (e.g., 0.7): More lenient, approve more loans (reduce false positives)

### Q9: How long did training take?
**A:** With the current configuration:
- **Per Epoch**: ~10-20 seconds (depending on hardware)
- **Total Training**: ~5-10 minutes for 30 epochs
- **Hardware**: GPU significantly faster than CPU (5-10x speedup)

### Q10: Could the model performance be improved?
**A:** Yes, several strategies:

**1. Feature Engineering:**
- More domain-specific features
- Interaction terms (e.g., income * DTI)
- Temporal features (seasonality, economic indicators)

**2. Architecture Tuning:**
- Deeper/wider networks
- Different activation functions (LeakyReLU, ELU)
- Attention mechanisms

**3. Class Imbalance Handling:**
- Weighted loss function (penalize false negatives more)
- SMOTE or oversampling minority class
- Focal loss (focuses on hard examples)

**4. Hyperparameter Optimization:**
- Grid search or Bayesian optimization
- Different learning rates, batch sizes
- Dropout rates, hidden layer sizes

**5. Ensemble Methods:**
- Train multiple models with different initializations
- Combine predictions for robustness

**6. Advanced Techniques:**
- Transfer learning from similar datasets
- AutoML tools for architecture search

### Q11: How does this model compare to the RL agent?
**A:** 
- **DL Model (This Task)**: Predicts **risk** (probability of default)
  - Objective: Maximize classification accuracy
  - Output: Risk score [0, 1]
  - Use case: Risk assessment, credit scoring

- **RL Agent (Task 3)**: Optimizes **decisions** (approve/deny for profit)
  - Objective: Maximize expected financial return
  - Output: Action (0=Deny, 1=Approve)
  - Use case: Loan approval policy

They're complementary: DL provides risk scores, RL uses those scores to make profit-maximizing decisions.

### Q12: What are the limitations of this approach?
**A:** 
1. **Historical Bias**: Trained only on approved loans (no rejected applications)
2. **Temporal Drift**: Economic conditions change; model may degrade over time
3. **Interpretability**: Neural networks are "black boxes" (less explainable than decision trees)
4. **Class Imbalance**: Struggles with minority class (defaults)
5. **Feature Limitations**: Missing behavioral data, real-time signals
6. **Threshold Sensitivity**: Performance varies significantly with decision threshold

### Q13: How would you deploy this model in production?
**A:** Production pipeline:
1. **Model Serialization**: Save model weights (`torch.save()`)
2. **API Wrapper**: Create REST API (Flask/FastAPI)
3. **Input Validation**: Ensure features match training schema
4. **Preprocessing Pipeline**: Apply same StandardScaler
5. **Prediction Service**: Return probabilities and explanations
6. **Monitoring**: Track prediction distribution, performance metrics
7. **Retraining Schedule**: Retrain monthly/quarterly on fresh data
8. **A/B Testing**: Compare against current system gradually

---

## Technical Details

### Model Code
```python
class LoanDefaultMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], dropout_rate=0.3):
        super(LoanDefaultMLP, self).__init__()
        # Architecture defined in notebook
```

### Training Loop
```python
for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate(model, val_loader, criterion, device)
    scheduler.step(val_loss)
```

### Files Generated
- **Trained Model**: In-memory PyTorch model (can be saved with `torch.save()`)
- **Predictions**: Test probabilities for 93K+ test samples
- **Visualizations**: Training curves, ROC curve, confusion matrix

**Last Updated**: December 8, 2025
