# Titanic Survival Classifier - Code Summary

## Overview
This is a browser-based binary classification application built with TensorFlow.js that predicts Titanic passenger survival. The app includes three major improvements from the original version:
1. **RFC 4180-compliant CSV parser** (fixes quoted field handling)
2. **Fixed evaluation metrics display** (corrected callback conflicts)
3. **Sigmoid gate mechanism for feature importance** (interpretable ML)

---

## Architecture & Data Flow

### 1. **Data Loading & Parsing**
**Files:** `loadData()`, `readFile()`, `parseCSV()`, `parseCSVLine()`

**Logic:**
- User uploads `train.csv` and `test.csv` via file inputs
- Files are read as text using FileReader API
- **KEY FIX:** CSV parser now handles quoted fields with commas correctly
  - Uses state machine with `inQuotes` flag to track quote context
  - Handles escaped quotes (`""` inside quoted strings)
  - Parses line character-by-character instead of naive `split(',')`
  - Example: `"Braund, Mr. Owen Harris"` is now correctly parsed as single field

**Output:** Arrays of JavaScript objects with column headers as keys

---

### 2. **Data Inspection**
**Files:** `inspectData()`, `createPreviewTable()`, `createVisualizations()`

**Logic:**
- Displays first 10 rows in HTML table
- Calculates dataset statistics:
  - Shape (rows × columns)
  - Survival rate (class distribution)
  - Missing value percentages per feature
- Creates visualizations using tfjs-vis:
  - Survival rate by Sex (bar chart)
  - Survival rate by Passenger Class (bar chart)

**Purpose:** Exploratory Data Analysis (EDA) to understand data quality and patterns

---

### 3. **Data Preprocessing**
**Files:** `preprocessData()`, `extractFeatures()`, helper functions

**Logic:**
1. **Imputation** (training data statistics used for both train/test):
   - `Age` → median
   - `Fare` → median
   - `Embarked` → mode

2. **Standardization** (Z-score normalization):
   ```
   standardizedValue = (value - median) / stdDev
   ```

3. **One-Hot Encoding:**
   - `Pclass` → 3 binary features (1st, 2nd, 3rd class)
   - `Sex` → 2 binary features (male, female)
   - `Embarked` → 3 binary features (C, Q, S ports)

4. **Optional Family Features:**
   - `FamilySize = SibSp + Parch + 1`
   - `IsAlone = 1 if FamilySize == 1, else 0`

5. **Tensor Conversion:**
   - Features → `tf.tensor2d` shape `[numSamples, numFeatures]`
   - Labels → `tf.tensor1d` shape `[numSamples]`

**Output:**
- Training: features tensor + labels tensor
- Test: features array + passengerIds array

---

### 4. **Model Architecture with Sigmoid Gate**
**File:** `createModel()`

**KEY INNOVATION:** Uses TensorFlow.js Functional API (not Sequential) to implement interpretable neural network

**Architecture:**
```
Input (k features)
    ↓
Sigmoid Gate Layer (k units, sigmoid activation)
    ↓ (element-wise multiply)
Gated Input = Input ⊙ Gate
    ↓
Hidden Layer (16 units, ReLU)
    ↓
Output Layer (1 unit, sigmoid)
    ↓
Prediction [0, 1]
```

**Sigmoid Gate Mechanism (from Slide 39):**
- **Purpose:** Learn which features are important for prediction
- **Implementation:**
  - Dense layer: `inputDim → inputDim` (diagonal-like transformation)
  - Sigmoid activation: outputs values in [0, 1] per feature
  - **Hadamard product (⊙):** Element-wise multiplication masks input
  - High gate value → feature passes through
  - Low gate value → feature suppressed
- **Interpretability:** Gate weights indicate feature importance

**Compilation:**
- Optimizer: Adam
- Loss: Binary Cross-Entropy
- Metrics: Accuracy

**Parameters:**
- Sigmoid Gate: `k × k` weights (no bias)
- Hidden Layer: `k × 16` weights + 16 biases
- Output Layer: `16 × 1` weights + 1 bias

---

### 5. **Training**
**File:** `trainModel()`

**Logic:**
1. **80/20 Split:** Training data split into train/validation sets
2. **Training Configuration:**
   - Epochs: 50
   - Batch Size: 32
   - Validation: Monitors val_loss and val_acc per epoch
3. **KEY FIX:** Merged callbacks properly using spread operator:
   ```javascript
   callbacks: {
       onEpochEnd: (epoch, logs) => { /* Update UI */ },
       ...tfvis.show.fitCallbacks(...) // Spread tfjs-vis callbacks
   }
   ```
4. **Live Visualization:** tfjs-vis displays loss/accuracy curves in real-time

**Output:**
- Trained model weights
- Validation predictions for evaluation
- Feature importance visualization (new!)

---

### 6. **Feature Importance Extraction**
**File:** `visualizeFeatureImportance()`

**Logic:**
1. **Extract Gate Weights:**
   - Get `sigmoid_gate` layer from model
   - Retrieve kernel weights matrix `[k × k]`

2. **Calculate Importance Scores:**
   - For each output feature `i`: average absolute weights across inputs
   - Normalize to [0, 1] range (divide by max)

3. **Visualization:**
   - Sort features by importance (descending)
   - Display bar chart in tfjs-vis
   - Show top 5 features in UI

**Purpose:** Answer "Which features does the model rely on most?"

**Example Output:**
```
Top 5 Most Important Features:
1. Sex_female: 0.9234
2. Pclass_3: 0.7891
3. Fare: 0.6543
4. Age: 0.5432
5. Sex_male: 0.4321
```

---

### 7. **Evaluation Metrics**
**Files:** `updateMetrics()`, `plotROC()`

**Logic:**
1. **Threshold Slider:** User can adjust classification threshold (default 0.5)
2. **Confusion Matrix:**
   ```
   TP (True Positive): predicted=1, actual=1
   TN (True Negative): predicted=0, actual=0
   FP (False Positive): predicted=1, actual=0
   FN (False Negative): predicted=0, actual=1
   ```

3. **Performance Metrics:**
   - Accuracy = (TP + TN) / Total
   - Precision = TP / (TP + FP)
   - Recall = TP / (TP + FN)
   - F1 Score = 2 × (Precision × Recall) / (Precision + Recall)

4. **ROC Curve & AUC:**
   - Calculate TPR/FPR for 100 threshold values [0.00, 0.01, ..., 0.99]
   - Plot ROC curve using tfjs-vis
   - Calculate AUC using trapezoidal rule

**Interactive:** Metrics update dynamically when threshold slider moves

---

### 8. **Prediction & Export**
**Files:** `predict()`, `exportResults()`

**Logic:**
1. **Prediction:**
   - Convert test features to tensor
   - Run `model.predict(testFeatures)`
   - Apply threshold (default 0.5) to get binary predictions

2. **Export:**
   - `submission.csv`: PassengerId, Survived (binary)
   - `probabilities.csv`: PassengerId, Probability (continuous)
   - Model weights: downloaded as `titanic-tfjs-model.json`

**Output:** Files ready for Kaggle submission

---

## Key Fixes Summary

### ✅ Fix #1: CSV Parser (Task 1)
**Problem:** `line.split(',')` breaks on `"Braund, Mr. Owen Harris"`
**Solution:** State machine parser with quote tracking
**Impact:** Data loads correctly, no garbage input

### ✅ Fix #2: Evaluation Metrics (Task 2)
**Problem:** Duplicate `callbacks` property in `model.fit()` (second overwrites first)
**Solution:** Merge callbacks using spread operator
**Impact:** tfjs-vis charts + UI updates both work

### ✅ Fix #3: Sigmoid Gate (Task 3)
**Problem:** No interpretability - can't see which features matter
**Solution:** Add sigmoid gate layer with Hadamard product
**Impact:** Model explains itself via feature importance scores

---

## Feature Names (Default: 14 features with Family Features enabled)
1. Age (standardized)
2. Fare (standardized)
3. SibSp (raw)
4. Parch (raw)
5. Pclass_1 (one-hot)
6. Pclass_2 (one-hot)
7. Pclass_3 (one-hot)
8. Sex_male (one-hot)
9. Sex_female (one-hot)
10. Embarked_C (one-hot)
11. Embarked_Q (one-hot)
12. Embarked_S (one-hot)
13. FamilySize (computed)
14. IsAlone (binary flag)

---

## Testing Recommendations

### Test Fix #1 (CSV Parser):
```javascript
// Test quoted fields with commas
const testCSV = 'Name,Age\n"Doe, John",30\n"Smith, Jane",25';
const result = parseCSV(testCSV);
console.assert(result[0].Name === "Doe, John"); // Should pass now
```

### Test Fix #2 (Evaluation Metrics):
1. Train model for 2-3 epochs
2. Check if confusion matrix appears in UI
3. Verify ROC curve renders in tfjs-vis visor
4. Move threshold slider → metrics should update

### Test Fix #3 (Feature Importance):
1. Complete training
2. Check "Feature Importance" tab in tfjs-vis visor
3. Verify top 5 features appear in Model Summary section
4. Confirm bar chart shows all 14 features ranked

---

## Browser Compatibility
- **Requires:** Modern browser with ES6+ support (Chrome 90+, Firefox 88+, Safari 14+)
- **No server needed:** Runs entirely client-side
- **Dependencies:** TensorFlow.js and tfjs-vis loaded via CDN

---

## Deployment
1. Upload `index.html` and `app.js` to GitHub repository
2. Enable GitHub Pages (Settings → Pages → main branch)
3. Access at: `https://<username>.github.io/<repo-name>/`

---

## Performance Notes
- Training time: ~10-20 seconds for 50 epochs (depends on hardware)
- Model size: ~5-10KB (lightweight, downloads instantly)
- Feature importance calculation: <1 second post-training

---

## Academic Integrity Note
This code is homework for NNDL course (Neural Networks & Deep Learning).
**Fixes implemented:**
1. CSV parser (RFC 4180 compliance)
2. Evaluation metrics display (callback merge)
3. Sigmoid gate (interpretable ML feature)

**Original source:** https://github.com/dryjins/nndl2025/week2
**Modified by:** [Student Name] for homework submission
