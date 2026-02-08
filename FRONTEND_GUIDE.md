# Frontend UI Workflow - Complete Guide

This document explains every section of the Titanic Survival Classifier web interface, including the purpose, code implementation, and data flow for each component.

---

## Table of Contents
1. [Data Load](#1-data-load)
2. [Data Inspection](#2-data-inspection)
3. [Preprocessing](#3-preprocessing)
4. [Model Setup](#4-model-setup)
5. [Training](#5-training)
6. [Evaluation Metrics](#6-evaluation-metrics)
7. [Prediction](#7-prediction)
8. [Export Results](#8-export-results)

---

## 1. Data Load

### Purpose
Upload training and test CSV files from the user's local filesystem and parse them into JavaScript objects.

### UI Elements
```html
<input type="file" id="train-file" accept=".csv">
<input type="file" id="test-file" accept=".csv">
<button id="load-data-btn" onclick="loadData()">Load Data</button>
<div id="data-status"></div>
```

### Code Flow

#### 1.1 Button Click → `loadData()`
**Location:** `app.js` lines 121-157

**What Happens:**
1. Gets file input elements by ID
2. Validates that both files are selected
3. Reads both files using `readFile()` helper
4. Calls `parseCSV()` on each file's text content
5. Stores results in global variables `trainData` and `testData`
6. Displays success message with row counts
7. Enables "Inspect Data" button

```javascript
async function loadData() {
    const trainFile = document.getElementById('train-file').files[0];
    const testFile = document.getElementById('test-file').files[0];

    if (!trainFile || !testFile) {
        alert('Please select both training and test files.');
        return;
    }

    // Read files
    const trainText = await readFile(trainFile);
    const testText = await readFile(testFile);

    // Parse CSV
    trainData = parseCSV(trainText);
    testData = parseCSV(testText);

    // Update UI
    statusDiv.innerHTML = `Data loaded successfully! Training: ${trainData.length} samples, Test: ${testData.length} samples`;
    document.getElementById('inspect-btn').disabled = false;
}
```

#### 1.2 Helper: `readFile(file)`
**Location:** `app.js` lines 46-59

**Purpose:** Reads a File object as text using FileReader API.

**How It Works:**
- Returns a Promise that resolves when file is read
- Uses `FileReader.readAsText()` to load file content
- Handles load success (`resolve`) and errors (`reject`)

```javascript
function readFile(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = (e) => resolve(e.target.result);
        reader.onerror = (e) => reject(e);
        reader.readAsText(file);
    });
}
```

#### 1.3 CSV Parser: `parseCSV(csvText)`
**Location:** `app.js` lines 61-119

**Purpose:** Parse CSV text into array of objects, handling quoted fields with commas.

**Key Features:**
- **RFC 4180 compliant** - handles quoted fields correctly
- Uses state machine via `parseCSVLine()` helper
- Auto-converts numeric strings to numbers
- Converts "NULL" strings to `null`

**How It Works:**
1. Split text by newlines
2. Parse first line as headers using `parseCSVLine()`
3. For each data line:
   - Parse values using `parseCSVLine()`
   - Create object with header keys and values
   - Convert numeric strings to numbers
   - Convert "NULL" to `null`

```javascript
function parseCSV(csvText) {
    const lines = csvText.trim().split('\n');
    const headers = parseCSVLine(lines[0]);

    const data = [];
    for (let i = 1; i < lines.length; i++) {
        const values = parseCSVLine(lines[i]);
        const row = {};

        headers.forEach((header, index) => {
            let value = values[index]?.trim() || '';

            // Convert to number if possible
            if (!isNaN(value) && value !== '') {
                value = Number(value);
            } else if (value === 'NULL' || value === '') {
                value = null;
            }

            row[header] = value;
        });

        data.push(row);
    }

    return data;
}
```

#### 1.4 Line Parser: `parseCSVLine(line)`
**Location:** `app.js` lines 79-107

**Purpose:** Parse a single CSV line character-by-character, tracking quote context.

**State Machine Logic:**
- Tracks `inQuotes` boolean flag
- Only splits on commas when NOT inside quotes
- Handles escaped quotes (`""` inside quoted strings)

```javascript
function parseCSVLine(line) {
    const result = [];
    let current = '';
    let inQuotes = false;

    for (let i = 0; i < line.length; i++) {
        const char = line[i];
        const nextChar = line[i + 1];

        if (char === '"') {
            if (inQuotes && nextChar === '"') {
                current += '"';  // Escaped quote
                i++;
            } else {
                inQuotes = !inQuotes;  // Toggle quote mode
            }
        } else if (char === ',' && !inQuotes) {
            result.push(current);  // Field separator
            current = '';
        } else {
            current += char;
        }
    }

    result.push(current);
    return result;
}
```

**Example:**
```javascript
parseCSVLine('"Braund, Mr. Owen Harris",22,male')
// Returns: ["Braund, Mr. Owen Harris", "22", "male"]
// ✓ Comma inside quotes is preserved!
```

### Data Structure After Loading
```javascript
trainData = [
    {
        PassengerId: 1,
        Survived: 0,
        Pclass: 3,
        Name: "Braund, Mr. Owen Harris",
        Sex: "male",
        Age: 22,
        SibSp: 1,
        Parch: 0,
        Ticket: "A/5 21171",
        Fare: 7.25,
        Cabin: null,
        Embarked: "S"
    },
    // ... 890 more rows
]
```

---

## 2. Data Inspection

### Purpose
Display data preview, statistics, and visualizations to understand the dataset before training.

### UI Elements
```html
<button id="inspect-btn" onclick="inspectData()">Inspect Data</button>
<div id="data-preview"></div>
<div id="data-stats"></div>
<div id="charts"></div>
```

### Code Flow

#### 2.1 Button Click → `inspectData()`
**Location:** `app.js` lines 159-244

**What Happens:**
1. Creates preview table (first 10 rows)
2. Calculates dataset statistics
3. Generates survival rate charts
4. Enables "Preprocess Data" button

```javascript
function inspectData() {
    if (!trainData) {
        alert('Please load data first.');
        return;
    }

    // Create preview table
    document.getElementById('data-preview').innerHTML = '<h3>Data Preview (First 10 Rows)</h3>';
    document.getElementById('data-preview').appendChild(createPreviewTable(trainData.slice(0, 10)));

    // Calculate statistics
    createDataStatistics();

    // Create visualizations
    createVisualizations();

    // Enable preprocessing button
    document.getElementById('preprocess-btn').disabled = false;
}
```

#### 2.2 Helper: `createPreviewTable(data)`
**Location:** `app.js` lines 109-134

**Purpose:** Generate HTML table from array of objects.

**How It Works:**
1. Create `<table>` element
2. Add header row from first object's keys
3. Add data rows for each object
4. Replace `null` with "NULL" for display

```javascript
function createPreviewTable(data) {
    const table = document.createElement('table');

    // Header row
    const headerRow = document.createElement('tr');
    Object.keys(data[0]).forEach(key => {
        const th = document.createElement('th');
        th.textContent = key;
        headerRow.appendChild(th);
    });
    table.appendChild(headerRow);

    // Data rows
    data.forEach(row => {
        const tr = document.createElement('tr');
        Object.values(row).forEach(value => {
            const td = document.createElement('td');
            td.textContent = value === null ? 'NULL' : value;
            tr.appendChild(td);
        });
        table.appendChild(tr);
    });

    return table;
}
```

#### 2.3 Statistics Calculation
**Location:** `app.js` lines 136-157

**Calculates:**
- Dataset shape (rows × columns)
- Survival rate (percentage)
- Missing values per feature

```javascript
function createDataStatistics() {
    const statsDiv = document.getElementById('data-stats');
    statsDiv.innerHTML = '<h3>Data Statistics</h3>';

    // Shape
    const shapeInfo = `Dataset shape: ${trainData.length} rows x ${Object.keys(trainData[0]).length} columns`;

    // Survival rate
    const survivalCount = trainData.filter(row => row[TARGET_FEATURE] === 1).length;
    const survivalRate = (survivalCount / trainData.length * 100).toFixed(2);
    const targetInfo = `Survival rate: ${survivalCount}/${trainData.length} (${survivalRate}%)`;

    // Missing values
    let missingInfo = '<h4>Missing Values Percentage:</h4><ul>';
    Object.keys(trainData[0]).forEach(feature => {
        const missingCount = trainData.filter(row => row[feature] === null || row[feature] === undefined).length;
        const missingPercent = (missingCount / trainData.length * 100).toFixed(2);
        missingInfo += `<li>${feature}: ${missingPercent}%</li>`;
    });
    missingInfo += '</ul>';

    statsDiv.innerHTML += `<p>${shapeInfo}</p><p>${targetInfo}</p>${missingInfo}`;
}
```

#### 2.4 Visualizations: `createVisualizations()`
**Location:** `app.js` lines 188-243

**Creates:**
1. Survival rate by Sex (bar chart)
2. Survival rate by Passenger Class (bar chart)

**How It Works:**

**Step 1: Aggregate survival data by Sex**
```javascript
const survivalBySex = {};
trainData.forEach(row => {
    if (row.Sex && row.Survived !== undefined) {
        if (!survivalBySex[row.Sex]) {
            survivalBySex[row.Sex] = { survived: 0, total: 0 };
        }
        survivalBySex[row.Sex].total++;
        if (row.Survived === 1) {
            survivalBySex[row.Sex].survived++;
        }
    }
});
```

**Step 2: Format for tfjs-vis**
```javascript
const sexData = Object.entries(survivalBySex).map(([sex, stats]) => ({
    index: sex,
    value: (stats.survived / stats.total) * 100
}));
```

**Step 3: Render bar chart**
```javascript
tfvis.render.barchart(
    { name: 'Survival Rate by Sex', tab: 'Charts' },
    sexData,
    { xLabel: 'Sex', yLabel: 'Survival Rate (%)' }
);
```

**Same logic for Passenger Class chart.**

### Output Example
```
Data Preview (First 10 Rows)
┌─────────────┬──────────┬────────┬─────────────────────────┬────────┬─────┐
│ PassengerId │ Survived │ Pclass │ Name                    │ Sex    │ Age │
├─────────────┼──────────┼────────┼─────────────────────────┼────────┼─────┤
│ 1           │ 0        │ 3      │ Braund, Mr. Owen Harris │ male   │ 22  │
└─────────────┴──────────┴────────┴─────────────────────────┴────────┴─────┘

Data Statistics
Dataset shape: 891 rows x 12 columns
Survival rate: 342/891 (38.38%)

Missing Values Percentage:
• Age: 19.87%
• Cabin: 77.10%
• Embarked: 0.22%
```

---

## 3. Preprocessing

### Purpose
Transform raw data into numerical features suitable for neural network training.

### UI Elements
```html
<button id="preprocess-btn" onclick="preprocessData()">Preprocess Data</button>
<input type="checkbox" id="add-family-features" checked>
<div id="preprocessing-output"></div>
```

### Code Flow

#### 3.1 Button Click → `preprocessData()`
**Location:** `app.js` lines 246-304

**Pipeline:**
1. Calculate imputation values from training data
2. Extract features from training data
3. Extract features from test data
4. Convert to TensorFlow tensors

```javascript
function preprocessData() {
    if (!trainData || !testData) {
        alert('Please load data first.');
        return;
    }

    try {
        // Step 1: Calculate imputation values from training data ONLY
        const ageMedian = calculateMedian(trainData.map(row => row.Age).filter(age => age !== null));
        const fareMedian = calculateMedian(trainData.map(row => row.Fare).filter(fare => fare !== null));
        const embarkedMode = calculateMode(trainData.map(row => row.Embarked).filter(e => e !== null));

        // Step 2: Preprocess training data
        preprocessedTrainData = {
            features: [],
            labels: []
        };

        trainData.forEach(row => {
            const features = extractFeatures(row, ageMedian, fareMedian, embarkedMode);
            preprocessedTrainData.features.push(features);
            preprocessedTrainData.labels.push(row[TARGET_FEATURE]);
        });

        // Step 3: Preprocess test data
        preprocessedTestData = {
            features: [],
            passengerIds: []
        };

        testData.forEach(row => {
            const features = extractFeatures(row, ageMedian, fareMedian, embarkedMode);
            preprocessedTestData.features.push(features);
            preprocessedTestData.passengerIds.push(row[ID_FEATURE]);
        });

        // Step 4: Convert to tensors
        preprocessedTrainData.features = tf.tensor2d(preprocessedTrainData.features);
        preprocessedTrainData.labels = tf.tensor1d(preprocessedTrainData.labels);

        // Display success
        outputDiv.innerHTML = `
            <p>Preprocessing completed!</p>
            <p>Training features shape: ${preprocessedTrainData.features.shape}</p>
            <p>Training labels shape: ${preprocessedTrainData.labels.shape}</p>
            <p>Test features shape: [${preprocessedTestData.features.length}, ${preprocessedTestData.features[0].length}]</p>
        `;

        // Enable model creation
        document.getElementById('create-model-btn').disabled = false;
    } catch (error) {
        outputDiv.innerHTML = `Error during preprocessing: ${error.message}`;
        console.error(error);
    }
}
```

#### 3.2 Feature Extraction: `extractFeatures(row, ageMedian, fareMedian, embarkedMode)`
**Location:** `app.js` lines 306-348

**Transformations:**

**1. Imputation (fill missing values)**
```javascript
const age = row.Age !== null ? row.Age : ageMedian;
const fare = row.Fare !== null ? row.Fare : fareMedian;
const embarked = row.Embarked || embarkedMode;
```

**2. Standardization (Z-score normalization)**
```javascript
const ageStd = (age - ageMedian) / calculateStdDev(trainData.map(r => r.Age).filter(a => a !== null));
const fareStd = (fare - fareMedian) / calculateStdDev(trainData.map(r => r.Fare).filter(f => f !== null));
```

**3. One-Hot Encoding**
```javascript
// Pclass: 1, 2, 3 → [1,0,0], [0,1,0], [0,0,1]
const pclass1 = row.Pclass === 1 ? 1 : 0;
const pclass2 = row.Pclass === 2 ? 1 : 0;
const pclass3 = row.Pclass === 3 ? 1 : 0;

// Sex: male, female → [1,0], [0,1]
const sexMale = row.Sex === 'male' ? 1 : 0;
const sexFemale = row.Sex === 'female' ? 1 : 0;

// Embarked: C, Q, S → [1,0,0], [0,1,0], [0,0,1]
const embarkedC = embarked === 'C' ? 1 : 0;
const embarkedQ = embarked === 'Q' ? 1 : 0;
const embarkedS = embarked === 'S' ? 1 : 0;
```

**4. Family Features (optional)**
```javascript
const familySize = row.SibSp + row.Parch + 1;
const isAlone = familySize === 1 ? 1 : 0;
```

**5. Assemble feature vector**
```javascript
const features = [
    ageStd,
    fareStd,
    row.SibSp,
    row.Parch,
    pclass1, pclass2, pclass3,
    sexMale, sexFemale,
    embarkedC, embarkedQ, embarkedS,
    familySize,
    isAlone
];

return features;  // Array of 14 numbers
```

#### 3.3 Helper Functions

**`calculateMedian(arr)`** - Location: `app.js` lines 350-355
```javascript
function calculateMedian(arr) {
    const sorted = arr.sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 === 0
        ? (sorted[mid - 1] + sorted[mid]) / 2
        : sorted[mid];
}
```

**`calculateMode(arr)`** - Location: `app.js` lines 357-368
```javascript
function calculateMode(arr) {
    const counts = {};
    arr.forEach(val => counts[val] = (counts[val] || 0) + 1);
    return Object.keys(counts).reduce((a, b) => counts[a] > counts[b] ? a : b);
}
```

**`calculateStdDev(arr)`** - Location: `app.js` lines 370-374
```javascript
function calculateStdDev(arr) {
    const mean = arr.reduce((sum, val) => sum + val, 0) / arr.length;
    const variance = arr.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / arr.length;
    return Math.sqrt(variance);
}
```

### Data Transformation Example
```javascript
// Raw data
{
    Age: 22,
    Fare: 7.25,
    Pclass: 3,
    Sex: "male",
    Embarked: "S",
    SibSp: 1,
    Parch: 0
}

// After preprocessing
[
    -0.565,  // Age (standardized)
    -0.502,  // Fare (standardized)
    1,       // SibSp
    0,       // Parch
    0, 0, 1, // Pclass (one-hot: 3rd class)
    1, 0,    // Sex (one-hot: male)
    0, 0, 1, // Embarked (one-hot: S)
    2,       // FamilySize
    0        // IsAlone
]
// 14 numerical features ready for neural network!
```

---

## 4. Model Setup

### Purpose
Create a neural network with Sigmoid Gate layer for interpretable feature importance.

### UI Elements
```html
<button id="create-model-btn" onclick="createModel()">Create Model</button>
<div id="model-summary"></div>
```

### Code Flow

#### 4.1 Button Click → `createModel()`
**Location:** `app.js` lines 376-457

**Architecture:**
```
Input (14 features)
    ↓
Sigmoid Gate Layer (14→14, sigmoid) [Named: 'sigmoid_gate']
    ↓
Hadamard Product (element-wise multiply)
    ↓
Hidden Layer (14→16, ReLU)
    ↓
Output Layer (16→1, sigmoid)
    ↓
Prediction (probability)
```

**Code Implementation:**

```javascript
function createModel() {
    if (!preprocessedTrainData) {
        alert('Please preprocess data first.');
        return;
    }

    const summaryDiv = document.getElementById('model-summary');
    summaryDiv.innerHTML = '<h3>Model Summary (with Sigmoid Gate for Feature Importance)</h3>';

    try {
        // Get number of features
        const k = preprocessedTrainData.features.shape[1];

        // Store feature names globally
        featureNames = [
            'Age', 'Fare', 'SibSp', 'Parch',
            'Pclass_1', 'Pclass_2', 'Pclass_3',
            'Sex_male', 'Sex_female',
            'Embarked_C', 'Embarked_Q', 'Embarked_S',
            'FamilySize', 'IsAlone'
        ];

        // Build model using Functional API
        const input = tf.input({shape: [k]});

        // Sigmoid Gate Layer (k → k)
        // Purpose: Learn feature importance weights
        const gateLayer = tf.layers.dense({
            units: k,
            activation: 'sigmoid',
            name: 'sigmoid_gate',
            kernelInitializer: 'ones',  // Start neutral (all features pass through)
            useBias: false  // No bias term needed
        });
        const gate = gateLayer.apply(input);

        // Hadamard Product: Input ⊙ Gate
        // Purpose: Multiply input by learned importance weights
        const gatedInput = tf.layers.multiply().apply([input, gate]);

        // Hidden Layer (k → 16)
        const hidden = tf.layers.dense({
            units: 16,
            activation: 'relu'
        }).apply(gatedInput);

        // Output Layer (16 → 1)
        const output = tf.layers.dense({
            units: 1,
            activation: 'sigmoid'
        }).apply(hidden);

        // Create model
        model = tf.model({inputs: input, outputs: output});

        // Compile model
        model.compile({
            optimizer: 'adam',
            loss: 'binaryCrossentropy',
            metrics: ['accuracy']
        });

        // Display summary
        summaryDiv.innerHTML += `
            <ul>
                <li><strong>Input Layer:</strong> ${k} features</li>
                <li><strong>Sigmoid Gate Layer:</strong> ${k} units (learns feature importance)</li>
                <li><strong>Gated Input:</strong> Element-wise multiplication (Hadamard product)</li>
                <li><strong>Hidden Layer:</strong> 16 units with ReLU activation</li>
                <li><strong>Output Layer:</strong> 1 unit with Sigmoid activation (binary classification)</li>
            </ul>
            <p>Total parameters: ${model.countParams()}</p>
            <p><em>Note: The sigmoid gate layer learns which features are most important for prediction.</em></p>
        `;

        // Enable training button
        document.getElementById('train-btn').disabled = false;
    } catch (error) {
        summaryDiv.innerHTML = `Error creating model: ${error.message}`;
        console.error(error);
    }
}
```

### Why Functional API vs Sequential?

**Sequential API (not used):**
```javascript
model = tf.sequential();
model.add(tf.layers.dense({units: 16, activation: 'relu', inputShape: [14]}));
model.add(tf.layers.dense({units: 1, activation: 'sigmoid'}));
```
- **Limitation:** Only supports linear stacks (layer1 → layer2 → layer3)
- **Cannot do:** Branching, multiple inputs/outputs, layer merging

**Functional API (used):**
```javascript
const input = tf.input({shape: [14]});
const gate = gateLayer.apply(input);
const gatedInput = tf.layers.multiply().apply([input, gate]);
```
- **Advantage:** Supports complex architectures with branching
- **Needed for:** Sigmoid gate requires parallel paths (input → gate, then merge)

### Sigmoid Gate Mechanism

**Mathematical Operation:**
```
Let x = input features [x₁, x₂, ..., x₁₄]
Let w = gate weights [w₁, w₂, ..., w₁₄]

Gate output: g = σ(Wx) where σ = sigmoid
             g = [σ(w₁x₁), σ(w₂x₂), ..., σ(w₁₄x₁₄)]

Gated input: x' = x ⊙ g (Hadamard product)
             x' = [x₁·g₁, x₂·g₂, ..., x₁₄·g₁₄]
```

**Interpretation:**
- If `gᵢ ≈ 1`: Feature `i` is important, passes through fully
- If `gᵢ ≈ 0`: Feature `i` is unimportant, gets suppressed
- During training, the model learns which features matter!

### Model Parameters Breakdown
```
Sigmoid Gate Layer: 14 × 14 = 196 weights (no bias)
Hidden Layer:       14 × 16 + 16 = 240 weights + 16 biases
Output Layer:       16 × 1 + 1 = 16 weights + 1 bias
─────────────────────────────────────────────────────────
Total:              196 + 240 + 16 + 16 + 1 = 453 parameters
```

---

## 5. Training

### Purpose
Train the neural network on preprocessed data with real-time visualization and validation.

### UI Elements
```html
<button id="train-btn" onclick="trainModel()">Train Model</button>
<div id="training-status"></div>
<div id="training-vis"></div>
```

### Code Flow

#### 5.1 Button Click → `trainModel()`
**Location:** `app.js` lines 459-530

**Training Pipeline:**
1. Split training data (80/20 train/validation)
2. Train model with callbacks
3. Visualize feature importance
4. Update metrics
5. Enable threshold slider

```javascript
async function trainModel() {
    if (!model || !preprocessedTrainData) {
        alert('Please create model first.');
        return;
    }

    const statusDiv = document.getElementById('training-status');
    statusDiv.innerHTML = 'Training model...';

    try {
        // Split training data (80% train, 20% validation)
        const numSamples = preprocessedTrainData.features.shape[0];
        const trainSize = Math.floor(numSamples * 0.8);

        const trainFeatures = preprocessedTrainData.features.slice([0, 0], [trainSize, -1]);
        const trainLabels = preprocessedTrainData.labels.slice([0], [trainSize]);

        const valFeatures = preprocessedTrainData.features.slice([trainSize, 0], [-1, -1]);
        const valLabels = preprocessedTrainData.labels.slice([trainSize], [-1]);

        // Train model
        trainingHistory = await model.fit(trainFeatures, trainLabels, {
            epochs: 50,
            batchSize: 32,
            validationData: [valFeatures, valLabels],
            callbacks: {
                onEpochEnd: async (epoch, logs) => {
                    // Update training status in the UI
                    statusDiv.innerHTML = `Epoch ${epoch + 1}/50 - loss: ${logs.loss.toFixed(4)}, acc: ${logs.acc.toFixed(4)}, val_loss: ${logs.val_loss.toFixed(4)}, val_acc: ${logs.val_acc.toFixed(4)}`;
                },
                // Add tfjs-vis callbacks for visualization
                ...tfvis.show.fitCallbacks(
                    { name: 'Training Performance', tab: 'Training' },
                    ['loss', 'acc', 'val_loss', 'val_acc'],
                    { callbacks: ['onEpochEnd'] }
                )
            }
        });

        statusDiv.innerHTML += '<p>Training completed!</p>';

        // Visualize feature importance
        visualizeFeatureImportance();

        // Get validation predictions for metrics
        valPredictions = model.predict(valFeatures);

        // Update evaluation metrics
        updateMetrics(valLabels.arraySync(), valPredictions.arraySync());

        // Enable threshold slider
        document.getElementById('threshold-slider').disabled = false;
        document.getElementById('predict-btn').disabled = false;

        // Cleanup tensors
        trainFeatures.dispose();
        trainLabels.dispose();
        valFeatures.dispose();
        valLabels.dispose();
    } catch (error) {
        statusDiv.innerHTML = `Error during training: ${error.message}`;
        console.error(error);
    }
}
```

### Training Configuration

#### Hyperparameters
```javascript
{
    epochs: 50,          // Full passes through training data
    batchSize: 32,       // Samples per gradient update
    validationData: [...] // Hold-out set for monitoring overfitting
}
```

**Why These Values?**
- **Epochs (50):** Enough for convergence on small dataset
- **Batch Size (32):** Balance between:
  - Large batches → stable gradients, faster computation
  - Small batches → noisy gradients, better generalization
- **Validation Split (20%):** Standard practice for model evaluation

#### Optimizer & Loss
```javascript
model.compile({
    optimizer: 'adam',           // Adaptive learning rate
    loss: 'binaryCrossentropy',  // For binary classification
    metrics: ['accuracy']        // Track classification accuracy
})
```

**Why Adam Optimizer?**
- Adapts learning rate per parameter
- Combines momentum (past gradients) + RMSprop (adaptive rates)
- Works well with minimal tuning

**Why Binary Cross-Entropy Loss?**
- Standard for binary classification
- Formula: `L = -[y·log(ŷ) + (1-y)·log(1-ŷ)]`
- Penalizes confident wrong predictions heavily

### Callbacks

#### UI Status Callback
```javascript
onEpochEnd: async (epoch, logs) => {
    statusDiv.innerHTML = `Epoch ${epoch + 1}/50 - loss: ${logs.loss.toFixed(4)}, acc: ${logs.acc.toFixed(4)}, val_loss: ${logs.val_loss.toFixed(4)}, val_acc: ${logs.val_acc.toFixed(4)}`;
}
```

**Updates every epoch with:**
- `loss`: Training loss (lower is better)
- `acc`: Training accuracy (0-1 scale)
- `val_loss`: Validation loss (watch for overfitting)
- `val_acc`: Validation accuracy

#### tfjs-vis Visualization Callback
```javascript
...tfvis.show.fitCallbacks(
    { name: 'Training Performance', tab: 'Training' },
    ['loss', 'acc', 'val_loss', 'val_acc'],
    { callbacks: ['onEpochEnd'] }
)
```

**Creates real-time line charts:**
- Loss curve (training vs validation)
- Accuracy curve (training vs validation)
- Updates every epoch
- Displayed in visor panel (right side)

### What Happens During Training

**Forward Pass:**
1. Take batch of 32 samples
2. Pass through network: Input → Gate → Gated → Hidden → Output
3. Get predictions (probabilities)

**Loss Calculation:**
4. Compare predictions to true labels
5. Calculate binary cross-entropy loss
6. Average over batch

**Backward Pass:**
7. Calculate gradients (∂Loss/∂Weight for each parameter)
8. Update weights using Adam optimizer
9. Repeat for next batch

**Epoch End:**
10. Calculate validation loss/accuracy
11. Trigger callbacks (update UI, charts)
12. Repeat for next epoch

### Expected Training Behavior

**Good Training:**
```
Epoch 1/50  - loss: 0.6234, acc: 0.6543, val_loss: 0.6198, val_acc: 0.6592
Epoch 10/50 - loss: 0.5123, acc: 0.7456, val_loss: 0.5234, val_acc: 0.7321
Epoch 20/50 - loss: 0.4567, acc: 0.7823, val_loss: 0.4789, val_acc: 0.7698
Epoch 50/50 - loss: 0.4123, acc: 0.8034, val_loss: 0.4456, val_acc: 0.7912
```
- Loss decreases steadily
- Accuracy increases toward 80%
- Validation metrics track training closely (< 5% gap)

**Overfitting (Bad):**
```
Epoch 50/50 - loss: 0.3000, acc: 0.9000, val_loss: 0.6000, val_acc: 0.7000
```
- Large gap between train and val metrics
- Model memorized training data, doesn't generalize

---

## 6. Evaluation Metrics

### Purpose
Assess model performance on validation data with confusion matrix, ROC curve, and adjustable threshold.

### UI Elements
```html
<input type="range" id="threshold-slider" min="0" max="1" step="0.01" value="0.5">
<span id="threshold-value">0.5</span>

<div id="confusion-matrix"></div>
<div id="performance-metrics"></div>
```

### Code Flow

#### 6.1 Threshold Slider → `updateMetrics()`
**Location:** `app.js` lines 532-589

**Triggered by:**
- Moving threshold slider
- Training completion (called with default 0.5)

**What It Does:**
1. Get current threshold value
2. Convert predictions to binary (0/1) using threshold
3. Calculate confusion matrix
4. Calculate performance metrics
5. Plot ROC curve and AUC

```javascript
async function updateMetrics(trueLabels, predictions) {
    // Get threshold from slider
    const threshold = parseFloat(document.getElementById('threshold-slider').value);
    document.getElementById('threshold-value').textContent = threshold.toFixed(2);

    // Flatten predictions if 2D
    const predVals = predictions.map(p => Array.isArray(p) ? p[0] : p);
    const trueVals = trueLabels.map(t => Array.isArray(t) ? t[0] : t);

    // Calculate confusion matrix
    let tp = 0, fn = 0, fp = 0, tn = 0;

    for (let i = 0; i < predVals.length; i++) {
        const prediction = predVals[i] >= threshold ? 1 : 0;
        const actual = trueVals[i];

        if (actual === 1) {
            if (prediction === 1) tp++;
            else fn++;
        } else {
            if (prediction === 1) fp++;
            else tn++;
        }
    }

    // Display confusion matrix
    const confusionDiv = document.getElementById('confusion-matrix');
    confusionDiv.innerHTML = `
        <table>
            <tr><th></th><th>Predicted Positive</th><th>Predicted Negative</th></tr>
            <tr><th>Actual Positive</th><td>${tp}</td><td>${fn}</td></tr>
            <tr><th>Actual Negative</th><td>${fp}</td><td>${tn}</td></tr>
        </table>
    `;

    // Calculate metrics
    const accuracy = (tp + tn) / (tp + tn + fp + fn);
    const precision = tp / (tp + fp) || 0;
    const recall = tp / (tp + fn) || 0;
    const f1 = 2 * (precision * recall) / (precision + recall) || 0;

    // Display metrics
    const metricsDiv = document.getElementById('performance-metrics');
    metricsDiv.innerHTML = `
        <p>Accuracy: ${(accuracy * 100).toFixed(2)}%</p>
        <p>Precision: ${precision.toFixed(4)}</p>
        <p>Recall: ${recall.toFixed(4)}</p>
        <p>F1 Score: ${f1.toFixed(4)}</p>
    `;

    // Calculate and plot ROC curve
    await plotROC(trueVals, predVals);
}
```

### Confusion Matrix Explained

```
                    Predicted
                 Positive  Negative
Actual Positive     TP        FN      ← Sensitivity = TP / (TP + FN)
Actual Negative     FP        TN      ← Specificity = TN / (FP + TN)
                    ↑         ↑
                Precision   NPV
```

**Example:**
```
Threshold = 0.5
179 validation samples

Predicted Survived=1  Predicted Survived=0
     35                    29             ← 64 actual survivors
     8                     107            ← 115 actual non-survivors
```

**Interpretation:**
- **TP (35):** Correctly predicted survivors
- **FN (29):** Missed survivors (Type II error)
- **FP (8):** False alarms (Type I error)
- **TN (107):** Correctly predicted deaths

### Performance Metrics

#### Accuracy
```javascript
accuracy = (tp + tn) / total
         = (35 + 107) / 179
         = 79.33%
```
**Meaning:** Overall correctness rate.

#### Precision
```javascript
precision = tp / (tp + fp)
          = 35 / (35 + 8)
          = 0.8140
```
**Meaning:** Of predicted survivors, 81.4% actually survived.
**Use Case:** When false positives are costly.

#### Recall (Sensitivity)
```javascript
recall = tp / (tp + fn)
       = 35 / (35 + 29)
       = 0.5469
```
**Meaning:** Of actual survivors, we detected 54.7%.
**Use Case:** When false negatives are costly (e.g., medical diagnosis).

#### F1 Score
```javascript
f1 = 2 × (precision × recall) / (precision + recall)
   = 2 × (0.814 × 0.547) / (0.814 + 0.547)
   = 0.6535
```
**Meaning:** Harmonic mean of precision and recall.
**Use Case:** Balance between precision and recall.

### ROC Curve & AUC

#### 6.2 ROC Plotting: `plotROC(trueLabels, predictions)`
**Location:** `app.js` lines 591-644

**Algorithm:**
1. Generate 100 threshold values (0.00 to 0.99)
2. For each threshold, calculate TPR and FPR
3. Plot TPR vs FPR
4. Calculate AUC using trapezoidal rule

```javascript
async function plotROC(trueLabels, predictions) {
    // Calculate TPR and FPR for different thresholds
    const thresholds = Array.from({ length: 100 }, (_, i) => i / 100);
    const rocData = [];

    thresholds.forEach(threshold => {
        let tp = 0, fn = 0, fp = 0, tn = 0;

        for (let i = 0; i < predictions.length; i++) {
            const prediction = predictions[i] >= threshold ? 1 : 0;
            const actual = trueLabels[i];

            if (actual === 1) {
                if (prediction === 1) tp++;
                else fn++;
            } else {
                if (prediction === 1) fp++;
                else tn++;
            }
        }

        const tpr = tp / (tp + fn) || 0;  // True Positive Rate (Recall)
        const fpr = fp / (fp + tn) || 0;  // False Positive Rate

        rocData.push({ threshold, fpr, tpr });
    });

    // Sort by FPR for correct AUC calculation
    rocData.sort((a, b) => a.fpr - b.fpr);

    // Calculate AUC (area under curve) using trapezoidal rule
    let auc = 0;
    for (let i = 1; i < rocData.length; i++) {
        const width = rocData[i].fpr - rocData[i-1].fpr;
        const avgHeight = (rocData[i].tpr + rocData[i-1].tpr) / 2;
        auc += width * avgHeight;
    }

    // Plot ROC curve
    tfvis.render.linechart(
        { name: 'ROC Curve', tab: 'Evaluation' },
        { values: rocData.map(d => ({ x: d.fpr, y: d.tpr })) },
        {
            xLabel: 'False Positive Rate',
            yLabel: 'True Positive Rate',
            series: ['ROC Curve'],
            width: 400,
            height: 400
        }
    );

    // Add AUC to performance metrics
    const metricsDiv = document.getElementById('performance-metrics');
    metricsDiv.innerHTML += `<p>AUC: ${auc.toFixed(4)}</p>`;
}
```

### ROC Curve Explained

**Concept:**
- X-axis: False Positive Rate (FPR) = FP / (FP + TN)
- Y-axis: True Positive Rate (TPR) = TP / (TP + FN)
- Each point represents a different threshold

**Ideal ROC:**
```
TPR
 1 ┤  *******
   │  *
   │  *
   │ *
 0 └─────────┬ FPR
   0         1
```
- Hugs top-left corner (high TPR, low FPR)
- AUC ≈ 1.0 (perfect classifier)

**Random Classifier:**
```
TPR
 1 ┤       *
   │     *
   │   *
   │ *
 0 └───────┬ FPR
   0       1
```
- Diagonal line (TPR = FPR)
- AUC = 0.5 (no better than coin flip)

**AUC Interpretation:**
- **AUC = 0.90-1.00:** Excellent
- **AUC = 0.80-0.90:** Good (typical for this dataset)
- **AUC = 0.70-0.80:** Fair
- **AUC = 0.50-0.70:** Poor
- **AUC < 0.50:** Worse than random (model is inverted!)

### Threshold Adjustment

**Why Adjustable Threshold?**

Default threshold = 0.5, but optimal threshold depends on use case:

**Example 1: Medical Screening (prioritize recall)**
- Set threshold = 0.3
- More false positives, fewer false negatives
- Better to have extra tests than miss disease

**Example 2: Spam Filter (prioritize precision)**
- Set threshold = 0.7
- Fewer false positives, more false negatives
- Better to let spam through than block real emails

**Interactive Slider:**
```javascript
document.getElementById('threshold-slider').addEventListener('input', () => {
    updateMetrics(trueLabels, predictions);
});
```
- Drag slider → instantly updates confusion matrix and metrics
- See real-time trade-off between precision and recall

---

## 7. Prediction

### Purpose
Apply trained model to test data and display results.

### UI Elements
```html
<button id="predict-btn" onclick="predict()">Predict on Test Data</button>
<div id="prediction-output"></div>
```

### Code Flow

#### 7.1 Button Click → `predict()`
**Location:** `app.js` lines 646-684

**Pipeline:**
1. Convert test features to tensor
2. Run model.predict()
3. Extract probability values
4. Create results table
5. Enable export button

```javascript
async function predict() {
    if (!model || !preprocessedTestData) {
        alert('Please train model first.');
        return;
    }

    const outputDiv = document.getElementById('prediction-output');
    outputDiv.innerHTML = 'Making predictions...';

    try {
        // Convert test features to tensor
        const testFeatures = tf.tensor2d(preprocessedTestData.features);

        // Make predictions
        testPredictions = model.predict(testFeatures);
        const predValues = testPredictions.arraySync();

        // Create prediction results
        // Note: predValues is 2D array [[val1], [val2], ...], so access predValues[i][0]
        const results = preprocessedTestData.passengerIds.map((id, i) => {
            // Handle both 1D and 2D array formats
            const prob = Array.isArray(predValues[i]) ? predValues[i][0] : predValues[i];
            return {
                PassengerId: id,
                Survived: prob >= 0.5 ? 1 : 0,
                Probability: Number(prob) // Ensure it's a number
            };
        });

        // Show first 10 predictions
        outputDiv.innerHTML = '<h3>Prediction Results (First 10 Rows)</h3>';
        outputDiv.appendChild(createPredictionTable(results.slice(0, 10)));

        outputDiv.innerHTML += `<p>Predictions completed! Total: ${results.length} samples</p>`;

        // Enable the export button
        document.getElementById('export-btn').disabled = false;
    } catch (error) {
        outputDiv.innerHTML = `Error during prediction: ${error.message}`;
        console.error(error);
    }
}
```

#### 7.2 Helper: `createPredictionTable(data)`
**Location:** `app.js` lines 686-711

**Purpose:** Display predictions in HTML table format.

```javascript
function createPredictionTable(data) {
    const table = document.createElement('table');

    // Create header row
    const headerRow = document.createElement('tr');
    ['PassengerId', 'Survived', 'Probability'].forEach(header => {
        const th = document.createElement('th');
        th.textContent = header;
        headerRow.appendChild(th);
    });
    table.appendChild(headerRow);

    // Create data rows
    data.forEach(row => {
        const tr = document.createElement('tr');
        ['PassengerId', 'Survived', 'Probability'].forEach(key => {
            const td = document.createElement('td');
            const value = row[key];
            // Format probability with defensive check
            if (key === 'Probability') {
                td.textContent = typeof value === 'number' ? value.toFixed(4) : String(value);
            } else {
                td.textContent = value;
            }
            tr.appendChild(td);
        });
        table.appendChild(tr);
    });

    return table;
}
```

### Prediction Output Format

**Example Table:**
```
┌─────────────┬──────────┬─────────────┐
│ PassengerId │ Survived │ Probability │
├─────────────┼──────────┼─────────────┤
│ 892         │ 0        │ 0.1234      │
│ 893         │ 1        │ 0.8765      │
│ 894         │ 0        │ 0.2456      │
│ 895         │ 0        │ 0.3789      │
│ 896         │ 1        │ 0.9123      │
│ 897         │ 0        │ 0.0987      │
│ 898         │ 1        │ 0.7654      │
│ 899         │ 0        │ 0.4321      │
│ 900         │ 1        │ 0.8234      │
│ 901         │ 0        │ 0.2789      │
└─────────────┴──────────┴─────────────┘

Predictions completed! Total: 418 samples
```

### How Prediction Works

**Step-by-Step:**

1. **Input:** Preprocessed test features (418 rows × 14 features)
   ```javascript
   const testFeatures = tf.tensor2d([
       [-0.565, -0.502, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 2, 0],  // Passenger 892
       [0.345, 0.123, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1],     // Passenger 893
       // ... 416 more rows
   ]);
   ```

2. **Forward Pass Through Network:**
   ```
   testFeatures [418×14]
        ↓
   Sigmoid Gate [418×14]  (each row gets gated independently)
        ↓
   Gated Input [418×14]
        ↓
   Hidden Layer [418×16]
        ↓
   Output Layer [418×1]  (probabilities)
   ```

3. **Output:** Probabilities [418×1]
   ```javascript
   predValues = [[0.1234], [0.8765], [0.2456], ...]  // 2D array!
   ```

4. **Threshold:** Convert to binary
   ```javascript
   prob >= 0.5 ? 1 : 0

   0.1234 → 0 (did not survive)
   0.8765 → 1 (survived)
   0.2456 → 0 (did not survive)
   ```

### Why 2D Array?

TensorFlow.js always returns predictions as 2D array:
- Shape: `[numSamples, outputUnits]`
- Even if outputUnits = 1, you get `[[val], [val], ...]`
- Must extract with `predValues[i][0]`

**Defensive Code:**
```javascript
const prob = Array.isArray(predValues[i]) ? predValues[i][0] : predValues[i];
```
This handles both formats (future-proofing).

---

## 8. Export Results

### Purpose
Download prediction results and trained model weights for submission or further analysis.

### UI Elements
```html
<button id="export-btn" onclick="exportResults()">Export Results</button>
<div id="export-status"></div>
```

### Code Flow

#### 8.1 Button Click → `exportResults()`
**Location:** `app.js` lines 801-848

**Exports:**
1. `submission.csv` - Binary predictions (Kaggle format)
2. `probabilities.csv` - Raw probabilities
3. `titanic-tfjs-model.json` - Model weights

```javascript
async function exportResults() {
    if (!testPredictions || !preprocessedTestData) {
        alert('Please make predictions first.');
        return;
    }

    const statusDiv = document.getElementById('export-status');
    statusDiv.innerHTML = 'Exporting results...';

    try {
        // Get predictions
        const predValues = testPredictions.arraySync();

        // Create submission CSV (PassengerId, Survived)
        let submissionCSV = 'PassengerId,Survived\n';
        preprocessedTestData.passengerIds.forEach((id, i) => {
            const prob = Array.isArray(predValues[i]) ? predValues[i][0] : predValues[i];
            submissionCSV += `${id},${prob >= 0.5 ? 1 : 0}\n`;
        });

        // Create probabilities CSV (PassengerId, Probability)
        let probabilitiesCSV = 'PassengerId,Probability\n';
        preprocessedTestData.passengerIds.forEach((id, i) => {
            const prob = Array.isArray(predValues[i]) ? predValues[i][0] : predValues[i];
            probabilitiesCSV += `${id},${Number(prob).toFixed(6)}\n`;
        });

        // Create download links
        const submissionLink = document.createElement('a');
        submissionLink.href = URL.createObjectURL(new Blob([submissionCSV], { type: 'text/csv' }));
        submissionLink.download = 'submission.csv';

        const probabilitiesLink = document.createElement('a');
        probabilitiesLink.href = URL.createObjectURL(new Blob([probabilitiesCSV], { type: 'text/csv' }));
        probabilitiesLink.download = 'probabilities.csv';

        // Trigger downloads
        submissionLink.click();
        probabilitiesLink.click();

        // Save model
        await model.save('downloads://titanic-tfjs-model');

        statusDiv.innerHTML = `
            <p>Results exported successfully!</p>
            <ul>
                <li>submission.csv - Binary predictions for Kaggle</li>
                <li>probabilities.csv - Raw probability values</li>
                <li>titanic-tfjs-model.json - Model weights</li>
            </ul>
        `;
    } catch (error) {
        statusDiv.innerHTML = `Error exporting results: ${error.message}`;
        console.error(error);
    }
}
```

### Export File Formats

#### 8.1.1 submission.csv (Kaggle Format)
```csv
PassengerId,Survived
892,0
893,1
894,0
895,0
896,1
897,0
898,1
...
```

**Purpose:** Direct submission to Kaggle competition.
**Format:** Binary predictions (0 or 1).
**Threshold:** Uses 0.5 by default.

#### 8.1.2 probabilities.csv (Analysis Format)
```csv
PassengerId,Probability
892,0.123456
893,0.876543
894,0.245678
895,0.378901
896,0.912345
897,0.098765
898,0.765432
...
```

**Purpose:** Further analysis, custom thresholding, ensembling.
**Format:** Continuous probabilities (6 decimal places).
**Use Case:** Adjust threshold after submission, combine with other models.

#### 8.1.3 Model Files (TensorFlow.js Format)
**Downloads:**
- `titanic-tfjs-model.json` - Model architecture and metadata
- `titanic-tfjs-model.weights.bin` - Trained weights (binary)

**Contents of .json:**
```json
{
    "modelTopology": {
        "class_name": "Functional",
        "config": {
            "name": "model",
            "layers": [...],
            "input_layers": [...],
            "output_layers": [...]
        }
    },
    "weightsManifest": [
        {
            "paths": ["titanic-tfjs-model.weights.bin"],
            "weights": [
                {"name": "sigmoid_gate/kernel", "shape": [14, 14], "dtype": "float32"},
                {"name": "dense/kernel", "shape": [14, 16], "dtype": "float32"},
                ...
            ]
        }
    ]
}
```

**Purpose:**
- Load model later without retraining
- Deploy to production
- Share with others

**How to Load Later:**
```javascript
const model = await tf.loadLayersModel('file://path/to/titanic-tfjs-model.json');
```

### Download Implementation

#### Blob Creation
```javascript
const blob = new Blob([csvText], { type: 'text/csv' });
```
- Creates in-memory file from string
- MIME type: `text/csv` for CSV files

#### Object URL
```javascript
const url = URL.createObjectURL(blob);
```
- Creates temporary URL pointing to blob
- Format: `blob:http://localhost:3000/abc123-def456`
- Valid only during current session

#### Programmatic Download
```javascript
const link = document.createElement('a');
link.href = url;
link.download = 'submission.csv';
link.click();
```
- Creates invisible link element
- Sets download attribute (forces download, not open)
- Programmatically clicks link
- Browser triggers download dialog

#### Model Save
```javascript
await model.save('downloads://titanic-tfjs-model');
```
- `downloads://` is special TensorFlow.js handler
- Triggers browser download for both .json and .weights.bin
- Alternative: `indexeddb://` for browser storage, `localstorage://` for localStorage

---

## Summary: Complete Workflow

### Data Flow Diagram
```
User Files (train.csv, test.csv)
    ↓ [readFile()]
Raw CSV Text
    ↓ [parseCSV()]
JavaScript Objects [{}, {}, ...]
    ↓ [extractFeatures()]
Numerical Arrays [[...], [...], ...]
    ↓ [tf.tensor2d()]
TensorFlow Tensors
    ↓ [model.fit()]
Trained Model Weights
    ↓ [model.predict()]
Predictions [[prob1], [prob2], ...]
    ↓ [exportResults()]
CSV Files + Model Files
```

### Button Dependency Chain
```
Load Data
    ↓ (enables)
Inspect Data
    ↓ (enables)
Preprocess Data
    ↓ (enables)
Create Model
    ↓ (enables)
Train Model
    ↓ (enables)
Predict on Test Data
    ↓ (enables)
Export Results
```

### Global Variables
```javascript
let trainData = [];              // Parsed training data
let testData = [];               // Parsed test data
let preprocessedTrainData = {}; // Features + labels tensors
let preprocessedTestData = {};  // Features arrays + IDs
let model = null;               // TensorFlow.js model
let trainingHistory = null;     // Training metrics per epoch
let valPredictions = null;      // Validation predictions
let testPredictions = null;     // Test predictions
let featureNames = [];          // Feature name labels
let gateWeights = [];           // Sigmoid gate importance scores
```

### Key Technologies Used

**Frontend:**
- HTML5 File API (file uploads)
- JavaScript ES6+ (async/await, arrow functions)
- DOM manipulation (createElement, innerHTML)

**Machine Learning:**
- TensorFlow.js (neural networks in browser)
- tfjs-vis (visualization library)

**Data Processing:**
- RFC 4180 CSV parsing
- Z-score normalization
- One-hot encoding
- Median/mode imputation

**Neural Network:**
- Functional API (complex architectures)
- Sigmoid Gate (interpretable ML)
- Adam optimizer
- Binary cross-entropy loss

---

## Performance Considerations

### Memory Management
```javascript
// Always dispose tensors when done
trainFeatures.dispose();
trainLabels.dispose();

// Or use tf.tidy() for automatic cleanup
const result = tf.tidy(() => {
    const x = tf.tensor([1, 2, 3]);
    const y = x.square();
    return y.dataSync();
});
```

### Computation Time
- **Data Loading:** ~100ms (depends on file size)
- **Preprocessing:** ~200ms (891 samples)
- **Model Creation:** ~50ms
- **Training:** ~15-20 seconds (50 epochs, CPU)
- **Prediction:** ~100ms (418 samples)
- **Export:** ~500ms (file I/O)

### Browser Compatibility
- **Chrome:** ✓ Full support
- **Firefox:** ✓ Full support
- **Safari:** ✓ Full support (may be slower)
- **Edge:** ✓ Full support
- **Mobile:** ⚠️ Limited (slow, memory constraints)

### Optimization Tips
1. **Use smaller batch sizes** on low-end devices (16 instead of 32)
2. **Reduce epochs** for faster training (30 instead of 50)
3. **Disable visualizations** if slow (comment out tfvis calls)
4. **Use WebGL backend** (TensorFlow.js auto-detects)

---

## Troubleshooting Common Issues

### Issue 1: "Cannot read property 'files' of null"
**Cause:** Button clicked before file selected
**Fix:** Check `if (!trainFile || !testFile)` before processing

### Issue 2: "CSV parsing error"
**Cause:** File encoding or line endings
**Fix:** Ensure UTF-8 encoding, Unix line endings (LF)

### Issue 3: "Model training stuck"
**Cause:** Browser tab in background (throttled)
**Fix:** Keep tab active during training

### Issue 4: "Out of memory"
**Cause:** Too many tensors not disposed
**Fix:** Use `.dispose()` or `tf.tidy()`

### Issue 5: "Predictions all 0 or all 1"
**Cause:** Model not trained properly
**Fix:** Check training loss decreases, increase epochs

---

## Next Steps & Improvements

### Possible Enhancements
1. **Early stopping** - Stop training if val_loss increases
2. **Learning rate scheduler** - Reduce LR when plateaued
3. **Dropout layers** - Regularization to prevent overfitting
4. **Hyperparameter tuning** - Grid search for optimal config
5. **Ensemble models** - Combine multiple models
6. **SHAP values** - Local explanations per prediction
7. **Data augmentation** - Synthetic samples for imbalanced classes

### Educational Value
This application demonstrates:
- ✓ End-to-end ML pipeline in browser
- ✓ Interpretable deep learning (sigmoid gate)
- ✓ Real-time training visualization
- ✓ Interactive model evaluation
- ✓ No server/API required (privacy-preserving)
- ✓ Suitable for education and demos

---

**End of Frontend Guide**
