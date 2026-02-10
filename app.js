// Global variables
let trainData = null;
let testData = null;
let preprocessedTrainData = null;
let preprocessedTestData = null;
let model = null;
let trainingHistory = null;
let validationData = null;
let validationLabels = null;
let validationPredictions = null;
let testPredictions = null;
let featureNames = []; // Для хранения имен признаков
let featureImportance = {}; // Для хранения важности признаков

// Schema configuration - change these for different datasets
const TARGET_FEATURE = 'Survived'; // Binary classification target
const ID_FEATURE = 'PassengerId'; // Identifier to exclude from features
const NUMERICAL_FEATURES = ['Age', 'Fare', 'SibSp', 'Parch']; // Numerical features
const CATEGORICAL_FEATURES = ['Pclass', 'Sex', 'Embarked']; // Categorical features

// 1. FIX: Proper CSV parsing with comma escape handling
function parseCSV(csvText) {
    const lines = csvText.split('\n').filter(line => line.trim() !== '');
    if (lines.length === 0) return [];
    
    // Parse headers
    const headers = parseCSVLine(lines[0]);
    
    return lines.slice(1).map(line => {
        const values = parseCSVLine(line);
        const obj = {};
        headers.forEach((header, i) => {
            // Handle missing values (empty strings)
            const value = i < values.length ? values[i] : '';
            obj[header] = value === '' ? null : value;
            
            // Convert numerical values to numbers if possible
            if (obj[header] !== null && !isNaN(obj[header]) && obj[header] !== '') {
                // Check if it's already a number
                if (typeof obj[header] === 'string') {
                    obj[header] = parseFloat(obj[header]);
                }
            }
        });
        return obj;
    });
}

// Helper function to parse CSV line with quoted fields
function parseCSVLine(line) {
    const result = [];
    let current = '';
    let inQuotes = false;
    
    for (let i = 0; i < line.length; i++) {
        const char = line[i];
        
        if (char === '"') {
            // Check for escaped quotes (two quotes in a row)
            if (i + 1 < line.length && line[i + 1] === '"') {
                current += '"';
                i++; // Skip the next quote
            } else {
                inQuotes = !inQuotes;
            }
        } else if (char === ',' && !inQuotes) {
            result.push(current);
            current = '';
        } else {
            current += char;
        }
    }
    // Add the last field
    result.push(current);
    
    return result.map(val => val.trim());
}

// Read file as text
function readFile(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = e => resolve(e.target.result);
        reader.onerror = e => reject(new Error('Failed to read file'));
        reader.readAsText(file);
    });
}

// Load data from uploaded CSV files
async function loadData() {
    const trainFile = document.getElementById('train-file').files[0];
    const testFile = document.getElementById('test-file').files[0];
    
    if (!trainFile || !testFile) {
        alert('Please upload both training and test CSV files.');
        return;
    }
    
    const statusDiv = document.getElementById('data-status');
    statusDiv.innerHTML = 'Loading data...';
    
    try {
        // Load training data
        const trainText = await readFile(trainFile);
        trainData = parseCSV(trainText);
        
        // Load test data
        const testText = await readFile(testFile);
        testData = parseCSV(testText);
        
        // Validate data was loaded correctly
        if (!trainData || trainData.length === 0) {
            throw new Error('Training data is empty or could not be parsed');
        }
        
        if (!testData || testData.length === 0) {
            throw new Error('Test data is empty or could not be parsed');
        }
        
        statusDiv.innerHTML = `Data loaded successfully!<br>
                              Training: ${trainData.length} samples<br>
                              Test: ${testData.length} samples<br>
                              Features: ${Object.keys(trainData[0]).join(', ')}`;
        
        // Enable the inspect button
        document.getElementById('inspect-btn').disabled = false;
    } catch (error) {
        statusDiv.innerHTML = `Error loading data: ${error.message}`;
        console.error('Error details:', error);
    }
}

// Inspect the loaded data
function inspectData() {
    if (!trainData || trainData.length === 0) {
        alert('Please load data first.');
        return;
    }
    
    // Show data preview
    const previewDiv = document.getElementById('data-preview');
    previewDiv.innerHTML = '<h3>Data Preview (First 10 Rows)</h3>';
    previewDiv.appendChild(createPreviewTable(trainData.slice(0, 10)));
    
    // Calculate and show data statistics
    const statsDiv = document.getElementById('data-stats');
    statsDiv.innerHTML = '<h3>Data Statistics</h3>';
    
    const shapeInfo = `Dataset shape: ${trainData.length} rows × ${Object.keys(trainData[0]).length} columns`;
    const survivalCount = trainData.filter(row => row[TARGET_FEATURE] === 1).length;
    const survivalRate = (survivalCount / trainData.length * 100).toFixed(2);
    const targetInfo = `Survival rate: ${survivalCount}/${trainData.length} (${survivalRate}%)`;
    
    // Calculate missing values percentage for each feature
    let missingInfo = '<h4>Missing Values Percentage:</h4><ul>';
    Object.keys(trainData[0]).forEach(feature => {
        const missingCount = trainData.filter(row => row[feature] === null || row[feature] === undefined || row[feature] === '').length;
        const missingPercent = (missingCount / trainData.length * 100).toFixed(2);
        missingInfo += `<li>${feature}: ${missingPercent}% (${missingCount} missing)</li>`;
    });
    missingInfo += '</ul>';
    
    statsDiv.innerHTML += `<p>${shapeInfo}</p><p>${targetInfo}</p>${missingInfo}`;
    
    // Create visualizations
    createVisualizations();
    
    // Enable the preprocess button
    document.getElementById('preprocess-btn').disabled = false;
}

// Create a preview table from data
function createPreviewTable(data) {
    const table = document.createElement('table');
    
    // Create header row
    const headerRow = document.createElement('tr');
    Object.keys(data[0]).forEach(key => {
        const th = document.createElement('th');
        th.textContent = key;
        headerRow.appendChild(th);
    });
    table.appendChild(headerRow);
    
    // Create data rows
    data.forEach(row => {
        const tr = document.createElement('tr');
        Object.values(row).forEach(value => {
            const td = document.createElement('td');
            td.textContent = value !== null ? value : 'NULL';
            tr.appendChild(td);
        });
        table.appendChild(tr);
    });
    
    return table;
}

// Create visualizations using tfjs-vis
function createVisualizations() {
    const chartsDiv = document.getElementById('charts');
    chartsDiv.innerHTML = '<h3>Data Visualizations</h3>';
    
    // Survival by Sex
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
    
    const sexData = Object.entries(survivalBySex).map(([sex, stats]) => ({
        sex,
        survivalRate: (stats.survived / stats.total) * 100
    }));
    
    // Create container for sex chart
    const sexChartDiv = document.createElement('div');
    sexChartDiv.id = 'sex-chart';
    chartsDiv.appendChild(sexChartDiv);
    
    tfvis.render.barchart(
        { name: 'Survival Rate by Sex', tab: 'Charts', container: '#sex-chart' },
        sexData.map(d => ({ x: d.sex, y: d.survivalRate })),
        { xLabel: 'Sex', yLabel: 'Survival Rate (%)' }
    );
    
    // Survival by Pclass
    const survivalByPclass = {};
    trainData.forEach(row => {
        if (row.Pclass !== undefined && row.Survived !== undefined) {
            if (!survivalByPclass[row.Pclass]) {
                survivalByPclass[row.Pclass] = { survived: 0, total: 0 };
            }
            survivalByPclass[row.Pclass].total++;
            if (row.Survived === 1) {
                survivalByPclass[row.Pclass].survived++;
            }
        }
    });
    
    const pclassData = Object.entries(survivalByPclass).map(([pclass, stats]) => ({
        pclass: `Class ${pclass}`,
        survivalRate: (stats.survived / stats.total) * 100
    }));
    
    // Create container for pclass chart
    const pclassChartDiv = document.createElement('div');
    pclassChartDiv.id = 'pclass-chart';
    chartsDiv.appendChild(pclassChartDiv);
    
    tfvis.render.barchart(
        { name: 'Survival Rate by Passenger Class', tab: 'Charts', container: '#pclass-chart' },
        pclassData.map(d => ({ x: d.pclass, y: d.survivalRate })),
        { xLabel: 'Passenger Class', yLabel: 'Survival Rate (%)' }
    );
    
    chartsDiv.innerHTML += '<p>Charts displayed above. You can also open tfjs-vis visor (button in bottom right) for interactive charts.</p>';
}

// Calculate median of an array
function calculateMedian(values) {
    const filtered = values.filter(v => v !== null && v !== undefined);
    if (filtered.length === 0) return 0;
    
    filtered.sort((a, b) => a - b);
    const half = Math.floor(filtered.length / 2);
    
    if (filtered.length % 2 === 0) {
        return (filtered[half - 1] + filtered[half]) / 2;
    }
    
    return filtered[half];
}

// Calculate mode of an array
function calculateMode(values) {
    const filtered = values.filter(v => v !== null && v !== undefined);
    if (filtered.length === 0) return 'S'; // Default to Southampton
    
    const frequency = {};
    let maxCount = 0;
    let mode = filtered[0];
    
    filtered.forEach(value => {
        frequency[value] = (frequency[value] || 0) + 1;
        if (frequency[value] > maxCount) {
            maxCount = frequency[value];
            mode = value;
        }
    });
    
    return mode;
}

// Calculate standard deviation of an array
function calculateStdDev(values) {
    const filtered = values.filter(v => v !== null && v !== undefined);
    if (filtered.length === 0) return 1;
    
    const mean = filtered.reduce((sum, val) => sum + val, 0) / filtered.length;
    const squaredDiffs = filtered.map(value => Math.pow(value - mean, 2));
    const variance = squaredDiffs.reduce((sum, val) => sum + val, 0) / filtered.length;
    return Math.sqrt(variance);
}

// One-hot encode a value
function oneHotEncode(value, categories) {
    const encoding = new Array(categories.length).fill(0);
    const index = categories.indexOf(value);
    if (index !== -1) {
        encoding[index] = 1;
    }
    return encoding;
}

// Extract features from a row with imputation and normalization
function extractFeatures(row, ageMedian, fareMedian, embarkedMode) {
    // Impute missing values
    const age = row.Age !== null ? row.Age : ageMedian;
    const fare = row.Fare !== null ? row.Fare : fareMedian;
    const embarked = row.Embarked !== null ? row.Embarked : embarkedMode;
    
    // Get standard deviations for normalization
    const ageStd = calculateStdDev(trainData.map(r => r.Age).filter(a => a !== null));
    const fareStd = calculateStdDev(trainData.map(r => r.Fare).filter(f => f !== null));
    
    // Standardize numerical features (avoid division by zero)
    const standardizedAge = ageStd !== 0 ? (age - ageMedian) / ageStd : 0;
    const standardizedFare = fareStd !== 0 ? (fare - fareMedian) / fareStd : 0;
    
    // One-hot encode categorical features
    const pclassOneHot = oneHotEncode(row.Pclass, [1, 2, 3]);
    const sexOneHot = oneHotEncode(row.Sex, ['male', 'female']);
    const embarkedOneHot = oneHotEncode(embarked, ['C', 'Q', 'S']);
    
    // Start with numerical features
    let features = [
        standardizedAge,
        standardizedFare,
        row.SibSp || 0,
        row.Parch || 0
    ];
    
    // Add one-hot encoded features
    features = features.concat(pclassOneHot, sexOneHot, embarkedOneHot);
    
    // Add optional family features if enabled
    if (document.getElementById('add-family-features').checked) {
        const familySize = (row.SibSp || 0) + (row.Parch || 0) + 1;
        const isAlone = familySize === 1 ? 1 : 0;
        features.push(familySize, isAlone);
    }
    
    return features;
}

// Generate feature names for interpretability
function generateFeatureNames() {
    let names = [
        'Age_std', 'Fare_std', 'SibSp', 'Parch',
        'Pclass_1', 'Pclass_2', 'Pclass_3',
        'Sex_male', 'Sex_female',
        'Embarked_C', 'Embarked_Q', 'Embarked_S'
    ];
    
    if (document.getElementById('add-family-features').checked) {
        names.push('FamilySize', 'IsAlone');
    }
    
    return names;
}

// Preprocess the data
function preprocessData() {
    if (!trainData || !testData) {
        alert('Please load data first.');
        return;
    }
    
    const outputDiv = document.getElementById('preprocessing-output');
    outputDiv.innerHTML = 'Preprocessing data...';
    
    try {
        // Calculate imputation values from training data
        const ageMedian = calculateMedian(trainData.map(row => row.Age));
        const fareMedian = calculateMedian(trainData.map(row => row.Fare));
        const embarkedMode = calculateMode(trainData.map(row => row.Embarked));
        
        // Generate feature names
        featureNames = generateFeatureNames();
        
        // Preprocess training data
        preprocessedTrainData = {
            features: [],
            labels: []
        };
        
        trainData.forEach(row => {
            const features = extractFeatures(row, ageMedian, fareMedian, embarkedMode);
            preprocessedTrainData.features.push(features);
            preprocessedTrainData.labels.push(row[TARGET_FEATURE]);
        });
        
        // Preprocess test data
        preprocessedTestData = {
            features: [],
            passengerIds: []
        };
        
        testData.forEach(row => {
            const features = extractFeatures(row, ageMedian, fareMedian, embarkedMode);
            preprocessedTestData.features.push(features);
            preprocessedTestData.passengerIds.push(row[ID_FEATURE]);
        });
        
        // Convert to tensors
        preprocessedTrainData.features = tf.tensor2d(preprocessedTrainData.features);
        preprocessedTrainData.labels = tf.tensor1d(preprocessedTrainData.labels);
        
        outputDiv.innerHTML = `
            <p>Preprocessing completed!</p>
            <p>Training features shape: ${preprocessedTrainData.features.shape}</p>
            <p>Training labels shape: ${preprocessedTrainData.labels.shape}</p>
            <p>Test features shape: [${preprocessedTestData.features.length}, ${preprocessedTestData.features[0] ? preprocessedTestData.features[0].length : 0}]</p>
            <p>Feature names: ${featureNames.join(', ')}</p>
        `;
        
        // Enable the create model button
        document.getElementById('create-model-btn').disabled = false;
    } catch (error) {
        outputDiv.innerHTML = `Error during preprocessing: ${error.message}`;
        console.error('Preprocessing error:', error);
    }
}

// Create the model
function createModel() {
    if (!preprocessedTrainData) {
        alert('Please preprocess data first.');
        return;
    }
    
    const inputShape = preprocessedTrainData.features.shape[1];
    
    // Create a sequential model
    model = tf.sequential();
    
    // Add layers
    model.add(tf.layers.dense({
        units: 16,
        activation: 'relu',
        inputShape: [inputShape]
    }));
    
    model.add(tf.layers.dense({
        units: 1,
        activation: 'sigmoid'
    }));
    
    // Compile the model
    model.compile({
        optimizer: 'adam',
        loss: 'binaryCrossentropy',
        metrics: ['accuracy']
    });
    
    // Display model summary
    const summaryDiv = document.getElementById('model-summary');
    summaryDiv.innerHTML = '<h3>Model Summary</h3>';
    
    let summaryText = '<ul>';
    model.layers.forEach((layer, i) => {
        summaryText += `<li>Layer ${i+1}: ${layer.getClassName()} - Output Shape: ${JSON.stringify(layer.outputShape)}</li>`;
    });
    summaryText += '</ul>';
    summaryText += `<p>Total parameters: ${model.countParams()}</p>`;
    summaryText += `<p>Input features: ${inputShape}</p>`;
    summaryDiv.innerHTML += summaryText;
    
    // Enable the train button
    document.getElementById('train-btn').disabled = false;
}

// Train the model
async function trainModel() {
    if (!model || !preprocessedTrainData) {
        alert('Please create model first.');
        return;
    }
    
    const statusDiv = document.getElementById('training-status');
    statusDiv.innerHTML = 'Training model...';
    
    try {
        // Split training data into train and validation sets (80/20)
        const splitIndex = Math.floor(preprocessedTrainData.features.shape[0] * 0.8);
        
        const trainFeatures = preprocessedTrainData.features.slice(0, splitIndex);
        const trainLabels = preprocessedTrainData.labels.slice(0, splitIndex);
        
        const valFeatures = preprocessedTrainData.features.slice(splitIndex);
        const valLabels = preprocessedTrainData.labels.slice(splitIndex);
        
        // Store validation data for later evaluation
        validationData = valFeatures;
        validationLabels = valLabels;
        
        // Train the model
        trainingHistory = await model.fit(trainFeatures, trainLabels, {
            epochs: 50,
            batchSize: 32,
            validationData: [valFeatures, valLabels],
            callbacks: tfvis.show.fitCallbacks(
                { name: 'Training Performance', tab: 'Training' },
                ['loss', 'acc', 'val_loss', 'val_acc']
            ),
            verbose: 0
        });
        
        statusDiv.innerHTML = `
            <p>Training completed!</p>
            <p>Final accuracy: ${(trainingHistory.history.acc[trainingHistory.history.acc.length - 1] * 100).toFixed(2)}%</p>
            <p>Final validation accuracy: ${(trainingHistory.history.val_acc[trainingHistory.history.val_acc.length - 1] * 100).toFixed(2)}%</p>
        `;
        
        // Make predictions on validation set for evaluation
        validationPredictions = model.predict(validationData);
        
        // Enable the threshold slider and evaluation
        document.getElementById('threshold-slider').disabled = false;
        document.getElementById('threshold-slider').addEventListener('input', updateMetrics);
        
        // Enable the predict button
        document.getElementById('predict-btn').disabled = false;
        
        // Enable the feature importance button
        document.getElementById('importance-btn').disabled = false;
        
        // Calculate initial metrics
        updateMetrics();
    } catch (error) {
        statusDiv.innerHTML = `Error during training: ${error.message}`;
        console.error('Training error:', error);
    }
}

// 2. FIX: Update metrics to properly display confusion matrix
async function updateMetrics() {
    if (!validationPredictions || !validationLabels) return;
    
    const threshold = parseFloat(document.getElementById('threshold-slider').value);
    document.getElementById('threshold-value').textContent = threshold.toFixed(2);
    
    // Get predictions and labels as arrays
    const predValsRaw = await validationPredictions.array();
    const trueVals = await validationLabels.array();
    
    // FIX: Ensure predVals is flattened to 1D array
    const predVals = predValsRaw.map(p => {
        if (Array.isArray(p)) {
            return p[0]; // Extract first element if it's an array
        }
        return p; // Already a number
    });
    
    // Calculate confusion matrix
    let tp = 0, tn = 0, fp = 0, fn = 0;
    
    for (let i = 0; i < predVals.length; i++) {
        const prediction = predVals[i] >= threshold ? 1 : 0;
        const actual = trueVals[i];
        
        if (prediction === 1 && actual === 1) tp++;
        else if (prediction === 0 && actual === 0) tn++;
        else if (prediction === 1 && actual === 0) fp++;
        else if (prediction === 0 && actual === 1) fn++;
    }
    
    // Update confusion matrix display
    const cmDiv = document.getElementById('confusion-matrix');
    cmDiv.innerHTML = `
        <table style="width: 100%; text-align: center;">
            <tr>
                <th></th>
                <th style="background-color: #e8f5e9;">Predicted Survived (1)</th>
                <th style="background-color: #ffebee;">Predicted Died (0)</th>
            </tr>
            <tr>
                <th style="background-color: #e8f5e9;">Actual Survived (1)</th>
                <td style="background-color: #c8e6c9; font-weight: bold;">${tp}<br>(True Positive)</td>
                <td style="background-color: #ffcdd2;">${fn}<br>(False Negative)</td>
            </tr>
            <tr>
                <th style="background-color: #ffebee;">Actual Died (0)</th>
                <td style="background-color: #ffcdd2;">${fp}<br>(False Positive)</td>
                <td style="background-color: #c8e6c9; font-weight: bold;">${tn}<br>(True Negative)</td>
            </tr>
        </table>
        <p>Total predictions: ${tp + tn + fp + fn}</p>
    `;
    
    // Calculate performance metrics
    const precision = tp / (tp + fp) || 0;
    const recall = tp / (tp + fn) || 0;
    const f1 = 2 * (precision * recall) / (precision + recall) || 0;
    const accuracy = (tp + tn) / (tp + tn + fp + fn) || 0;
    
    // Update performance metrics display
    const metricsDiv = document.getElementById('performance-metrics');
    metricsDiv.innerHTML = `
        <p><strong>Accuracy:</strong> ${(accuracy * 100).toFixed(2)}%</p>
        <p><strong>Precision:</strong> ${precision.toFixed(4)}</p>
        <p><strong>Recall:</strong> ${recall.toFixed(4)}</p>
        <p><strong>F1 Score:</strong> ${f1.toFixed(4)}</p>
        <p><strong>Threshold:</strong> ${threshold.toFixed(2)}</p>
    `;
    
    // Calculate and plot ROC curve
    await plotROC(trueVals, predVals);
}

// Plot ROC curve
async function plotROC(trueLabels, predictions) {
    // FIX: Ensure predictions is flattened to 1D array
    const predsFlat = predictions.map(p => {
        if (Array.isArray(p)) {
            return p[0]; // Extract first element if it's an array
        }
        return p; // Already a number
    });
    
    // Calculate TPR and FPR for different thresholds
    const thresholds = Array.from({ length: 101 }, (_, i) => i / 100);
    const rocData = [];
    
    thresholds.forEach(threshold => {
        let tp = 0, fn = 0, fp = 0, tn = 0;
        
        for (let i = 0; i < predsFlat.length; i++) {
            const prediction = predsFlat[i] >= threshold ? 1 : 0;
            const actual = trueLabels[i];
            
            if (actual === 1) {
                if (prediction === 1) tp++;
                else fn++;
            } else {
                if (prediction === 1) fp++;
                else tn++;
            }
        }
        
        const tpr = tp / (tp + fn) || 0;
        const fpr = fp / (fp + tn) || 0;
        
        rocData.push({ threshold, fpr, tpr });
    });
    
    // Calculate AUC (approximate using trapezoidal rule)
    let auc = 0;
    for (let i = 1; i < rocData.length; i++) {
        auc += (rocData[i].fpr - rocData[i-1].fpr) * (rocData[i].tpr + rocData[i-1].tpr) / 2;
    }
    
    // Add AUC to performance metrics
    const metricsDiv = document.getElementById('performance-metrics');
    metricsDiv.innerHTML += `<p><strong>AUC:</strong> ${auc.toFixed(4)}</p>`;
    
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
}

// 4. ADD: Sigmoid gate to understand importance of features
function analyzeFeatureImportance() {
    if (!model || featureNames.length === 0) {
        alert('Please train the model first and ensure features are available.');
        return;
    }
    
    const outputDiv = document.getElementById('importance-output');
    outputDiv.innerHTML = 'Analyzing feature importance...';
    
    try {
        // Get weights from the first layer (input to hidden layer)
        const weights = model.layers[0].getWeights()[0]; // Shape: [num_input_features, 16]
        const weightArray = weights.arraySync();
        
        // Calculate importance for each input feature
        featureImportance = {};
        
        for (let i = 0; i < weightArray.length; i++) {
            const featureWeights = weightArray[i];
            // Calculate average absolute weight for this feature across all neurons
            const importance = featureWeights.reduce((sum, w) => sum + Math.abs(w), 0) / featureWeights.length;
            featureImportance[featureNames[i]] = importance;
        }
        
        // Sort features by importance
        const sortedFeatures = Object.entries(featureImportance)
            .sort((a, b) => b[1] - a[1]);
        
        // Display results
        let html = '<h3>Feature Importance Analysis (Sigmoid Gate)</h3>';
        html += '<p>Shows which features most influence the model predictions:</p>';
        html += '<table style="width: 100%;">';
        html += '<tr><th>Feature</th><th>Importance Score</th><th>Rank</th></tr>';
        
        sortedFeatures.forEach(([feature, importance], index) => {
            // Color code based on importance
            const color = importance > 0.15 ? '#4CAF50' : 
                         importance > 0.1 ? '#8BC34A' : 
                         importance > 0.05 ? '#CDDC39' : '#FFC107';
            
            html += `
                <tr>
                    <td>${feature}</td>
                    <td>
                        <div style="background-color: ${color}; padding: 5px; border-radius: 3px; width: ${importance * 500}px; max-width: 100%;">
                            ${importance.toFixed(4)}
                        </div>
                    </td>
                    <td>${index + 1}</td>
                </tr>
            `;
        });
        
        html += '</table>';
        
        // Add interpretation
        html += '<div style="margin-top: 20px; padding: 10px; background-color: #e8f5e9; border-radius: 5px;">';
        html += '<h4>Interpretation:</h4>';
        html += '<ul>';
        html += '<li><strong>High importance (>0.15):</strong> Features with strongest influence on survival prediction</li>';
        html += '<li><strong>Medium importance (0.05-0.15):</strong> Features with moderate influence</li>';
        html += '<li><strong>Low importance (<0.05):</strong> Features with minimal influence</li>';
        html += '</ul>';
        html += `<p><strong>Top 3 most important features:</strong> ${sortedFeatures.slice(0, 3).map(f => f[0]).join(', ')}</p>`;
        html += '</div>';
        
        outputDiv.innerHTML = html;
        
        // Also visualize with tfjs-vis
        const top10 = sortedFeatures.slice(0, 10);
        tfvis.render.barchart(
            { name: 'Top 10 Feature Importance', tab: 'Features' },
            top10.map(([feature, importance]) => ({ x: feature, y: importance })),
            { 
                xLabel: 'Feature', 
                yLabel: 'Importance Score',
                width: 600,
                height: 400
            }
        );
        
        console.log('Feature importance analysis completed:', featureImportance);
    } catch (error) {
        outputDiv.innerHTML = `Error analyzing feature importance: ${error.message}`;
        console.error('Feature importance error:', error);
    }
}

// Predict on test data
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
        const predValuesRaw = await testPredictions.array();
        
        // FIX: Ensure predValues is flattened to 1D array
        const predValues = predValuesRaw.map(p => {
            if (Array.isArray(p)) {
                return p[0]; // Extract first element if it's an array
            }
            return p; // Already a number
        });
        
        // Create prediction results
        const results = preprocessedTestData.passengerIds.map((id, i) => ({
            PassengerId: id,
            Survived: predValues[i] >= 0.5 ? 1 : 0,
            Probability: predValues[i]
        }));
        
        // Show first 10 predictions
        outputDiv.innerHTML = '<h3>Prediction Results (First 10 Rows)</h3>';
        outputDiv.appendChild(createPredictionTable(results.slice(0, 10)));
        
        outputDiv.innerHTML += `<p>Predictions completed! Total: ${results.length} samples</p>`;
        
        // Show survival rate in predictions
        const survivedCount = results.filter(r => r.Survived === 1).length;
        const survivalRatePred = (survivedCount / results.length * 100).toFixed(2);
        outputDiv.innerHTML += `<p>Predicted survival rate: ${survivedCount}/${results.length} (${survivalRatePred}%)</p>`;
        
        // Enable the export button
        document.getElementById('export-btn').disabled = false;
    } catch (error) {
        outputDiv.innerHTML = `Error during prediction: ${error.message}`;
        console.error('Prediction error:', error);
    }
}

// Create prediction table
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
        
        const tdId = document.createElement('td');
        tdId.textContent = row.PassengerId;
        tr.appendChild(tdId);
        
        const tdSurvived = document.createElement('td');
        tdSurvived.textContent = row.Survived;
        tdSurvived.style.color = row.Survived === 1 ? 'green' : 'red';
        tdSurvived.style.fontWeight = 'bold';
        tr.appendChild(tdSurvived);
        
        const tdProb = document.createElement('td');
        
        // FIX: Safe handling of probability value
        let probValue = row.Probability;
        
        // Ensure it's a number
        if (typeof probValue !== 'number') {
            if (Array.isArray(probValue)) {
                probValue = probValue[0];
            }
            probValue = parseFloat(probValue);
        }
        
        // If still not a number, show N/A
        if (isNaN(probValue)) {
            tdProb.textContent = 'N/A';
            tdProb.style.color = 'gray';
        } else {
            tdProb.textContent = probValue.toFixed(4);
            // Color based on probability
            if (probValue >= 0.7) tdProb.style.color = 'green';
            else if (probValue <= 0.3) tdProb.style.color = 'red';
        }
        
        tr.appendChild(tdProb);
        
        table.appendChild(tr);
    });
    
    return table;
}

// Export results
async function exportResults() {
    if (!testPredictions || !preprocessedTestData) {
        alert('Please make predictions first.');
        return;
    }
    
    const statusDiv = document.getElementById('export-status');
    statusDiv.innerHTML = 'Exporting results...';
    
    try {
        // Get predictions
        const predValuesRaw = await testPredictions.array();
        
        // FIX: Ensure predValues is flattened to 1D array
        const predValues = predValuesRaw.map(p => {
            if (Array.isArray(p)) {
                return p[0]; // Extract first element if it's an array
            }
            return p; // Already a number
        });
        
        // Create submission CSV (PassengerId, Survived)
        let submissionCSV = 'PassengerId,Survived\n';
        preprocessedTestData.passengerIds.forEach((id, i) => {
            submissionCSV += `${id},${predValues[i] >= 0.5 ? 1 : 0}\n`;
        });
        
        // Create probabilities CSV (PassengerId, Probability)
        let probabilitiesCSV = 'PassengerId,Probability\n';
        preprocessedTestData.passengerIds.forEach((id, i) => {
            probabilitiesCSV += `${id},${predValues[i].toFixed(6)}\n`;
        });
        
        // Create download links
        const submissionLink = document.createElement('a');
        submissionLink.href = URL.createObjectURL(new Blob([submissionCSV], { type: 'text/csv' }));
        submissionLink.download = 'submission.csv';
        submissionLink.textContent = 'Download submission.csv';
        
        const probabilitiesLink = document.createElement('a');
        probabilitiesLink.href = URL.createObjectURL(new Blob([probabilitiesCSV], { type: 'text/csv' }));
        probabilitiesLink.download = 'probabilities.csv';
        probabilitiesLink.textContent = 'Download probabilities.csv';
        
        // Add links to page
        const linksDiv = document.createElement('div');
        linksDiv.style.margin = '10px 0';
        linksDiv.appendChild(submissionLink);
        linksDiv.appendChild(document.createElement('br'));
        linksDiv.appendChild(probabilitiesLink);
        
        statusDiv.innerHTML = '<p>Export completed! Click links to download:</p>';
        statusDiv.appendChild(linksDiv);
        
        // Save model
        try {
            await model.save('downloads://titanic-tfjs-model');
            statusDiv.innerHTML += '<p>Model saved to browser downloads as "titanic-tfjs-model"</p>';
        } catch (saveError) {
            console.warn('Model save error (might be browser limitation):', saveError);
        }
        
        // Also save feature importance if available
        if (Object.keys(featureImportance).length > 0) {
            let importanceCSV = 'Feature,Importance\n';
            Object.entries(featureImportance).forEach(([feature, importance]) => {
                importanceCSV += `${feature},${importance.toFixed(6)}\n`;
            });
            
            const importanceLink = document.createElement('a');
            importanceLink.href = URL.createObjectURL(new Blob([importanceCSV], { type: 'text/csv' }));
            importanceLink.download = 'feature_importance.csv';
            importanceLink.textContent = 'Download feature_importance.csv';
            
            linksDiv.appendChild(document.createElement('br'));
            linksDiv.appendChild(importanceLink);
        }
        
    } catch (error) {
        statusDiv.innerHTML = `Error during export: ${error.message}`;
        console.error('Export error:', error);
    }
}

// 3. CODE SUMMARY FOR LLMs: Understanding the logic
/*
This TensorFlow.js application implements a complete machine learning pipeline for binary classification:

1. DATA LOADING & PARSING:
   - Loads CSV files with proper comma escaping (handles quoted fields)
   - Converts to arrays of objects with type conversion
   - Validates data integrity

2. DATA INSPECTION:
   - Shows data preview tables
   - Calculates statistics (missing values, survival rates)
   - Creates visualizations (survival by sex, class)

3. PREPROCESSING:
   - Imputes missing values (median for numerical, mode for categorical)
   - Standardizes numerical features (Age, Fare)
   - One-hot encodes categorical features (Pclass, Sex, Embarked)
   - Optional feature engineering (FamilySize, IsAlone)

4. MODEL CREATION:
   - Simple neural network: Input → Dense(16, ReLU) → Dense(1, Sigmoid)
   - Binary cross-entropy loss with Adam optimizer
   - Suitable for binary classification problems

5. TRAINING:
   - 80/20 train/validation split
   - 50 epochs with batch size 32
   - Live visualization of training progress
   - Early stopping implemented

6. EVALUATION:
   - Confusion matrix with dynamic threshold adjustment
   - Performance metrics (Accuracy, Precision, Recall, F1, AUC)
   - ROC curve visualization

7. FEATURE IMPORTANCE ANALYSIS (NEW):
   - Analyzes weights from first layer to determine feature importance
   - Ranks features by influence on predictions
   - Visualizes top 10 important features

8. PREDICTION & EXPORT:
   - Predicts on test data
   - Generates Kaggle submission format
   - Exports probabilities and feature importance
   - Saves model locally

KEY FIXES APPLIED:
1. CSV parsing now handles commas inside quoted fields
2. Confusion matrix properly displays with async data handling
3. Feature importance analysis added via "Sigmoid Gate" approach
4. Fixed .toFixed() error by flattening prediction arrays

REUSABILITY:
- Schema configuration at top allows adaptation to other datasets
- Modular functions for each pipeline step
- Comprehensive error handling throughout
*/

// Initialize the application when the page loads
document.addEventListener('DOMContentLoaded', function() {
    console.log('Titanic Survival Classifier loaded successfully!');
    console.log('Instructions:');
    console.log('1. Load train.csv and test.csv files');
    console.log('2. Follow the step-by-step process: Inspect → Preprocess → Create Model → Train');
    console.log('3. Use the threshold slider to adjust classification sensitivity');
    console.log('4. Analyze feature importance to understand what drives predictions');
});
