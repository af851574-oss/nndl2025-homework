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
let featureNames = []; // Store feature names for visualization
let featureImportance = {}; // Store feature importance

// Schema configuration - change these for different datasets
const TARGET_FEATURE = 'Survived'; // Binary classification target
const ID_FEATURE = 'PassengerId'; // Identifier to exclude from features
const NUMERICAL_FEATURES = ['Age', 'Fare', 'SibSp', 'Parch']; // Numerical features
const CATEGORICAL_FEATURES = ['Pclass', 'Sex', 'Embarked']; // Categorical features

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
        
        statusDiv.innerHTML = `Data loaded successfully! Training: ${trainData.length} samples, Test: ${testData.length} samples`;
        
        // Enable the inspect button
        document.getElementById('inspect-btn').disabled = false;
    } catch (error) {
        statusDiv.innerHTML = `Error loading data: ${error.message}`;
        console.error(error);
    }
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

// Parse CSV text to array of objects (RFC 4180 compliant - handles quoted fields with commas)
function parseCSV(csvText) {
    const lines = csvText.split('\n').filter(line => line.trim() !== '');

    // Parse header row
    const headers = parseCSVLine(lines[0]);

    // Parse data rows
    return lines.slice(1).map(line => {
        const values = parseCSVLine(line);
        const obj = {};
        headers.forEach((header, i) => {
            // Handle missing values (empty strings)
            const value = values[i] !== undefined ? values[i].trim() : '';
            obj[header] = value === '' ? null : value;

            // Convert numerical values to numbers if possible
            if (!isNaN(obj[header]) && obj[header] !== null && obj[header] !== '') {
                obj[header] = parseFloat(obj[header]);
            }
        });
        return obj;
    });
}

// Parse a single CSV line handling quoted fields with commas
function parseCSVLine(line) {
    const result = [];
    let current = '';
    let inQuotes = false;

    for (let i = 0; i < line.length; i++) {
        const char = line[i];
        const nextChar = line[i + 1];

        if (char === '"') {
            if (inQuotes && nextChar === '"') {
                // Escaped quote (two consecutive quotes)
                current += '"';
                i++; // Skip next quote
            } else {
                // Toggle quote mode
                inQuotes = !inQuotes;
            }
        } else if (char === ',' && !inQuotes) {
            // Field separator (only outside quotes)
            result.push(current);
            current = '';
        } else {
            current += char;
        }
    }

    // Push the last field
    result.push(current);

    return result;
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
    
    const shapeInfo = `Dataset shape: ${trainData.length} rows x ${Object.keys(trainData[0]).length} columns`;
    const survivalCount = trainData.filter(row => row[TARGET_FEATURE] === 1).length;
    const survivalRate = (survivalCount / trainData.length * 100).toFixed(2);
    const targetInfo = `Survival rate: ${survivalCount}/${trainData.length} (${survivalRate}%)`;
    
    // Calculate missing values percentage for each feature
    let missingInfo = '<h4>Missing Values Percentage:</h4><ul>';
    Object.keys(trainData[0]).forEach(feature => {
        const missingCount = trainData.filter(row => row[feature] === null || row[feature] === undefined).length;
        const missingPercent = (missingCount / trainData.length * 100).toFixed(2);
        missingInfo += `<li>${feature}: ${missingPercent}%</li>`;
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
        index: sex,
        value: (stats.survived / stats.total) * 100
    }));

    tfvis.render.barchart(
        { name: 'Survival Rate by Sex', tab: 'Charts' },
        sexData,
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
        index: `Class ${pclass}`,
        value: (stats.survived / stats.total) * 100
    }));

    tfvis.render.barchart(
        { name: 'Survival Rate by Passenger Class', tab: 'Charts' },
        pclassData,
        { xLabel: 'Passenger Class', yLabel: 'Survival Rate (%)' }
    );
    
    chartsDiv.innerHTML += '<p>Charts are displayed in the tfjs-vis visor. Click the button in the bottom right to view.</p>';
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
        const ageMedian = calculateMedian(trainData.map(row => row.Age).filter(age => age !== null));
        const fareMedian = calculateMedian(trainData.map(row => row.Fare).filter(fare => fare !== null));
        const embarkedMode = calculateMode(trainData.map(row => row.Embarked).filter(e => e !== null));
        
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
        `;
        
        // Enable the create model button
        document.getElementById('create-model-btn').disabled = false;
    } catch (error) {
        outputDiv.innerHTML = `Error during preprocessing: ${error.message}`;
        console.error(error);
    }
}

// Extract features from a row with imputation and normalization
function extractFeatures(row, ageMedian, fareMedian, embarkedMode) {
    // Impute missing values
    const age = row.Age !== null ? row.Age : ageMedian;
    const fare = row.Fare !== null ? row.Fare : fareMedian;
    const embarked = row.Embarked !== null ? row.Embarked : embarkedMode;
    
    // Standardize numerical features
    const standardizedAge = (age - ageMedian) / (calculateStdDev(trainData.map(r => r.Age).filter(a => a !== null)) || 1);
    const standardizedFare = (fare - fareMedian) / (calculateStdDev(trainData.map(r => r.Fare).filter(f => f !== null)) || 1);
    
    // One-hot encode categorical features
    const pclassOneHot = oneHotEncode(row.Pclass, [1, 2, 3]); // Pclass values: 1, 2, 3
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

// Calculate median of an array
function calculateMedian(values) {
    if (values.length === 0) return 0;
    
    values.sort((a, b) => a - b);
    const half = Math.floor(values.length / 2);
    
    if (values.length % 2 === 0) {
        return (values[half - 1] + values[half]) / 2;
    }
    
    return values[half];
}

// Calculate mode of an array
function calculateMode(values) {
    if (values.length === 0) return null;
    
    const frequency = {};
    let maxCount = 0;
    let mode = null;
    
    values.forEach(value => {
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
    if (values.length === 0) return 0;
    
    const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
    const squaredDiffs = values.map(value => Math.pow(value - mean, 2));
    const variance = squaredDiffs.reduce((sum, val) => sum + val, 0) / values.length;
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

// Create the model - simple homework version
function createModel() {
    if (!preprocessedTrainData) {
        alert('Please preprocess data first.');
        return;
    }

    const inputShape = preprocessedTrainData.features.shape[1];

    // Define feature names for visualization
    featureNames = ['Age', 'Fare', 'SibSp', 'Parch',
                    'Pclass_1', 'Pclass_2', 'Pclass_3',
                    'Sex_male', 'Sex_female',
                    'Embarked_C', 'Embarked_Q', 'Embarked_S'];
    if (document.getElementById('add-family-features').checked) {
        featureNames.push('FamilySize', 'IsAlone');
    }

    // Create a sequential model
    model = tf.sequential();

    // Add layers
    model.add(tf.layers.dense({
        units: 16,
        activation: 'relu',
        inputShape: [inputShape],
        name: 'hidden_layer'
    }));

    model.add(tf.layers.dense({
        units: 1,
        activation: 'sigmoid',
        name: 'output_layer'
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
    summaryText += '<li><strong>Input Layer:</strong> ' + inputShape + ' features</li>';
    summaryText += '<li><strong>Hidden Layer:</strong> 16 units with ReLU activation</li>';
    summaryText += '<li><strong>Output Layer:</strong> 1 unit with Sigmoid activation (binary classification)</li>';
    summaryText += '</ul>';
    summaryText += `<p>Total parameters: ${model.countParams()}</p>`;
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
        
        // Train the model with combined callbacks
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
                    ['loss', 'acc', 'val_loss', 'val_acc']
                )
            }
        });
        
        statusDiv.innerHTML += '<p>Training completed!</p>';

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
        console.error(error);
    }
}

// Update metrics based on threshold
async function updateMetrics() {
    if (!validationPredictions || !validationLabels) return;
    
    const threshold = parseFloat(document.getElementById('threshold-slider').value);
    document.getElementById('threshold-value').textContent = threshold.toFixed(2);
    
    // Calculate confusion matrix
    const predVals = validationPredictions.arraySync();
    const trueVals = validationLabels.arraySync();
    
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
        <table>
            <tr><th></th><th>Predicted Positive</th><th>Predicted Negative</th></tr>
            <tr><th>Actual Positive</th><td>${tp}</td><td>${fn}</td></tr>
            <tr><th>Actual Negative</th><td>${fp}</td><td>${tn}</td></tr>
        </table>
    `;
    
    // Calculate performance metrics
    const precision = tp / (tp + fp) || 0;
    const recall = tp / (tp + fn) || 0;
    const f1 = 2 * (precision * recall) / (precision + recall) || 0;
    const accuracy = (tp + tn) / (tp + tn + fp + fn) || 0;
    
    // Update performance metrics display
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

// Plot ROC curve
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
        
        const tpr = tp / (tp + fn) || 0;
        const fpr = fp / (fp + tn) || 0;
        
        rocData.push({ threshold, fpr, tpr });
    });
    
    // Sort ROC data by FPR (ascending) for correct AUC calculation
    rocData.sort((a, b) => a.fpr - b.fpr);

    // Calculate AUC (approximate using trapezoidal rule)
    let auc = 0;
    for (let i = 1; i < rocData.length; i++) {
        auc += (rocData[i].fpr - rocData[i-1].fpr) * (rocData[i].tpr + rocData[i-1].tpr) / 2;
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

// Analyze feature importance from first layer weights
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
        let html = '<h3>Feature Importance Analysis</h3>';
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
            top10.map(([feature, importance]) => ({ index: feature, value: importance })),
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
            <p>Export completed!</p>
            <p>Downloaded: submission.csv (Kaggle submission format)</p>
            <p>Downloaded: probabilities.csv (Prediction probabilities)</p>
            <p>Model saved to browser downloads</p>
        `;
    } catch (error) {
        statusDiv.innerHTML = `Error during export: ${error.message}`;
        console.error(error);
    }
}
