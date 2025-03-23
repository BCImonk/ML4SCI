# Feature Selection: Code Comparison

This project explores different feature selection techniques for EEG data classification to identify the most significant features for distinguishing between two metabolic states.

## Approach 1: Basic Feature Selection

The initial approach focused on applying standard feature selection methods to EEG band power data without considering the spatial relationships between electrodes. The goal was to establish a baseline for identifying discriminative features using statistical and machine learning approaches.

### Architecture
- **Preprocessing**: Standard scaling of features
- **Feature Selection**:
  - Univariate Feature Selection (UFS) using ANOVA F-values
  - Recursive Feature Elimination (RFE) with RandomForest
  - Principal Component Analysis (PCA)
- **Model**: Random Forest Classifier
- **Evaluation**: 5-fold stratified cross-validation

### Performance
- UFS Mean Accuracy: 27.50%
- RFE Mean Accuracy: 57.50%
- PCA Mean Accuracy: 32.50%
- Best performing method: RFE (57.50%)
- Top features identified:
  - UFS: 'beta23', 'delta23', 'delta41', 'delta51', 'theta23'
  - RFE: 'beta23', 'delta43', 'delta44', 'delta47', 'gamma5'
  - PCA: 'delta14', 'delta24', 'delta27', 'alpha17', 'alpha18'
- Common features across methods: 'beta23' (in UFS and RFE)

### Limitations
- Treats EEG channels as independent features, ignoring spatial relationships
- Limited to a single classifier algorithm
- Relatively low classification performance
- No hyperparameter tuning
- Minimal visualization of results
- Lack of topological feature engineering

## Approach 2: Topological Feature Engineering

This enhanced approach was conceived to leverage neurophysiological knowledge by incorporating the topological organization of EEG electrodes. Instead of treating channels as independent, this method groups electrodes by brain regions and extracts regional features, recognizing the spatial relationships inherent in EEG data.

### Architecture
- **Preprocessing**: Standard scaling of features
- **Topological Feature Engineering**:
  - Electrode grouping by brain regions (frontal, central, parietal, temporal, occipital)
  - Regional aggregation (mean, std, max, min) for each frequency band
- **Feature Selection**:
  - Univariate Feature Selection using mutual information (instead of F-values)
  - Recursive Feature Elimination with RandomForest
  - Principal Component Analysis
- **Models**:
  - RandomForest Classifier
  - GradientBoosting Classifier
  - Support Vector Machine (SVM)
- **Evaluation**: 5-fold stratified cross-validation with variance reporting
- **Visualization**: Custom visualizations for model performance and feature importance

### Performance
- UFS (GradientBoost) Accuracy: 75.00% (±7.91%)
- RFE (GradientBoost) Accuracy: 70.00% (±15.00%)
- PCA (GradientBoost) Accuracy: 35.00% (±20.00%)
- Best performing method: UFS with GradientBoost (75.00%)
- Top features identified:
  - UFS: 'delta44', 'delta45', 'gamma36', 'gamma43', 'delta_temporal_min'
  - RFE: 'beta23', 'delta1', 'delta47', 'delta_temporal_std', 'gamma_occipital_std'
  - PCA: 'delta_occipital_max', 'delta_occipital_std', 'delta27', 'alpha_central_mean', 'alpha18'
- No common features found across methods

### Advantages Over Previous Approach
1. **Neurophysiological relevance**: Incorporates brain region topology for more meaningful feature extraction
2. **Enhanced feature engineering**: Creates 100 additional derived features based on neuroanatomical regions
3. **Model diversity**: Tests multiple classifiers to identify optimal performance
4. **Improved feature selection**: Uses mutual information which captures non-linear relationships
5. **Better visualization**: Includes detailed plots for analysis and interpretation
6. **More comprehensive evaluation**: Provides standard deviations for cross-validation scores
7. **Superior performance**: 75% accuracy compared to 57.5% in the first approach

### Limitations and Future Improvements
- High dimensionality (420+ features including derived ones) with limited sample size (40 participants) risks overfitting
- No implementation of nested cross-validation for more robust model evaluation
- Limited analysis of feature importance for the derived topological features
- No explicit handling of potential correlations between derived features

## Potential Improvements
1. Implement more advanced feature selection strategies (e.g., stability selection)
2. Apply deep learning approaches (CNNs, RNNs) that can automatically learn spatial patterns
3. Create ensemble methods combining different feature sets
4. Implement more sophisticated cross-validation strategies like nested CV
5. Incorporate temporal dynamics of EEG signals
6. Expand dataset size to improve generalizability
7. Apply regularization techniques to handle high dimensionality

## References
1. Lotte, F., Bougrain, L., Cichocki, A., Clerc, M., Congedo, M., Rakotomamonjy, A., & Yger, F. (2018). A review of classification algorithms for EEG-based brain-computer interfaces: a 10 year update. Journal of Neural Engineering, 15(3), 031005.
2. Roy, Y., Banville, H., Albuquerque, I., Gramfort, A., Falk, T. H., & Faubert, J. (2019). Deep learning-based electroencephalography analysis: a systematic review. Journal of Neural Engineering, 16(5), 051001.
3. Craik, A., He, Y., & Contreras-Vidal, J. L. (2019). Deep learning for electroencephalogram (EEG) classification tasks: a review. Journal of Neural Engineering, 16(3), 031001.
4. Schirrmeister, R. T., Springenberg, J. T., Fiederer, L. D. J., Glasstetter, M., Eggensperger, K., Tangermann, M., ... & Ball, T. (2017). Deep learning with convolutional neural networks for EEG decoding and visualization. Human Brain Mapping, 38(11), 5391-5420.
