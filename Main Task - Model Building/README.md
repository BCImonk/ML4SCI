# Model Comparison

## Basic Model

This implementation uses a straightforward neural network approach to EEG classification. The idea was to establish a baseline model with regularization techniques to prevent overfitting given the small dataset size (40 samples).

### Architecture
- Simple feedforward neural network (`SimpleEEGModel`)
- 4 fully connected layers: input → 64 → 32 → 16 → 1
- LeakyReLU activation and dropout for regularization
- Sigmoid output for binary classification
- 5-fold cross-validation strategy
- SelectKBest feature selection

### Advantages
- Simple and straightforward implementation as a baseline
- Effective regularization techniques to combat overfitting
- Data augmentation to increase training set size
- Ensemble prediction from cross-validation models
- Comprehensive visualization of model performance

### Performance
- Final Accuracy: 0.7000
- Final F1 Score: 0.6250
- Mean CV Accuracy: 0.5000 ± 0.1118
- Mean CV F1 Score: 0.4022 ± 0.2205

### Drawbacks and Scope of Improvements
- Basic model architecture with moderate performance
- High variance in cross-validation results suggests instability
- No utilization of spatial information from EEG channels
- No specific handling of different frequency bands
- Could benefit from more sophisticated regularization techniques
- Early stopping implementation could be refined

## Basic Model + Topological

This implementation hypothesizes that delta band features have higher discriminative power for the classification task. It creates specialized features focusing on delta band and relationships between frequency bands.

### Architecture
- `DeltaFeatureModel`: Neural network with batch normalization
- 3 hidden layers with ReLU activation: input → 128 → 64 → 32 → 1
- Batch normalization and dropout for regularization
- Ensemble of 10 models for robust prediction
- Custom feature extraction pipeline for delta band
- Threshold optimization for decision boundary tuning

### Advantages
- Band-specific feature extraction leverages domain knowledge
- Batch normalization helps stabilize training
- Ensemble of 10 models increases robustness
- Extensive visualization of data distributions and model performance
- Threshold optimization improves classification performance

### Changes from Previous Model
- Introduction of batch normalization layers
- Focus on delta band features rather than using all features equally
- Calculation of band power ratios as additional features
- Larger ensemble (10 models vs. 5 models)
- Threshold optimization for final classification

### Performance
- Final Accuracy: 0.7250
- Final F1 Score: 0.7027
- Test Accuracy: 0.6250
- AUC Score: 0.4375

### Drawbacks and Scope of Improvements
- Limited cross-validation performance with high variance
- Focuses mainly on delta band features, potentially missing important information
- No spatial information from EEG channel relationships
- Could benefit from more sophisticated ensemble techniques
- Threshold optimization could be improved
- ROC curve analysis shows limited discriminative ability

## Basic + SVM-esque

This approach implements a dual-model strategy, comparing an SVM-inspired model with a neural network architecture. It focuses on delta band features but also incorporates band ratios, which are known biomarkers in EEG analysis.

### Architecture
- Two model architectures:
  1. `SVMInspiredModel`: Linear model with hinge loss
  2. `EEGClassifier`: Neural network with batch normalization and dropout
- Neural network: input → 32 → 16 → 8 → 1 with ReLU activation
- Comprehensive data augmentation (noise addition, interpolation, scaling)
- Strategic feature extraction focusing on important delta channels
- Ensemble prediction from both model types

### Advantages
- Dual-model approach provides different modeling perspectives
- SVM-inspired model offers better generalization for small datasets
- Advanced feature engineering with statistical features (kurtosis, skew)
- More sophisticated data augmentation techniques
- Auto-selection of the better-performing model per fold

### Changes from Previous Model
- Introduction of dual modeling approach (SVM + Neural Network)
- More sophisticated feature engineering (kurtosis, skew, etc.)
- Dynamic model selection based on validation performance
- Expanded set of evaluation metrics (MCC, Cohen's Kappa)
- Introduction of hinge loss for SVM-inspired training

### Performance
- Final Accuracy: 0.9000
- Final F1 Score: 0.8889
- Neural Network performance:
  - Accuracy: 0.9000
  - F1 Score: 0.8889
  - AUC: 0.9800
- SVM performance:
  - Accuracy: 0.6500
  - F1 Score: 0.7308

### Drawbacks and Scope of Improvements
- Relies heavily on manual feature engineering
- Complex implementation with multiple models increases complexity
- Potential overfitting indicated by performance discrepancy
- No deep learning approach for automatic feature extraction
- Could leverage more spatial information from EEG channels
- Augmentation techniques could be more diverse

## Optimized Model

This approach focuses on dimensionality reduction via PCA combined with ensemble learning for robust classification. It uses delta-specific features while employing diverse model architectures in the ensemble.

### Architecture
- `EEGClassifier`: Configurable neural network with flexible hidden layers
- Default architecture: input → 128 → 64 → 32 → 1 with ReLU activation
- PCA for feature dimensionality reduction
- Ensemble of 15 models with varied architectures
- Bias and threshold optimization for ensemble prediction
- Leave-one-out cross-validation for robust performance estimation

### Advantages
- PCA reduces dimensionality and potential noise in features
- Flexible model architecture allows for experimentation
- Large ensemble (15 models) with architectural diversity
- Leave-one-out validation provides robust performance estimates
- Bias and threshold optimization for ensemble predictions

### Changes from Previous Model
- Introduction of PCA for dimensionality reduction
- Configurable network architecture for each ensemble member
- Increased ensemble size to 15 models
- Implementation of leave-one-out validation
- Bias parameter alongside threshold for classification decisions

### Performance
- Final accuracy (from leave-one-out): approximately 0.52
- Full dataset accuracy: approximately 0.92
- Test accuracy: approximately 0.6
- AUC values varied by model

### Drawbacks and Scope of Improvements
- Complex ensemble approach reduces interpretability
- PCA may discard important discriminative features
- Leave-one-out validation is computationally expensive
- Multiple hyperparameters make optimization challenging
- No explicit spatial modeling of EEG channels
- Could benefit from more advanced deep learning architectures

## Final Model

This implementation takes a sophisticated approach by incorporating spatial information through Graph Neural Networks and attention mechanisms. It's designed to exploit the topological structure of EEG data and learn the importance of different frequency bands and channels.

### Architecture
- Complex models with spatial awareness:
  1. `EEGClassifier`: Incorporates spatial GNN, channel attention, and band attention
  2. `SimplifiedEEGClassifier`: Simplified version without band attention
- Spatial GNN uses adjacency matrix to model channel relationships
- Attention mechanisms learn importance weights for bands and channels
- Feature extraction and classification networks with batch normalization
- Comprehensive metrics tracking and overfitting analysis

### Advantages
- Incorporates spatial relationships between EEG channels
- Attention mechanisms identify important frequency bands and channels
- Graph Neural Network captures topological information
- Extensive metrics tracking and overfitting detection
- Learning curve analysis for model evaluation
- Feature activation visualization provides interpretability

### Changes from Previous Model
- Introduction of Graph Neural Networks for spatial modeling
- Addition of attention mechanisms for feature importance
- Adjacency matrix creation for modeling channel relationships
- Comprehensive overfitting analysis system
- Electrode importance visualization for interpretability
- Implementation of more sophisticated evaluation metrics

### Performance
- Mean Accuracy: 0.9806 ± 0.0180
- Mean F1 Score: 0.9803 ± 0.0182
- Mean AUC: 0.9970 ± 0.0041
- Ensemble Model:
  - Accuracy: 1.0000
  - F1 Score: 1.0000
  - AUC: 1.0000

### Drawbacks and Scope of Improvements
- Highly complex model architecture increases training time
- Signs of overfitting detected (average overfitting ratio: 4.2750)
- Perfect performance on ensemble may indicate data leakage or overfitting
- Computationally intensive
- Many hyperparameters that need tuning
- Could benefit from transfer learning or pre-training

## References
1. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. CVPR.
2. Ioffe, S., & Szegedy, C. (2015). Batch normalization: Accelerating deep network training by reducing internal covariate shift. ICML.
3. Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). Dropout: a simple way to prevent neural networks from overfitting. JMLR.
4. Lotte, F., Bougrain, L., Cichocki, A., Clerc, M., Congedo, M., Rakotomamonjy, A., & Yger, F. (2018). A review of classification algorithms for EEG-based brain-computer interfaces: a 10 year update. Journal of Neural Engineering.
5. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017). Attention is all you need. NeurIPS.
