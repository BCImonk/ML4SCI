# **ML4SCI**

The Codes here are my Proposal to ML4SCI as part of GSOC'25 for the Topic: 
**CEBRA-Based Data Processing Pipeline for Mapping Time-Locked EEG Paired Sets in Interacting Participants.**

The Colab Link for the Proposal: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1lZWr-ZsqOEZQMPtlSknswL9zUIQUx6Gb?usp=sharing)

---

## **Proposal / Idea  : CEBRA-Based Data Processing Pipeline for Mapping Time-Locked EEG Paired Sets in Interacting Participants**

---
## **Main Task: Build a model for classifying the participant data into neural states using PyTorch or Keras.** [![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/BCImonk/ML4SCI/blob/main/Main%20Task%20-%20Model%20Building/Final%20Model.ipynb)

---
### Implementation Overview

I've developed a comprehensive deep learning pipeline for EEG neural state classification that achieves exceptional accuracy. Starting with a challenging dataset (40 samples, 320 features from 64 channels × 5 frequency bands), my solution leverages both neurophysiological domain knowledge and advanced deep learning techniques to extract meaningful patterns from brain activity data.

My approach transforms EEG signals into a sophisticated classification framework by:
1. Organizing the data by both frequency bands (alpha, beta, delta, theta, gamma) and 64 spatial channels
2. Implementing feature selection to identify the most informative neural markers
3. Using a data augmentation strategy to address the limited sample size
4. Developing a custom neural network architecture with attention mechanisms and graph neural networks
5. Creating an ensemble of models for robust prediction

**Results:** The ensemble model achieved 100% accuracy on the test set, with individual cross-validation folds averaging 98.06% accuracy (±1.8%). The system demonstrates strong performance across multiple evaluation metrics (F1: 98.03%, AUC: 99.70%).

---
### Technical Differentiators

My implementation extends beyond traditional classification approaches:

1. **Spatial-spectral neural architecture:** Leverages both the spatial arrangement of EEG channels and frequency band information through a custom neural network design

2. **Attention mechanisms:** Implements channel and band attention to dynamically weight the importance of different brain regions and frequency bands during classification

3. **Graph Neural Network integration:** Uses a custom spatial GNN layer to model the topological relationships between EEG channels, capturing neighborhood information in brain activity

4. **Sophisticated regularization:** Combines dropout, batch normalization, early stopping, and learning rate scheduling to prevent overfitting on the small dataset

5. **Comprehensive evaluation:** Employs cross-validation with extensive metrics tracking and visualization to ensure robust model assessment

* *To check out a simple and straightforward model that I've created on the same dataset:* [![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/BCImonk/ML4SCI/blob/main/Main%20Task%20-%20Model%20Building/Basic%20Model.ipynb)

* *Simple model taking topological information into account:* [![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/BCImonk/ML4SCI/blob/main/Main%20Task%20-%20Model%20Building/Basic%20Model%20%2B%20Topological.ipynb)

* *Another model with slightly lower accuracy (90%) but much decreased overfitting tendencies (SVM+ML)*: [![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/BCImonk/ML4SCI/blob/main/Main%20Task%20-%20Model%20Building/Basic%20Model%20%2B%20SVM-esque.ipynb)

* *A less advanced version of this code (Non Attention):* [![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/BCImonk/ML4SCI/blob/main/Main%20Task%20-%20Model%20Building/Optimized%20Model.ipynb)

* *Directory with All Models:* [![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/BCImonk/ML4SCI/tree/main/Main%20Task%20-%20Model%20Building)

---
### Implementation Architecture

```
┌─────────────────────┐
│ Data Loading (n=40) │
└──────────┬──────────┘
           ▼
┌─────────────────────────────┐
│ Feature Selection (150/320) │
└──────────┬──────────────────┘
           ▼
┌──────────────────────────────┐
│ Data Augmentation (n=720)    │
│ - Noise addition             │
│ - Scaling                    │
│ - Channel dropping           │
│ - Feature permutation        │
└──────────┬───────────────────┘
           ▼
┌──────────────────────────────┐
│ 5-Fold Cross-Validation      │
└──────────┬───────────────────┘
           ▼
┌──────────────────────────────────────────────┐
│ Custom Neural Network Architecture           │
│ ┌──────────────┐ ┌───────────────────────┐  │
│ │Spatial GNN   │ │EEG Channel Attention  │  │
│ └──────────────┘ └───────────────────────┘  │
│ ┌──────────────┐ ┌───────────────────────┐  │
│ │Band Attention│ │MLP Classifier         │  │
│ └──────────────┘ └───────────────────────┘  │
└──────────────────┬───────────────────────────┘
                   ▼
┌──────────────────────────────────────────┐
│ Ensemble of Models + Simplified Model    │
└──────────────────┬───────────────────────┘
                   ▼
┌──────────────────────────────────────────┐
│ Comprehensive Visualization              │
│ - Confusion matrices                     │
│ - Learning curves                        │
│ - ROC curves                             │
│ - Feature importance                     │
│ - Electrode maps                         │
│ - Attention weights                      │
└──────────────────────────────────────────┘
```

---
### Technologies Used

- **PyTorch:** Deep learning framework for building and training neural network models
- **NumPy/Pandas:** Data manipulation and preprocessing
- **Scikit-learn:** Feature selection, metrics evaluation, and cross-validation
- **Matplotlib/Seaborn:** Visualization of model performance and feature importance
- **tqdm:** Progress tracking during training

**Advanced concepts implemented:**
- Graph Neural Networks for spatial relationships
- Attention mechanisms for feature weighting
- Ensemble learning for robust predictions
- Comprehensive learning dynamics tracking
- EEG-specific data augmentation techniques
- Topological EEG channel mapping

---
### Limitations and Future Work

Current limitations:
- Small dataset size (40 samples) leading to potential overfitting despite regularization
- High model complexity relative to dataset size
- Overfitting ratios in several folds suggest the model may memorize training data
- Limited hyperparameter optimization due to computational constraints

Future extensions:
1. **Advanced signal processing:** Incorporate time-domain features, connectivity measures, or coherence analysis
2. **Additional regularization techniques:** Investigate contrastive learning, mixup, or adversarial training
3. **Model interpretability:** Enhance visualization of neural attention patterns and decision boundaries
4. **Transfer learning:** Leverage pre-trained models from larger EEG datasets
5. **Real-time processing:** Optimize for online classification with reduced computational requirements
6. **Hyperparameter optimization:** Systematic tuning using Bayesian optimization or other advanced methods

These improvements would make the system more robust to new data, more interpretable for neuroscientific insights, and more applicable to real-world clinical settings.

---
### References

1. *Craik, A., He, Y., & Contreras-Vidal, J. L. (2019). Deep learning for electroencephalogram (EEG) classification tasks: a review. Journal of Neural Engineering, 16(3), 031001.*

2. *Lawhern, V. J., Solon, A. J., Waytowich, N. R., Gordon, S. M., Hung, C. P., & Lance, B. J. (2018). EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces. Journal of Neural Engineering, 15(5), 056013.*

3. *Zhang, D., Yao, L., Zhang, X., Wang, S., Chen, W., Boots, R., & Benatallah, B. (2018). Cascade and parallel convolutional recurrent neural networks on EEG-based intention recognition for brain computer interface. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 32, No. 1).*

4. *Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., & Bengio, Y. (2018). Graph attention networks. In International Conference on Learning Representations.*

5. *Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. In Advances in Neural Information Processing Systems (pp. 5998-6008).*

---

## **Task 1: Data Classification using Linear Regression, SVM and KNN algorithms.** [![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/BCImonk/ML4SCI/blob/main/Task%201%20-%20Data%20Classification/Adv%20Classification.ipynb)
---
### Implementation Overview

I've developed a comprehensive machine learning pipeline for EEG data classification that achieves significant accuracy improvements over baseline approaches. Starting with a challenging dataset (40 samples, 320 features), my solution leverages neurophysiological domain knowledge to extract meaningful patterns from brain activity data.

My approach transforms raw EEG signals into functionally relevant features by:
1. Organizing the 64 electrodes by brain regions (frontal, temporal, etc.) and wave types (alpha, beta, delta, theta, gamma)
2. Computing statistical measures and relationship metrics between different brain regions
3. Calculating hemisphere asymmetry and cross-frequency ratios known to be neurologically significant
4. Applying multiple feature selection and dimensionality reduction techniques

To address the small dataset challenge, I implemented a data augmentation strategy that creates synthetic samples while preserving class characteristics. The system evaluates multiple classification algorithms (Logistic Regression, SVM with various kernels, KNN with different distance metrics) and combines the best performers using ensemble learning.

**Results:** SVM with a linear kernel using Recursive Feature Elimination preprocessing achieved the highest accuracy (97.86% on augmented data, 70% on original data).

---

### Technical Differentiators

My implementation extends beyond standard classification techniques:

1. **Domain-specific feature engineering:** Incorporated neurophysiological knowledge by creating region-based features, hemispheric asymmetry measures, and frequency band relationships

2. **Multi-strategy preprocessing:** Systematically evaluated six different preprocessing pipelines (PCA, SelectKBest, mutual information, RFE, power transformation, combinations)

3. **Data augmentation:** Addressed sample size limitation with carefully designed augmentation that maintains statistical properties of the original dataset

4. **Ensemble methodology:** Combined specialized models with different strengths to create a more robust classification system

* *Basic Version of this code:* [![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/BCImonk/ML4SCI/blob/main/Task%201%20-%20Data%20Classification/Basic%20Classification.ipynb)

* *Overfitting Analysis of this Code:* [![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/BCImonk/ML4SCI/blob/main/Task%201%20-%20Data%20Classification/Adv%20Classification%20%2B%20OF%20Check.ipynb)

* *All the Codes:* [![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/BCImonk/ML4SCI/tree/main/Task%201%20-%20Data%20Classification)

---

### Implementation Architecture

```
┌─────────────────────┐
│ Data Loading (n=40) │
└──────────┬──────────┘
           ▼
┌─────────────────────────────────────────┐
│ Feature Engineering                     │
│ ┌─────────┐  ┌─────────┐  ┌─────────┐  │
│ │Raw Stats│  │Regional │  │Entropy  │  │
│ └─────────┘  └─────────┘  └─────────┘  │
│           ▼                            │
│     Combined Features (738)            │
└──────────────────┬──────────────────────┘
                   ▼
┌──────────────────────────────────────────┐
│ Preprocessing Strategies                 │
│ ┌─────┐ ┌────────┐ ┌────┐ ┌────┐ ┌────┐ │
│ │PCA  │ │Select  │ │MI  │ │RFE │ │More│ │
│ └─────┘ └────────┘ └────┘ └────┘ └────┘ │
└──────────────────┬───────────────────────┘
                   ▼
┌──────────────────────────────────────────┐
│    Data Augmentation (140 samples)       │
└──────────────────┬───────────────────────┘
                   ▼
┌──────────────────────────────────────────┐
│ Model Training & Evaluation              │
│ ┌────────┐   ┌────────┐   ┌────────┐    │
│ │LogReg  │   │SVM     │   │KNN     │    │
│ │80.00%  │   │97.86%  │   │92.86%  │    │
│ └────────┘   └────────┘   └────────┘    │
└──────────────────┬───────────────────────┘
                   ▼
┌──────────────────────────────────────────┐
│      Voting Ensemble (97.86%)            │
└──────────────────────────────────────────┘
```

---

### Technologies Used

- **NumPy/Pandas:** Efficient data manipulation and feature engineering
- **Scikit-learn:** Preprocessing, model training, and evaluation
- **SciPy:** Advanced statistical measures
- **Matplotlib:** Results visualization
- **Advanced ML concepts:** Feature selection, ensemble learning, cross-validation, synthetic sampling
- **Domain-specific knowledge:** EEG topography, frequency band analysis, brain asymmetry measures

---

### Limitations and Future Work

Current limitations:
- Performance gap between augmented (97.86%) and original data (70%) suggests potential overfitting
- Feature selection methodology may introduce subtle data leakage
- Limited hyperparameter optimization

Future extensions:
1. **Advanced signal processing:** Time-frequency analysis, connectivity measures, source localization
2. **Deep learning integration:** Specialized architectures for EEG (CNNs, RNNs, GNNs)
3. **Interpretability enhancements:** Visualization tools mapping model decisions to brain regions
4. **Real-time processing:** Optimizing for online classification and incremental learning

These improvements would make the system more robust, interpretable, and applicable to real-world neuroscience and clinical applications.

---

### References

1. *Lotte, F., et al. (2018). A review of classification algorithms for EEG-based brain-computer interfaces: A 10 year update. Journal of Neural Engineering, 15(3).*
2. *Subasi, A. (2007). EEG signal classification using wavelet feature extraction and a mixture of expert model. Expert Systems with Applications, 32(4), 1084-1093.*
3. *Blankertz, B., et al. (2008). Optimizing spatial filters for robust EEG single-trial analysis. IEEE Signal Processing Magazine, 25(1), 41-56.*
4. *Kroupi, E., et al. (2016). EEG-based functional brain networks: does the network size matter? PloS one, 11(8).*
5. *Chawla, N. V., et al. (2002). SMOTE: Synthetic minority over-sampling technique. Journal of Artificial Intelligence Research, 16, 321-357.*
6. *Dietterich, T. G. (2000). Ensemble methods in machine learning. Multiple Classifier Systems, 1857, 1-15.*
7. *Guyon, I., & Elisseeff, A. (2003). An introduction to variable and feature selection. Journal of Machine Learning Research, 3, 1157-1182.*
8. *Hosseini, M. P., et al. (2018). Deep learning with EEG spectrograms in rapid eye movement behavior disorder. Biomedical Engineering Online, 17(1), 116.*
---

## **Task 2: Feature Classification using UFS, RFE and PCA.** [![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/BCImonk/ML4SCI/blob/main/Task%202%20-%20Feature%20Selection/Adv%20Feature.ipynb)
---
### Implementation Overview
I've developed a comprehensive EEG analysis pipeline that tackles the problem of identifying the most discriminative features for classifying different metabolic states from brain activity. My approach goes beyond traditional methods by incorporating neuroanatomical knowledge and comparing multiple feature selection paradigms.

I built this analysis pipeline with a focus on brain-aware feature engineering:

- First, I organized the EEG data by grouping the 64 electrodes into anatomically meaningful regions (frontal, central, parietal, temporal, and occipital).
- I engineered 100 new topological features that capture regional patterns across different frequency bands (alpha, beta, delta, theta, gamma), calculating statistics like mean, standard deviation, maximum, and minimum values for each region-band combination.
- I implemented three fundamentally different feature selection methods to find the top 5 most discriminative features:
  * Univariate Feature Selection (UFS) using mutual information
  * Recursive Feature Elimination (RFE) with RandomForest
  * Principal Component Analysis (PCA) with feature importance extraction
- I evaluated each feature subset with three different classifiers (GradientBoosting, RandomForest, SVM) using 5-fold cross-validation.
- Finally, I analyzed common features across methods to identify the most robust biomarkers.

The results show that UFS with GradientBoosting achieves the highest accuracy (75%), with delta wave features in the temporal region appearing consistently important across methods.

---
### Technical Differentiators

My implementation improves upon conventional EEG analysis methods in several key ways:

- **Neuroanatomical awareness**: Instead of treating EEG channels as independent variables, I incorporated brain topology to create region-specific features, recognizing the spatial relationships between neighboring electrodes.
- **Multi-paradigm comparison**: I systematically compared filter-based (UFS), wrapper-based (RFE), and transformation-based (PCA) feature selection approaches, providing a more comprehensive evaluation than single-method analyses.
- **Advanced feature engineering**: By creating derived features based on regional aggregation, I captured higher-level patterns that simple channel-based approaches would miss.
- **Methodological triangulation**: By examining features selected by multiple methods, I identified signals that are consistently important regardless of the selection approach.

This contrasts with simpler approaches that often use raw channel data with a single feature selection method and minimal feature engineering.

* *A Basic Version of this code:* [![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/BCImonk/ML4SCI/blob/main/Task%202%20-%20Feature%20Selection/Basic%20Feature.ipynb)

* *Directory with all the codes:* [![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/BCImonk/ML4SCI/tree/main/Task%202%20-%20Feature%20Selection)

---
### Implementation Architecture

My implementation follows a structured pipeline:

1. **Data Preprocessing**:
   - Load EEG data and separate features from target labels
   - Group electrodes by brain region
   - Extract frequency band information

2. **Feature Engineering**:
   - Generate regional statistics for each frequency band
   - Combine original features with derived features
   - Standardize all features

3. **Parallel Feature Selection**:
   - UFS: Evaluate each feature independently using mutual information
   - RFE: Iteratively eliminate features using RandomForest importance
   - PCA: Transform data and identify influential original features

4. **Model Evaluation**:
   - Train and evaluate three classifier types on each feature subset
   - Use stratified 5-fold cross-validation for reliable performance estimates
   - Identify best-performing feature selection and classifier combination

5. **Results Analysis**:
   - Compare performance across methods
   - Analyze feature importance for each method
   - Identify common features across selection techniques

This architecture allows for comprehensive evaluation and comparison while maintaining neuroanatomical context.

---
### Technologies Used


I utilized several powerful libraries and techniques in this implementation:

**Libraries**:
- **pandas & numpy**: For efficient data manipulation and numerical operations
- **scikit-learn**: For machine learning components including feature selection, classifiers, and evaluation metrics
- **matplotlib & seaborn**: For creating visualizations of results
- **LinearSegmentedColormap**: For custom visualization gradients

**Key Methods**:
- **Information-theoretic feature selection**: Using mutual information to measure feature relevance without assuming linear relationships
- **Ensemble learning**: Leveraging tree-based ensembles for both feature selection and classification
- **Cross-validation**: Employing stratified k-fold to ensure reliable performance estimates despite the limited sample size
- **Dimensionality reduction**: Using PCA for feature transformation and extraction
- **Regional aggregation**: Creating summary statistics by brain region to capture spatial patterns
- **Multi-classifier evaluation**: Testing feature subsets with different classification paradigms

---
### Limitations and Future Development Opportunities

While my current implementation provides valuable insights, I recognize several limitations and opportunities for enhancement:

**Current Limitations**:
- The analysis is based on a relatively small dataset (40 samples), limiting generalizability
- The fixed selection of exactly 5 features may not be optimal for all scenarios
- Evaluation relies solely on accuracy without considering other metrics like precision or recall
- The current implementation does not explore hyperparameter optimization
- Time-domain dynamics and inter-regional connectivity are not explicitly modeled

**Future Development Opportunities**:
- **Enhanced connectivity analysis**: Implementing measures of functional connectivity between brain regions (coherence, phase lag index, etc.)
- **Advanced feature engineering**: Including entropy measures, complexity indices, and time-frequency representations
- **Optimization framework**: Developing automatic hyperparameter tuning and optimal feature count determination
- **Robust evaluation**: Implementing multi-metric evaluation and statistical significance testing
- **Neurophysiological visualization**: Creating topographical brain maps to visualize important features
- **Deep learning integration**: Exploring CNN or RNN architectures specifically designed for EEG signal processing
- **Transfer learning**: Leveraging pre-trained models from larger EEG datasets

As part of my GSoC contribution, I would be particularly interested in focusing on implementing the connectivity analysis and advanced feature engineering components, as these would significantly enhance the neurophysiological validity of the approach.

---
### References

My implementation builds upon established research in both machine learning and neuroscience:

1. *Lotte, F., et al. (2018). "A review of classification algorithms for EEG-based brain-computer interfaces: a 10 year update." Journal of Neural Engineering, 15(3).*
2. *Guyon, I., & Elisseeff, A. (2003). "An introduction to variable and feature selection." Journal of Machine Learning Research, 3, 1157-1182.*
3. *Subasi, A. (2007). "EEG signal classification using wavelet feature extraction and a mixture of expert model." Expert Systems with Applications, 32(4), 1084-1093.*
4. *Pedregosa, F., et al. (2011). "Scikit-learn: Machine learning in Python." Journal of Machine Learning Research, 12, 2825-2830.*
5. *Breiman, L. (2001). "Random forests." Machine Learning, 45(1), 5-32.*
6. *Friedman, J. H. (2001). "Greedy function approximation: a gradient boosting machine." Annals of Statistics, 29(5), 1189-1232.*
7. *Cohen, M. X. (2014). "Analyzing Neural Time Series Data: Theory and Practice." MIT Press.*
8. *Makeig, S., et al. (2004). "Mining event-related brain dynamics." Trends in Cognitive Sciences, 8(5), 204-210.*

---


