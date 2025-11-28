#   IA&DI subproject 4 - Supervised Flow Classification

## Summary

This project evaluates **1,029,447 network flows** across multiple application protocols to compare machine learning classifiers. Among the tested models, **k-Nearest Neighbors (k-NN)** achieved the best performance with an average **F1-score of 0.7341**, significantly outperforming **Naive Bayes**, which scored **0.3373**.

## 0. Project Overview & Methodology

### 0.1 System Architecture

**Installation**
Install the required Python packages using:

```bash
pip install -r requirements.txt
```

[requirements.txt](requirements.txt)

### Phase 1: Data Extraction & Preprocessing

* **Input**: Raw network packet captures (PCAP files)
* **Processing**: [extract.py](extract.py) converts PCAP files into structured CSV format
* **Output**: Flow-based features, including timing, packet statistics, and protocol information

### Phase 2: Feature Engineering & Labeling

* **Script**: [traitement.py](traitement.py) processes the extracted flows
* **Key Operations**:

  * `load_all_csv_flows()`: Aggregates all CSV files into a unified dataset
  * `add_fan_in_fan_out_features()`: Adds metrics describing network behavior
  * `load_ground_truth()` & `label_flows()`: Assigns attack labels using ground truth data

### Phase 3: Machine Learning Pipeline

* `vectorize_flows()`: Converts flows into numerical feature vectors
* `prepare_cv_splits()`: Creates stratified cross-validation splits by application protocol
* **Classification**: Applies k-NN and Naive Bayes algorithms
* **Evaluation**: Generates confusion matrices and computes performance metrics

### Output

* [output.txt](output.txt) — Detailed classification results
* Confusion matrices by application protocol:
  ![confusion\_matrices\_by\_application.png](confusion_matrices_by_application.png)
* Model performance comparison:
  ![model\_comparison.png](model_comparison.png)


## 1. Experimental Setup & Methodology

### 1.1 Dataset Characteristics
- **Total Flows Processed**: 1,029,447 network flows
- **Attack Flows**: 7,944 (0.77% of total) - **Highly Imbalanced Dataset**
- **Normal Flows**: 1,021,503 (99.23% of total)

### 1.2 Feature Engineering
**18 Features Used for Classification:**
- Basic Flow Metrics: `duration`, `packets`, `bytes`, `bytes_rev`
- Network Behavior: `fan_out_src`, `fan_in_dst`
- Port Information: `port_src`, `port_dst`
- Packet Statistics: `packets_src2dst`, `packets_dst2src`
- Statistical Features: `mean_ps`, `stddev_ps`, `mean_piat_ms`, `stddev_piat_ms`
- Protocol Flags: `syn_packets`, `ack_packets`, `fin_packets`, `rst_packets`

### 1.3 Evaluation Methodology
- **Stratified 5-Fold Cross-Validation** per application protocol
- **80/20 Train-Test Split** with stratification
- **Metrics**: Accuracy, Precision, Recall, F1-Score
- **Protocols Analyzed**: HTTP, IMAP, DNS, SMTP, ICMP, SSH

## 2. Protocol-Specific Performance Analysis

### 2.1 HTTP Traffic
**Class Distribution**: 2.6% attack traffic
- **k-NN Performance**: 
  - Accuracy: 99.07%
  - F1-Score: 81.69%
  - Excellent detection with high precision (82-83%) and recall (78-85%)
- **Naive Bayes**: 
  - Complete failure (F1 = 0%) - unable to detect attacks
  - Always predicts "Normal" due to class imbalance

### 2.2 IMAP Traffic
**Class Distribution**: 53.4% attack traffic - **Balanced Dataset**
- **Both Models**: Perfect performance (100% across all metrics)
- **k-NN & Naive Bayes**: F1-Score = 100%
- **Conclusion**: IMAP attack patterns are easily distinguishable

### 2.3 DNS Traffic
**Class Distribution**: 0.04% attack traffic - **Extremely Imbalanced**
- **k-NN**: Outstanding performance
  - Accuracy: 99.99%
  - F1-Score: 85.71%
  - High precision (92-100%) and recall (83-100%)
- **Naive Bayes**: Moderate performance
  - F1-Score: 64.71%
  - Lower precision (41-67%) affects overall performance

### 2.4 SMTP Traffic
**Class Distribution**: 1.7% attack traffic
- **k-NN**: Near-perfect detection
  - Accuracy: 99.99%
  - F1-Score: 99.67%
  - Consistent high performance across folds
- **Naive Bayes**: Poor performance
  - F1-Score: 3.92%
  - Very low recall (0.8-1.7%) indicates missed attacks

### 2.5 ICMP Traffic
**Critical Issue**: No attack samples in training or test sets
- Both models achieve 100% accuracy by always predicting "Normal"
- F1-Score: 0% - **Cannot evaluate true performance**
- **Dataset limitation**: Missing ground truth labels for ICMP attacks

### 2.6 SSH Traffic
**Class Distribution**: 0.02% attack traffic - **Extremely Sparse**
- Both models achieve ~99.98% accuracy
- F1-Score: 0% - Insufficient attack samples for meaningful detection

## 3. Model Comparison & Ranking

### 3.1 Overall Performance Summary
| Model | Average Accuracy | Average F1-Score | Ranking |
|-------|------------------|------------------|---------|
| **k-NN** | **99.80%** | **73.41%** | **1st** |
| Naive Bayes | 99.14% | 33.73% | 2nd |

### 3.2 Key Findings

**k-NN Advantages:**
- Robust to class imbalance
- Excellent detection across multiple protocols
- Consistent high F1-scores
- Better generalization to rare attack patterns

**Naive Bayes Limitations:**
- Severely affected by class imbalance
- Poor performance on HTTP and SMTP protocols
- Tends to predict majority class (Normal)
- Lower recall rates across most protocols

## 8. Conclusion

The  evaluation demonstrates that **k-NN is the optimal choice** for supervised learning-based IDS implementation. Its robustness to class imbalance and consistent performance across multiple protocols make it suitable for real-world deployment. However, the success of any IDS depends on addressing the identified dataset limitations and implementing a protocol-aware detection strategy.
