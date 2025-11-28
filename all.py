import pandas as pd
import numpy as np
import os
import glob
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# LOAD ALL CSV FILES INTO ONE DATAFRAME
# ============================================================================

def load_all_csv_flows(csv_dir):
    """
    Load all CSV files from extraction directory into one dataframe
    """
    print("=" * 80)
    print("LOADING EXTRACTED FLOWS FROM CSV")
    print("=" * 80)

    csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))

    if not csv_files:
        print("No CSV files found!")
        return None

    print(f"Found {len(csv_files)} CSV files")

    all_flows = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            all_flows.append(df)
            print(f"  Loaded {os.path.basename(csv_file)}: {len(df)} flows")
        except Exception as e:
            print(f"  Error loading {csv_file}: {e}")

    flows_df = pd.concat(all_flows, ignore_index=True)
    print(f"\nTotal flows loaded: {len(flows_df)}")
    return flows_df


# ============================================================================
# ADD FAN-IN AND FAN-OUT FEATURES
# ============================================================================

def add_fan_in_fan_out_features(flows_df, time_window=300):
    """
    Add fan-in and fan-out features using fast vectorized approach
    """
    print("\nAdding fan-in and fan-out features (vectorized)...")
    print(flows_df)
    flows_df = flows_df.sort_values('first_timestamp').reset_index(drop=True)

    # Fast approach: compute based on global connection patterns
    # Fan-out: how many unique destinations each source connects to
    fan_out_src = flows_df.groupby('ip_src')['ip_dst'].nunique().to_dict()
    flows_df['fan_out_src'] = flows_df['ip_src'].map(fan_out_src).fillna(0).astype(int)

    # Fan-in: how many unique sources connect to each destination
    fan_in_dst = flows_df.groupby('ip_dst')['ip_src'].nunique().to_dict()
    flows_df['fan_in_dst'] = flows_df['ip_dst'].map(fan_in_dst).fillna(0).astype(int)

    print("Fan-in and fan-out features added (FAST!)")
    return flows_df


# ============================================================================
# LOAD GROUND TRUTH AND LABEL FLOWS
# ============================================================================

def load_ground_truth(gt_file):
    """
    Load ground truth file with proper debugging
    """
    print("\nLoading ground truth...")
    print(f"File: {gt_file}")

    # Try different delimiters
    delimiters = ['\t', '\s+', ',', ' ']
    gt_df = None

    for delimiter in delimiters:
        try:
            gt_df = pd.read_csv(gt_file, sep=delimiter, header=None, engine='python')
            print(f"  ✓ Successfully read with delimiter: {repr(delimiter)}")
            print(f"  Shape: {gt_df.shape}")
            print(f"  First row: {gt_df.iloc[0].tolist()}")
            break
        except Exception as e:
            print(f"  ✗ Failed with delimiter {repr(delimiter)}: {e}")
            continue

    if gt_df is None:
        raise ValueError("Could not read ground truth file with any delimiter")

    # Check if first row is header (contains strings like 'dst_ip')
    first_row_str = str(gt_df.iloc[0, 0])
    print(f"  First cell value: {first_row_str}")

    if not first_row_str.replace('.', '').replace('-', '').isdigit():
        print("  → Detected header row, skipping...")
        gt_df = gt_df.iloc[1:].reset_index(drop=True)

    # Select first 7 columns
    if len(gt_df.columns) >= 7:
        gt_df = gt_df.iloc[:, :7]

    gt_df.columns = ['first_timestamp', 'last_timestamp', 'ip_src', 'ip_dst',
                     'port_src', 'port_dst', 'protocol']

    print(f"  ✓ Ground truth flows: {len(gt_df)}")
    print(f"  Sample row:\n{gt_df.iloc[0]}")

    return gt_df


def label_flows(flows_df, gt_df):
    """
    Label flows using ground truth (fast set-based matching)
    """
    print("\nLabeling flows with ground truth...")

    flows_df['label'] = 0

    # Create set for fast lookup
    gt_set = set()
    for _, row in gt_df.iterrows():
        key = (row['ip_src'], row['ip_dst'], int(row['port_src']), int(row['port_dst']), int(row['protocol']))
        gt_set.add(key)

    # Label flows
    for idx, row in flows_df.iterrows():
        key = (row['ip_src'], row['ip_dst'], row['port_src'], row['port_dst'], row['protocol'])
        if key in gt_set:
            flows_df.at[idx, 'label'] = 1

        if (idx + 1) % 5000 == 0:
            print(f"  Labeled {idx + 1} flows")

    attack_count = (flows_df['label'] == 1).sum()
    normal_count = (flows_df['label'] == 0).sum()
    print(f"\nAttack flows: {attack_count} ({100 * attack_count / len(flows_df):.2f}%)")
    print(f"Normal flows: {normal_count} ({100 * normal_count / len(flows_df):.2f}%)")

    return flows_df


# ============================================================================
# VECTORIZATION
# ============================================================================

def vectorize_flows(flows_df):
    """
    Convert flows to feature vectors
    """
    numeric_features = ['duration', 'packets', 'bytes', 'bytes_rev',
                        'fan_out_src', 'fan_in_dst', 'port_src', 'port_dst',
                        'packets_src2dst', 'packets_dst2src', 'mean_ps',
                        'stddev_ps', 'mean_piat_ms', 'stddev_piat_ms',
                        'syn_packets', 'ack_packets', 'fin_packets', 'rst_packets']

    # Fill missing values
    for feat in numeric_features:
        if feat in flows_df.columns:
            flows_df[feat] = flows_df[feat].fillna(0)

    # Select available features
    available_features = [f for f in numeric_features if f in flows_df.columns]
    X = flows_df[available_features].values.astype(np.float32)

    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, available_features


# ============================================================================
# CROSS-VALIDATION SETUP
# ============================================================================

def prepare_cv_splits(flows_df, X_scaled, test_size=0.2, n_splits=5):
    """
    Prepare stratified CV splits by application
    """
    print("\n" + "=" * 80)
    print("SETTING UP CROSS-VALIDATION BY APPLICATION")
    print("=" * 80)

    app_names = ['HTTP', 'IMAP', 'DNS', 'SMTP', 'ICMP', 'SSH', 'FTP']
    cv_tasks = {}

    for app in app_names:
        app_mask = flows_df['application_name'] == app
        app_indices = np.where(app_mask)[0]

        if len(app_indices) < 2:
            print(f"Skipping {app} (insufficient flows: {len(app_indices)})")
            continue

        print(f"\n{app}: {len(app_indices)} flows")

        y_app = flows_df.loc[app_indices, 'label'].values
        X_app = X_scaled[app_indices]

        # 80/20 split
        indices = np.arange(len(app_indices))
        train_idx, test_idx = train_test_split(
            indices, test_size=test_size,
            stratify=y_app, random_state=42
        )

        train_indices = app_indices[train_idx]
        test_indices = app_indices[test_idx]

        print(f"  Train: {len(train_indices)}, Test: {len(test_indices)}")
        print(f"  Train attacks: {y_app[train_idx].sum()}, Test attacks: {y_app[test_idx].sum()}")

        # 5-fold CV on training set
        X_train = X_scaled[train_indices]
        y_train = flows_df.loc[train_indices, 'label'].values

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        splits = list(skf.split(X_train, y_train))

        cv_tasks[app] = {
            'train_indices': train_indices,
            'test_indices': test_indices,
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_scaled[test_indices],
            'y_test': flows_df.loc[test_indices, 'label'].values,
            'splits': splits
        }

    return cv_tasks


# ============================================================================
# CLASSIFIERS
# ============================================================================

def classify_knn(X_train, y_train, X_test, y_test, k=5):
    """k-NN classifier"""
    knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'y_pred': y_pred,
        'cm': confusion_matrix(y_test, y_pred)
    }


def classify_naive_bayes(X_train, y_train, X_test, y_test):
    """Multinomial Naive Bayes classifier"""
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    nb = MultinomialNB()
    nb.fit(X_train_scaled, y_train)
    y_pred = nb.predict(X_test_scaled)

    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'y_pred': y_pred,
        'cm': confusion_matrix(y_test, y_pred)
    }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    CSV_DIR = "./extracted_flows"
    GT_FILE = "./TRAIN/TRAIN.gt"

    # Load flows from CSV
    flows_df = load_all_csv_flows(CSV_DIR)
    if flows_df is None or len(flows_df) == 0:
        print("No flows to process!")
        return

    # Add fan-in/fan-out
    flows_df = add_fan_in_fan_out_features(flows_df, time_window=300)

    # Load ground truth and label
    gt_df = load_ground_truth(GT_FILE)
    flows_df = label_flows(flows_df, gt_df)

    # Vectorize
    X_scaled, features = vectorize_flows(flows_df)

    # Prepare CV splits
    cv_tasks = prepare_cv_splits(flows_df, X_scaled)

    # ========================================================================
    # RUN CLASSIFICATIONS
    # ========================================================================

    results_knn = {}
    results_nb = {}

    for app, task_data in cv_tasks.items():
        print(f"\n{'=' * 80}")
        print(f"CLASSIFYING: {app}")
        print(f"{'=' * 80}")

        X_train = task_data['X_train']
        y_train = task_data['y_train']
        X_test = task_data['X_test']
        y_test = task_data['y_test']

        app_results_knn = []
        app_results_nb = []

        # 5-fold cross-validation
        for fold, (train_idx, val_idx) in enumerate(task_data['splits'], 1):
            X_fold_train = X_train[train_idx]
            y_fold_train = y_train[train_idx]
            X_fold_val = X_train[val_idx]
            y_fold_val = y_train[val_idx]

            print(f"\nFold {fold}:")

            # k-NN
            knn_result = classify_knn(X_fold_train, y_fold_train, X_fold_val, y_fold_val, k=5)
            app_results_knn.append(knn_result)
            print(f"  k-NN - Acc: {knn_result['accuracy']:.4f}, Prec: {knn_result['precision']:.4f}, "
                  f"Rec: {knn_result['recall']:.4f}, F1: {knn_result['f1']:.4f}")

            # Naive Bayes
            nb_result = classify_naive_bayes(X_fold_train, y_fold_train, X_fold_val, y_fold_val)
            app_results_nb.append(nb_result)
            print(f"  NB  - Acc: {nb_result['accuracy']:.4f}, Prec: {nb_result['precision']:.4f}, "
                  f"Rec: {nb_result['recall']:.4f}, F1: {nb_result['f1']:.4f}")

        results_knn[app] = app_results_knn
        results_nb[app] = app_results_nb

        # Final test evaluation
        print(f"\nTest Set ({len(X_test)} samples):")
        knn_test = classify_knn(X_train, y_train, X_test, y_test)
        nb_test = classify_naive_bayes(X_train, y_train, X_test, y_test)

        print(f"  k-NN - Acc: {knn_test['accuracy']:.4f}, F1: {knn_test['f1']:.4f}")
        print(f"  NB   - Acc: {nb_test['accuracy']:.4f}, F1: {nb_test['f1']:.4f}")

    # ========================================================================
    # SUMMARY
    # ========================================================================

    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    print("\nk-NN Results:")
    for app, results in results_knn.items():
        accs = [r['accuracy'] for r in results]
        f1s = [r['f1'] for r in results]
        print(
            f"  {app:6s} - Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}, F1: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

    print("\nNaive Bayes Results:")
    for app, results in results_nb.items():
        accs = [r['accuracy'] for r in results]
        f1s = [r['f1'] for r in results]
        print(
            f"  {app:6s} - Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}, F1: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")


if __name__ == "__main__":
    main()