import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime, timedelta
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')

# NFStream for flow extraction
from nfstream import NFStreamer

# Sklearn for classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report, roc_auc_score)

import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================================
# Q1: Extract flows from pcap files using NFStream
# ============================================================================

def extract_flows_from_pcaps(pcap_dir, idle_timeout=60, active_timeout=120):
    """
    Extract flows from all pcap files in a directory using NFStream
    """
    print("=" * 80)
    print("Q1: Extracting flows from pcap files using NFStream")
    print("=" * 80)

    pcap_files = glob.glob(os.path.join(pcap_dir, "*.pcap")) + \
                 glob.glob(os.path.join(pcap_dir, "*.pcapng"))

    flows_list = []

    for pcap_file in pcap_files:
        print(f"\nProcessing: {os.path.basename(pcap_file)}")
        try:
            streamer = NFStreamer(
                source=pcap_file,
                idle_timeout=idle_timeout,
                active_timeout=active_timeout,
                statistical_analysis=True
            )

            for flow in streamer:
                flow_dict = {
                    'first_timestamp': flow.bidirectional_first_seen_ms / 1000,
                    'last_timestamp': flow.bidirectional_last_seen_ms / 1000,
                    'ip_src': flow.src_ip,
                    'ip_dst': flow.dst_ip,
                    'port_src': flow.src_port,
                    'port_dst': flow.dst_port,
                    'protocol': flow.protocol,
                    'application_name': flow.application_name,
                    'duration': flow.bidirectional_duration_ms / 1000,
                    'packets': flow.bidirectional_packets,
                    'bytes': flow.bidirectional_bytes,
                    'bytes_rev': flow.dst2src_bytes,
                    'packets_src2dst': flow.src2dst_packets,
                    'packets_dst2src': flow.dst2src_packets,
                    'bytes_src2dst': flow.src2dst_bytes,
                    'bytes_dst2src': flow.dst2src_bytes,
                    'mean_ps': flow.bidirectional_mean_ps,
                    'stddev_ps': flow.bidirectional_stddev_ps,
                    'mean_piat_ms': flow.bidirectional_mean_piat_ms,
                    'stddev_piat_ms': flow.bidirectional_stddev_piat_ms,
                    'syn_packets': flow.bidirectional_syn_packets,
                    'ack_packets': flow.bidirectional_ack_packets,
                    'fin_packets': flow.bidirectional_fin_packets,
                    'rst_packets': flow.bidirectional_rst_packets,
                    'psh_packets': flow.bidirectional_psh_packets,
                }

                flows_list.append(flow_dict)

        except Exception as e:
            print(f"Error processing {pcap_file}: {e}")

    flows_df = pd.DataFrame(flows_list)
    print(f"\nTotal flows extracted: {len(flows_df)}")
    return flows_df


# ============================================================================
# Q1 (continued): Add fan-in and fan-out features
# ============================================================================

def add_fan_in_fan_out_features(flows_df, time_window=300):
    """
    Add fan-in and fan-out features based on temporal sliding window
    """
    print("\nAdding fan-in and fan-out features...")

    flows_df = flows_df.sort_values('first_timestamp').reset_index(drop=True)

    # Initialize features
    flows_df['fan_out_src'] = 0
    flows_df['fan_in_dst'] = 0

    for idx, row in flows_df.iterrows():
        current_time = row['first_timestamp']
        window_start = current_time - time_window

        # Fan-out: unique IPs that src_ip connects to within time window
        mask_fanout = (flows_df['first_timestamp'] >= window_start) & \
                      (flows_df['first_timestamp'] <= current_time) & \
                      (flows_df['ip_src'] == row['ip_src'])
        flows_df.at[idx, 'fan_out_src'] = flows_df.loc[mask_fanout, 'ip_dst'].nunique()

        # Fan-in: unique IPs that connect to dst_ip within time window
        mask_fanin = (flows_df['first_timestamp'] >= window_start) & \
                     (flows_df['first_timestamp'] <= current_time) & \
                     (flows_df['ip_dst'] == row['ip_dst'])
        flows_df.at[idx, 'fan_in_dst'] = flows_df.loc[mask_fanin, 'ip_src'].nunique()

        if (idx + 1) % 1000 == 0:
            print(f"Processed {idx + 1} flows")

    print("Fan-in and fan-out features added")
    return flows_df


# ============================================================================
# Q2: Load ground truth and label flows
# ============================================================================

def load_ground_truth(gt_file):
    """
    Load ground truth file with attack flows
    """
    print("\nLoading ground truth...")
    gt_df = pd.read_csv(gt_file, sep='\t', header=None)
    gt_df.columns = ['first_timestamp', 'last_timestamp', 'ip_src', 'ip_dst',
                     'port_src', 'port_dst', 'protocol']
    print(f"Ground truth flows: {len(gt_df)}")
    return gt_df


def label_flows(flows_df, gt_df, tolerance=5):
    """
    Label flows using ground truth with time tolerance
    Efficient implementation using set-based matching
    """
    print("\nLabeling flows...")

    flows_df['label'] = 0  # 0 = normal, 1 = attack

    # Create set of ground truth flows for fast lookup
    gt_set = set()
    for _, row in gt_df.iterrows():
        key = (row['ip_src'], row['ip_dst'], row['port_src'], row['port_dst'], int(row['protocol']))
        gt_set.add(key)

    # Label flows
    for idx, row in flows_df.iterrows():
        key = (row['ip_src'], row['ip_dst'], row['port_src'], row['port_dst'], row['protocol'])
        if key in gt_set:
            flows_df.at[idx, 'label'] = 1

    attack_count = (flows_df['label'] == 1).sum()
    normal_count = (flows_df['label'] == 0).sum()
    print(f"Attack flows: {attack_count} ({100 * attack_count / len(flows_df):.2f}%)")
    print(f"Normal flows: {normal_count} ({100 * normal_count / len(flows_df):.2f}%)")

    return flows_df


# ============================================================================
# Q3: Cross-validation setup
# ============================================================================

def prepare_cv_splits(flows_df, test_size=0.2, n_splits=5):
    """
    Prepare cross-validation splits stratified by application name
    """
    print("\n" + "=" * 80)
    print("Q3: Setting up cross-validation by application")
    print("=" * 80)

    app_names = ['HTTP', 'IMAP', 'DNS', 'SMTP', 'ICMP', 'SSH', 'FTP']
    cv_tasks = {}

    for app in app_names:
        app_flows = flows_df[flows_df['application_name'] == app].copy()

        if len(app_flows) < 2:
            print(f"Skipping {app} (insufficient flows: {len(app_flows)})")
            continue

        print(f"\n{app}: {len(app_flows)} flows")

        # Initial 80/20 split
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            app_flows, app_flows['label'],
            test_size=test_size,
            stratify=app_flows['label'],
            random_state=42
        )

        print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
        print(f"  Train attacks: {y_train.sum()}, Test attacks: {y_test.sum()}")

        # 5-fold stratified split on training set
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        splits = list(skf.split(X_train, y_train))

        cv_tasks[app] = {
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test,
            'splits': splits
        }

    return cv_tasks


# ============================================================================
# Vectorization and preprocessing
# ============================================================================

def vectorize_flows(flows_df, numeric_features=None):
    """
    Convert flow data to numerical vectors
    """
    if numeric_features is None:
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
    X = flows_df[available_features].values

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, available_features, scaler


# ============================================================================
# Classification with k-NN
# ============================================================================

def classify_knn(X_train, y_train, X_test, y_test, k=5):
    """
    k-NN classifier
    """
    knn = KNeighborsClassifier(n_neighbors=k)
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


# ============================================================================
# Classification with Naive Bayes
# ============================================================================

def classify_naive_bayes(X_train, y_train, X_test, y_test):
    """
    Multinomial Naive Bayes classifier (requires non-negative features)
    """
    # Scale to non-negative range for Multinomial NB
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
# Main execution
# ============================================================================

def main():
    # Configuration
    PCAP_DIR = "./TRAIN"  # Directory with pcap files
    GT_FILE = "./TRAIN/TRAIN.gt"  # Ground truth file

    # Q1: Extract flows
    flows_df = extract_flows_from_pcaps(PCAP_DIR)
    flows_df = add_fan_in_fan_out_features(flows_df, time_window=300)

    # Q2: Load ground truth and label flows
    gt_df = load_ground_truth(GT_FILE)
    flows_df = label_flows(flows_df, gt_df)

    # Vectorize flows
    X, features, scaler = vectorize_flows(flows_df)

    # Q3: Prepare cross-validation splits
    cv_tasks = prepare_cv_splits(flows_df)

    # ========================================================================
    # Q3 & Q4: Run classifications
    # ========================================================================

    results_knn = {}
    results_nb = {}

    for app, task_data in cv_tasks.items():
        print(f"\n{'=' * 80}")
        print(f"Processing: {app}")
        print(f"{'=' * 80}")

        X_train_full = X[task_data['X_train'].index]
        y_train_full = task_data['y_train'].values
        X_test_final = X[task_data['X_test'].index]
        y_test_final = task_data['y_test'].values

        app_results_knn = []
        app_results_nb = []

        # 5-fold cross-validation
        for fold, (train_idx, val_idx) in enumerate(task_data['splits'], 1):
            X_train = X_train_full[train_idx]
            y_train = y_train_full[train_idx]
            X_val = X_train_full[val_idx]
            y_val = y_train_full[val_idx]

            print(f"\nFold {fold}:")

            # k-NN
            knn_result = classify_knn(X_train, y_train, X_val, y_val, k=5)
            app_results_knn.append(knn_result)
            print(f"  k-NN - Acc: {knn_result['accuracy']:.4f}, "
                  f"Prec: {knn_result['precision']:.4f}, "
                  f"Rec: {knn_result['recall']:.4f}, "
                  f"F1: {knn_result['f1']:.4f}")

            # Naive Bayes
            nb_result = classify_naive_bayes(X_train, y_train, X_val, y_val)
            app_results_nb.append(nb_result)
            print(f"  NB  - Acc: {nb_result['accuracy']:.4f}, "
                  f"Prec: {nb_result['precision']:.4f}, "
                  f"Rec: {nb_result['recall']:.4f}, "
                  f"F1: {nb_result['f1']:.4f}")

        results_knn[app] = app_results_knn
        results_nb[app] = app_results_nb

        # Final test set evaluation
        print(f"\nFinal Test Set ({len(X_test_final)} samples):")
        knn_test = classify_knn(X_train_full, y_train_full, X_test_final, y_test_final)
        nb_test = classify_naive_bayes(X_train_full, y_train_full, X_test_final, y_test_final)

        print(f"  k-NN - Acc: {knn_test['accuracy']:.4f}, F1: {knn_test['f1']:.4f}")
        print(f"  NB   - Acc: {nb_test['accuracy']:.4f}, F1: {nb_test['f1']:.4f}")

    # ========================================================================
    # Summary and comments
    # ========================================================================

    print("\n" + "=" * 80)
    print("SUMMARY AND COMMENTS")
    print("=" * 80)

    print("\nk-NN Classifier Results:")
    for app, results in results_knn.items():
        accs = [r['accuracy'] for r in results]
        print(f"  {app}: Mean Accuracy = {np.mean(accs):.4f} ± {np.std(accs):.4f}")

    print("\nNaive Bayes Classifier Results:")
    for app, results in results_nb.items():
        accs = [r['accuracy'] for r in results]
        print(f"  {app}: Mean Accuracy = {np.mean(accs):.4f} ± {np.std(accs):.4f}")


if __name__ == "__main__":
    main()