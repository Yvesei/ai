import pandas as pd
import numpy as np
import os
import glob
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report,
                             ConfusionMatrixDisplay)  
import matplotlib.pyplot as plt  
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
            if len(all_flows) == 1:
                print(f"    Columns: {df.columns.tolist()}")
        except Exception as e:
            print(f"  Error loading {csv_file}: {e}")

    flows_df = pd.concat(all_flows, ignore_index=True)
    print(f"\nTotal flows loaded: {len(flows_df)}")
    print(f"Columns: {flows_df.columns.tolist()}")
    print(f"\nFirst few rows:")
    print(flows_df.head())
    return flows_df


# ============================================================================
# ADD FAN-IN AND FAN-OUT FEATURES
# ============================================================================

def add_fan_in_fan_out_features(flows_df, time_window=300):
    """
    Add fan-in and fan-out features using fast vectorized approach
    """
    print("\nAdding fan-in and fan-out features (vectorized)...")
    print(f"Available columns: {flows_df.columns.tolist()}")

    # Check if timestamp column exists
    timestamp_col = None
    for col in ['first_timestamp', 'bidirectional_first_seen_ms', 'timestamp']:
        if col in flows_df.columns:
            timestamp_col = col
            break

    if timestamp_col:
        flows_df = flows_df.sort_values(timestamp_col).reset_index(drop=True)
        print(f"  Sorted by {timestamp_col}")
    else:
        print(f"  WARNING: No timestamp column found, skipping sort")

    # Fast approach: compute based on global connection patterns
    # Fan-out: how many unique destinations each source connects to
    fan_out_src = flows_df.groupby('ip_src')['ip_dst'].nunique().to_dict()
    flows_df['fan_out_src'] = flows_df['ip_src'].map(fan_out_src).fillna(0).astype(int)

    # Fan-in: how many unique sources connect to each destination
    fan_in_dst = flows_df.groupby('ip_dst')['ip_src'].nunique().to_dict()
    flows_df['fan_in_dst'] = flows_df['ip_dst'].map(fan_in_dst).fillna(0).astype(int)

    print(" Fan-in and fan-out features added")
    print(f"  fan_out_src range: {flows_df['fan_out_src'].min()} to {flows_df['fan_out_src'].max()}")
    print(f"  fan_in_dst range: {flows_df['fan_in_dst'].min()} to {flows_df['fan_in_dst'].max()}")
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

    # Try with header=0 first (comma-separated CSV with header)
    try:
        gt_df = pd.read_csv(gt_file, sep=',', header=0)
        print(f"   Successfully read with comma delimiter and header=0")
        print(f"  Shape: {gt_df.shape}")
        print(f"  Columns: {gt_df.columns.tolist()}")
        print(f"  First row:\n{gt_df.iloc[0]}")
    except Exception as e:
        print(f"  ✗ Failed with comma + header: {e}")
        # Try different delimiters without header
        delimiters = ['\t', '\s+', ' ']
        gt_df = None

        for delimiter in delimiters:
            try:
                gt_df = pd.read_csv(gt_file, sep=delimiter, header=None, engine='python')
                print(f"   Successfully read with delimiter: {repr(delimiter)}")
                print(f"  Shape: {gt_df.shape}")
                print(f"  First row: {gt_df.iloc[0].tolist()}")
                break
            except Exception as e2:
                print(f"  ✗ Failed with delimiter {repr(delimiter)}: {e2}")
                continue

        if gt_df is None:
            raise ValueError("Could not read ground truth file with any delimiter")

        # Check if first row is header
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

    print(f"   Ground truth flows: {len(gt_df)}")
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
        key = (row['src_ip'], row['dst_ip'], int(row['src_port']), int(row['dst_port']), int(row['protocol']))
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
    print("\nVectorizing flows...")
    print(f"Available columns: {flows_df.columns.tolist()}")

    # Numeric features to use (will select available ones)
    possible_features = ['duration', 'packets', 'bytes', 'bytes_rev',
                         'fan_out_src', 'fan_in_dst', 'port_src', 'port_dst',
                         'packets_src2dst', 'packets_dst2src', 'mean_ps',
                         'stddev_ps', 'mean_piat_ms', 'stddev_piat_ms',
                         'syn_packets', 'ack_packets', 'fin_packets', 'rst_packets',
                         'src2dst_packets', 'dst2src_packets', 'src2dst_bytes', 'dst2src_bytes',
                         'bidirectional_packets', 'bidirectional_bytes', 'bidirectional_duration_ms']

    # Select available features
    available_features = [f for f in possible_features if f in flows_df.columns]
    print(f"Using {len(available_features)} features: {available_features}")

    # Fill missing values
    for feat in available_features:
        flows_df[feat] = flows_df[feat].fillna(0)

    X = flows_df[available_features].values.astype(np.float32)

    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f" Vectorization complete: {X_scaled.shape}")
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
# CLASSIFICATION FUNCTIONS WITH EDGE CASE HANDLING
# ============================================================================

def safe_classify_knn(X_train, y_train, X_test, y_test, k=5):
    """k-NN classifier with edge case handling"""
    # Check if we have both classes in training
    unique_train = np.unique(y_train)
    if len(unique_train) == 1:
        # Only one class in training - predict that class for all test samples
        y_pred = np.full(len(y_test), unique_train[0])
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0) if 1 in y_test else 0
        recall = recall_score(y_test, y_pred, zero_division=0) if 1 in y_test else 0
        f1 = f1_score(y_test, y_pred, zero_division=0) if 1 in y_test else 0
    else:
        knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'y_pred': y_pred,
        'cm': confusion_matrix(y_test, y_pred)
    }


def safe_classify_naive_bayes(X_train, y_train, X_test, y_test):
    """Multinomial Naive Bayes classifier with edge case handling"""
    # Check if we have both classes in training
    unique_train = np.unique(y_train)
    if len(unique_train) == 1:
        # Only one class in training - predict that class for all test samples
        y_pred = np.full(len(y_test), unique_train[0])
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0) if 1 in y_test else 0
        recall = recall_score(y_test, y_pred, zero_division=0) if 1 in y_test else 0
        f1 = f1_score(y_test, y_pred, zero_division=0) if 1 in y_test else 0
    else:
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        nb = MultinomialNB()
        nb.fit(X_train_scaled, y_train)
        y_pred = nb.predict(X_test_scaled)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'y_pred': y_pred,
        'cm': confusion_matrix(y_test, y_pred)
    }


# ============================================================================
# CONFUSION MATRIX PLOTTING
# ============================================================================

def safe_plot_confusion_matrix(y_true, y_pred, ax, title, cmap='Blues'):
    """Safely plot confusion matrix handling edge cases"""
    cm = confusion_matrix(y_true, y_pred)

    # Handle case where we might have only one class
    labels = ['Normal', 'Attack']
    present_labels = np.unique(np.concatenate([y_true, y_pred]))

    # Filter labels to only those present
    display_labels = [labels[i] for i in present_labels]

    # Create the display
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=display_labels)
    disp.plot(cmap=cmap, ax=ax, values_format='d')
    ax.set_title(title)

    return disp


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
    # RUN CLASSIFICATIONS WITH ROBUST HANDLING
    # ========================================================================

    results_knn = {}
    results_nb = {}
    test_results = {}

    # Only create plots if we have applications to plot
    valid_apps = [app for app in cv_tasks.keys() if len(cv_tasks[app]['test_indices']) > 0]

    if valid_apps:
        fig, axes = plt.subplots(2, len(valid_apps), figsize=(6 * len(valid_apps), 10))
        if len(valid_apps) == 1:
            axes = axes.reshape(2, 1)
    else:
        axes = None

    for idx, (app, task_data) in enumerate(cv_tasks.items()):
        print(f"\n{'=' * 80}")
        print(f"CLASSIFYING: {app}")
        print(f"{'=' * 80}")

        X_train = task_data['X_train']
        y_train = task_data['y_train']
        X_test = task_data['X_test']
        y_test = task_data['y_test']

        # Print class distribution for debugging
        unique_train, counts_train = np.unique(y_train, return_counts=True)
        unique_test, counts_test = np.unique(y_test, return_counts=True)
        print(f"Training set - Class distribution: {dict(zip(unique_train, counts_train))}")
        print(f"Test set - Class distribution: {dict(zip(unique_test, counts_test))}")

        app_results_knn = []
        app_results_nb = []

        # 5-fold cross-validation
        for fold, (train_idx, val_idx) in enumerate(task_data['splits'], 1):
            X_fold_train = X_train[train_idx]
            y_fold_train = y_train[train_idx]
            X_fold_val = X_train[val_idx]
            y_fold_val = y_train[val_idx]

            print(f"\nFold {fold}:")
            print(f"  Train classes: {np.unique(y_fold_train, return_counts=True)}")
            print(f"  Val classes: {np.unique(y_fold_val, return_counts=True)}")

            # k-NN with safe handling
            knn_result = safe_classify_knn(X_fold_train, y_fold_train, X_fold_val, y_fold_val, k=5)
            app_results_knn.append(knn_result)
            print(f"  k-NN - Acc: {knn_result['accuracy']:.4f}, Prec: {knn_result['precision']:.4f}, "
                  f"Rec: {knn_result['recall']:.4f}, F1: {knn_result['f1']:.4f}")

            # Naive Bayes with safe handling
            nb_result = safe_classify_naive_bayes(X_fold_train, y_fold_train, X_fold_val, y_fold_val)
            app_results_nb.append(nb_result)
            print(f"  NB  - Acc: {nb_result['accuracy']:.4f}, Prec: {nb_result['precision']:.4f}, "
                  f"Rec: {nb_result['recall']:.4f}, F1: {nb_result['f1']:.4f}")

        results_knn[app] = app_results_knn
        results_nb[app] = app_results_nb

        # Final test evaluation with safe handling
        print(f"\nTest Set ({len(X_test)} samples):")
        knn_test = safe_classify_knn(X_train, y_train, X_test, y_test)
        nb_test = safe_classify_naive_bayes(X_train, y_train, X_test, y_test)

        print(f"  k-NN - Acc: {knn_test['accuracy']:.4f}, F1: {knn_test['f1']:.4f}")
        print(f"  NB   - Acc: {nb_test['accuracy']:.4f}, F1: {nb_test['f1']:.4f}")

        # Store test results
        test_results[app] = {
            'knn': knn_test,
            'nb': nb_test
        }

        # Only plot if we have valid axes and test samples
        if axes is not None and app in valid_apps and idx < len(valid_apps):
            app_idx = valid_apps.index(app)

            # k-NN confusion matrix (top row)
            safe_plot_confusion_matrix(
                y_test, knn_test['y_pred'],
                ax=axes[0, app_idx],
                title=f'k-NN - {app}\nAcc: {knn_test["accuracy"]:.3f}, F1: {knn_test["f1"]:.3f}',
                cmap='Blues'
            )

            # Naive Bayes confusion matrix (bottom row)
            safe_plot_confusion_matrix(
                y_test, nb_test['y_pred'],
                ax=axes[1, app_idx],
                title=f'Naive Bayes - {app}\nAcc: {nb_test["accuracy"]:.3f}, F1: {nb_test["f1"]:.3f}',
                cmap='Oranges'
            )

    # Only save plot if we created one
    if axes is not None and valid_apps:
        plt.tight_layout()
        plt.savefig('confusion_matrices_by_application.png', dpi=300, bbox_inches='tight')
        plt.show()
    else:
        print("\nNo valid applications for plotting confusion matrices")

    # ========================================================================
    # MODEL COMPARISON PLOTS (with safe handling)
    # ========================================================================

    # Filter applications that have valid test results
    valid_apps_for_plots = [app for app in test_results.keys()
                            if len(cv_tasks[app]['test_indices']) > 0]

    if valid_apps_for_plots:
        # Create model comparison plot
        apps = valid_apps_for_plots
        knn_acc = [test_results[app]['knn']['accuracy'] for app in apps]
        nb_acc = [test_results[app]['nb']['accuracy'] for app in apps]
        knn_f1 = [test_results[app]['knn']['f1'] for app in apps]
        nb_f1 = [test_results[app]['nb']['f1'] for app in apps]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Accuracy comparison
        x = np.arange(len(apps))
        width = 0.35
        ax1.bar(x - width / 2, knn_acc, width, label='k-NN', alpha=0.8)
        ax1.bar(x + width / 2, nb_acc, width, label='Naive Bayes', alpha=0.8)
        ax1.set_xlabel('Application')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Model Comparison - Accuracy')
        ax1.set_xticks(x)
        ax1.set_xticklabels(apps, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # F1-score comparison
        ax2.bar(x - width / 2, knn_f1, width, label='k-NN', alpha=0.8)
        ax2.bar(x + width / 2, nb_f1, width, label='Naive Bayes', alpha=0.8)
        ax2.set_xlabel('Application')
        ax2.set_ylabel('F1-Score')
        ax2.set_title('Model Comparison - F1 Score')
        ax2.set_xticks(x)
        ax2.set_xticklabels(apps, rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    else:
        print("\nNo valid applications for model comparison plots")

    # ========================================================================
    # ANALYSIS AND RECOMMENDATIONS (with safe handling)
    # ========================================================================

    print("\n" + "=" * 80)
    print("DATASET ANALYSIS")
    print("=" * 80)

    # Analyze the dataset issues
    for app, task_data in cv_tasks.items():
        y_train = task_data['y_train']
        y_test = task_data['y_test']

        train_attack_ratio = np.sum(y_train) / len(y_train) if len(y_train) > 0 else 0
        test_attack_ratio = np.sum(y_test) / len(y_test) if len(y_test) > 0 else 0

        print(f"\n{app}:")
        print(f"  Training samples: {len(y_train)}, Attack ratio: {train_attack_ratio:.3f}")
        print(f"  Test samples: {len(y_test)}, Attack ratio: {test_attack_ratio:.3f}")

        if train_attack_ratio == 0:
            print(f"  WARNING: No attack samples in training set!")
        if test_attack_ratio == 0:
            print(f"  WARNING: No attack samples in test set!")

    # Only proceed with analysis if we have valid results
    valid_apps_for_analysis = [app for app in test_results.keys()
                               if len(cv_tasks[app]['test_indices']) > 0 and
                               np.sum(cv_tasks[app]['y_test']) > 0]  # Only apps with attack samples in test

    if valid_apps_for_analysis:
        print("\n" + "=" * 80)
        print("MODEL RANKING AND IDS IMPLEMENTATION RECOMMENDATIONS")
        print("=" * 80)

        # Calculate average performance across valid applications
        knn_test_avg_acc = np.mean([test_results[app]['knn']['accuracy'] for app in valid_apps_for_analysis])
        nb_test_avg_acc = np.mean([test_results[app]['nb']['accuracy'] for app in valid_apps_for_analysis])
        knn_test_avg_f1 = np.mean([test_results[app]['knn']['f1'] for app in valid_apps_for_analysis])
        nb_test_avg_f1 = np.mean([test_results[app]['nb']['f1'] for app in valid_apps_for_analysis])

        print(f"\nPerformance on applications with attack samples:")
        print(f"k-NN  - Avg Acc: {knn_test_avg_acc:.4f}, Avg F1: {knn_test_avg_f1:.4f}")
        print(f"Naive Bayes - Avg Acc: {nb_test_avg_acc:.4f}, Avg F1: {nb_test_avg_f1:.4f}")

        # Rank models
        if knn_test_avg_f1 > nb_test_avg_f1:
            print(f"\n RECOMMENDED MODEL: k-NN (better F1-score)")
        else:
            print(f"\n RECOMMENDED MODEL: Naive Bayes (better F1-score)")

if __name__ == "__main__":
    main()