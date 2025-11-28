import pandas as pd
import numpy as np
import os
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
from nfstream import NFStreamer
import time
import multiprocessing


# ============================================================================
# FAST EXTRACTION OF PCAP FILES TO CSV (WITH THREADING)
# ============================================================================

def extract_single_pcap(pcap_file, output_dir, idle_timeout=60, active_timeout=120):
    """
    Extract flows from a single pcap file and save to CSV
    """
    try:
        filename = os.path.basename(pcap_file)
        csv_file = os.path.join(output_dir, filename.replace('.pcap', '.csv'))

        print(f"[EXTRACTING] {filename}...")
        start_time = time.time()

        flows_list = []
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

        # Save to CSV
        if flows_list:
            df = pd.DataFrame(flows_list)
            df.to_csv(csv_file, index=False)
            elapsed = time.time() - start_time
            print(f"[✓ DONE] {filename} -> {len(df)} flows ({elapsed:.2f}s)")
            return csv_file, len(df)
        else:
            print(f"[✗ EMPTY] {filename} (no flows extracted)")
            return None, 0

    except Exception as e:
        print(f"[✗ ERROR] {filename}: {e}")
        return None, 0


def extract_all_pcaps_threaded(pcap_dir, output_dir, num_threads=4, idle_timeout=60, active_timeout=120):
    """
    Extract all pcap files to CSV using process pool
    """
    print("=" * 80)
    print(f"PCAP TO CSV EXTRACTION (Fast Mode - {num_threads} processes)")
    print("=" * 80)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Find all pcap files
    pcap_files = glob.glob(os.path.join(pcap_dir, "*.pcap")) + \
                 glob.glob(os.path.join(pcap_dir, "*.pcapng"))

    print(f"Found {len(pcap_files)} pcap files\n")

    if not pcap_files:
        print("No pcap files found!")
        return []

    # Extract using process pool
    csv_files = []
    total_flows = 0

    with ProcessPoolExecutor(max_workers=num_threads) as executor:
        futures = {
            executor.submit(extract_single_pcap, pcap_file, output_dir, idle_timeout, active_timeout): pcap_file
            for pcap_file in pcap_files
        }

        for future in as_completed(futures):
            csv_file, flow_count = future.result()
            if csv_file:
                csv_files.append(csv_file)
                total_flows += flow_count

    print(f"\n{'=' * 80}")
    print(f"Extraction Complete: {len(csv_files)} CSV files, {total_flows} total flows")
    print(f"{'=' * 80}\n")

    return csv_files

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    PCAP_DIR = "./TRAIN"
    OUTPUT_CSV_DIR = "./extracted_flows"

    csv_files = extract_all_pcaps_threaded(PCAP_DIR, OUTPUT_CSV_DIR, num_threads=4)