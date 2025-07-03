import pandas as pd
import matplotlib.pyplot as plt
import re
import glob

csv_files = glob.glob('batch*.csv')

for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    df = df.drop_duplicates()
    
    batch_size = re.search(r'batch(\d+)', csv_file).group(1)
    
    grouped_data = df.groupby(['algorithm', 'write_ratio'])['search_qps'].mean().reset_index()
    
    plt.figure(figsize=(12, 8))
    
    algorithms = ['hnsw', 'parlayhnsw', 'parlayvamana', 'vamana']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    markers = ['o', 's', '^', 'D']
    
    for i, algo in enumerate(algorithms):
        algo_data = grouped_data[grouped_data['algorithm'] == algo]
        plt.plot(algo_data['write_ratio'], algo_data['search_qps'], 
                 marker=markers[i], linewidth=2, markersize=8, 
                 color=colors[i], label=algo.upper())
    
    plt.xlabel('Write Ratio', fontsize=14)
    plt.ylabel('Search QPS', fontsize=14)
    plt.title(f'Search QPS Comparison Across Algorithms and Workloads (Batch Size: {batch_size})', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks([0.1, 0.2, 0.5, 0.8, 0.9], fontsize=12)
    plt.yticks(fontsize=12)
    
    for i, algo in enumerate(algorithms):
        algo_data = grouped_data[grouped_data['algorithm'] == algo]
        for _, row in algo_data.iterrows():
            plt.annotate(f'{row["search_qps"]:.0f}', 
                        (row['write_ratio'], row['search_qps']),
                        textcoords="offset points", 
                        xytext=(0,10), 
                        ha='center', 
                        fontsize=9,
                        color=colors[i])
    
    plt.tight_layout()
    output_file = f'search_qps_comparison_batch{batch_size}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nBatch Size {batch_size} - Average Search QPS for each algorithm across different write ratios:")
    pivot_table = grouped_data.pivot(index='write_ratio', columns='algorithm', values='search_qps')
    print(pivot_table.round(2))
