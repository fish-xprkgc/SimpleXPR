import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# --- Configuration ---
LOG_DIR = r'D:\PycharmProjects\XPR-KGC\log_new/wn_path_new/'
ADVANCED_STRATEGY_WEIGHT = 0.9
THRESHOLDS = [1, 3, 10]


def get_rank(sorted_entity_list, true_tail):
    """
    Finds the rank (1-based index) of the true tail entity in a sorted list.
    Returns -1 if the entity is not found.
    """
    try:
        return sorted_entity_list.index(true_tail) + 1
    except ValueError:
        return -1

# --- 1. Data Loading ---
print("Loading data...")
try:
    with open(LOG_DIR + 'WN18RR_path_result.json', 'r') as f:
        path_data = json.load(f)
    with open(LOG_DIR + 'WN18RR_hr_result.json', 'r') as f:
        hr_data = json.load(f)
except FileNotFoundError:
    print(f"Error: Could not find JSON files in the specified directory: {LOG_DIR}")
    print("Please ensure the LOG_DIR variable is set correctly.")
    exit()

# --- 2. Data Alignment and Preprocessing ---
all_results = []
for rel in path_data:
    if rel not in hr_data:
        continue
    for idx, item in enumerate(path_data[rel]):
        if idx >= len(hr_data[rel]):
            break
        hr_item = hr_data[rel][idx]
        path_logits = [val**(1/6)for idx,val in enumerate(item['finish_path_logit'])]
        all_results.append({
            "path_tails": [p[-1] for p in item['finish_path']],
            "path_logits": path_logits,
            "hr_tails": hr_item['hr_tail'],
            "hr_logits": hr_item['hr_logit'],
            "tail": item['tail'],
        })
total_samples = len(all_results)
print(f"Successfully loaded and aligned {total_samples} samples.")

# --- 3. Main Evaluation Loop ---
ranks = {
    'HR Only': [],
    'Path Only': [],
    'Simple Combined': [],
    'Advanced Combined': []
}
for result in all_results:
    tail = result["tail"]
    # Strategy 1: HR Only
    ranks['HR Only'].append(get_rank(result["hr_tails"], tail))
    # Strategy 2: Path Only
    ranks['Path Only'].append(get_rank(result["path_tails"], tail))
    # Strategy 3: Simple Combined
    simple_scores = {ent: logit for ent, logit in zip(result["hr_tails"], result["hr_logits"])}
    for ent, logit in zip(result["path_tails"], result["path_logits"]):
        simple_scores[ent] = max(simple_scores.get(ent, -1.0), logit)
    simple_sorted_entities = [e[0] for e in sorted(simple_scores.items(), key=lambda x: x[1], reverse=True)]
    ranks['Simple Combined'].append(get_rank(simple_sorted_entities, tail))
    # Strategy 4: Advanced Combined
    advanced_scores = {}
    path_tails, path_logits = result["path_tails"], result["path_logits"]
    hr_tails, hr_logits = result["hr_tails"], result["hr_logits"]
    if path_tails and path_logits:
        top1_ent, top1_logit = path_tails[0], path_logits[0]
        if top1_ent in hr_tails:
            top1_logit += 0.1
        if top1_logit >= ADVANCED_STRATEGY_WEIGHT:
             advanced_scores[top1_ent] = top1_logit / 0.9
    for i, (ent, logit) in enumerate(zip(path_tails[1:], path_logits[1:])):
        decay = 0.9 ** i
        if ent in hr_tails: logit += 0.1
        if logit >= ADVANCED_STRATEGY_WEIGHT:
            advanced_scores[ent] = max(advanced_scores.get(ent, -1.0), decay * logit)
    for i, ent in enumerate(hr_tails[:10]):
        score = hr_logits[i] if i < len(hr_logits) else 0.0
        advanced_scores[ent] = max(advanced_scores.get(ent, -1.0), score)
    advanced_sorted_entities = [e[0] for e in sorted(advanced_scores.items(), key=lambda x: x[1], reverse=True)]
    ranks['Advanced Combined'].append(get_rank(advanced_sorted_entities, tail))

# --- 4. Calculate and Display Hit@K Table ---
hit_at_k_results = {}
for strategy, rank_list in ranks.items():
    hit_counts = {t: 0 for t in THRESHOLDS}
    for rank in rank_list:
        if rank != -1:
            for t in THRESHOLDS:
                if rank <= t:
                    hit_counts[t] += 1
    hit_at_k_results[strategy] = {t: count / total_samples for t, count in hit_counts.items()}
print("\n--- Hit@K Performance Comparison ---")
header = f"{'Strategy':<20}" + "".join([f"Hits@{t:<10}" for t in THRESHOLDS])
print(header)
print("-" * len(header))
for strategy, hits in hit_at_k_results.items():
    row = f"{strategy:<20}" + "".join([f"{hits[t]:.4%}".ljust(10) for t in THRESHOLDS])
    print(row)

# --- 5. Generate Cumulative Distribution Plot ---
def count_rank_frequencies(rank_list, limit=10):
    freq = defaultdict(int)
    for rank in rank_list:
        if rank == -1: freq['Not Found'] += 1
        elif rank > limit: freq['>'+str(limit)] += 1
        else: freq[rank] += 1
    return freq

def get_cumulative_proportions(rank_list, total, limit=10):
    freqs = count_rank_frequencies(rank_list, limit)
    cumulative_proportions = []
    current_sum = 0
    for k in range(1, limit + 1):
        current_sum += freqs.get(k, 0)
        cumulative_proportions.append(current_sum / total)
    return cumulative_proportions

plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(12, 8))
indices_range = list(range(1, 11))
markers = ['o', 's', '^', 'D']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

all_y_values = [] # <-- NEW: List to store all data points for range calculation

for i, (strategy, rank_list) in enumerate(ranks.items()):
    cumulative_data = get_cumulative_proportions(rank_list, total_samples, limit=10)
    all_y_values.extend(cumulative_data) # <-- NEW: Add data to the list
    plt.plot(indices_range, cumulative_data, label=strategy, marker=markers[i], color=colors[i])
    for x, y in zip(indices_range, cumulative_data):
        if x % 2 == 1 or x == 10:
             plt.text(x, y, f" {y:.2%}", ha='left', va='center', fontsize=16)

# --- Y-Axis Adjustment Logic ---
if all_y_values:
    min_val = min(all_y_values) # <-- NEW: Find the minimum value plotted
    max_val = max(all_y_values) # <-- NEW: Find the maximum value plotted
    padding = (max_val - min_val) * 0.1 # <-- NEW: Calculate 10% padding
    plt.ylim(min_val - padding, max_val + padding) # <-- MODIFIED: Set dynamic Y-axis limits

plt.xticks(indices_range)
plt.xlabel('Entity Rank Position', fontsize=16)
plt.ylabel('Cumulative Proportion (Hits@K)', fontsize=16)
plt.title('Comparison of Link Prediction Strategies (Cumulative Hits@K)', fontsize=16, weight='bold')
plt.legend(title='Strategy', fontsize=16)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

output_path = LOG_DIR + 'four_way_comparison_plot_zoomed.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n📈 Zoomed plot successfully saved to: {output_path}")
plt.show()