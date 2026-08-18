"""
STELLAR Paper Figures (450 samples, 9 domains x 50)
Run: python generate_figures.py
Output: results/figures/
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os

matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['font.size'] = 11

output_dir = 'results/figures'
os.makedirs(output_dir, exist_ok=True)

# ============================================================
# Data (450 samples, 9 domains x 50)
# ============================================================

domains = ['Politics', 'Business', 'Disaster', 'Sports', 'Science',
           'Health', 'Culture', 'Crime', 'Environment']
overalls = [56.0, 56.0, 34.0, 88.0, 50.0,
            62.0, 72.0, 20.0, 54.0]

api_elements = ['Entity1', 'Relation', 'Entity2', 'Time', 'Location', 'Overall']
qwen = [94.7, 72.7, 79.7, 97.7, 89.7, 51.0]
mini = [94.0, 76.0, 78.0, 100.0, 94.0, 60.0]
gpt4o = [90.0, 80.0, 84.0, 100.0, 86.0, 60.0]

heatmap_data = np.array([
    [96.0, 74.0, 82.0, 100.0, 86.0],
    [92.0, 68.0, 84.0, 98.0, 96.0],
    [74.0, 66.0, 78.0, 100.0, 98.0],
    [98.0, 92.0, 92.0, 100.0, 96.0],
    [72.0, 76.0, 82.0, 100.0, 98.0],
    [96.0, 76.0, 92.0, 100.0, 96.0],
    [96.0, 80.0, 86.0, 100.0, 98.0],
    [40.0, 72.0, 84.0, 100.0, 100.0],
    [74.0, 76.0, 92.0, 100.0, 98.0],
])

stages = ['Direct', '+Atomic\nDecomp', '+CoT', '+Entity\nNorm',
          '+Domain\nBalance', '+Rule\nCorrect', '+Relation\nNorm', 'GPT-4o\n-mini']
stage_values = [2, 0, 22, 38, 46, 51, 52, 60]

bert_f1 = [97.8, 97.0, 96.6, 97.6, 97.2,
           96.9, 96.9, 95.9, 97.5]

elem_avg = [89.6, 87.6, 79.2, 94.8, 84.4, 92.4, 85.6, 82.0, 87.6]
strict = [60.0, 52.0, 28.0, 86.0, 42.0, 66.0, 40.0, 30.0, 52.0]


# ============================================================
# Fig. 2: Performance Improvement Journey
# ============================================================
fig, ax = plt.subplots(figsize=(12, 5))
colors = ['#c44e52', '#c44e52', '#1baf7a', '#1baf7a', '#1baf7a',
          '#1baf7a', '#1baf7a', '#2a78d6']

bars = ax.bar(stages, stage_values, color=colors, width=0.6, edgecolor='white')
for bar, val in zip(bars, stage_values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{val}%', ha='center', va='bottom', fontweight='bold', fontsize=11)

ax.set_ylabel('Overall PASS Rate (%)', fontsize=13)
ax.set_ylim(0, 75)
ax.set_title('Performance Improvement Journey',
             fontsize=14, fontweight='bold', pad=15)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{output_dir}/fig2_improvement.png', dpi=300, bbox_inches='tight')
plt.close()
print("Fig. 2 done")


# ============================================================
# Fig. 3: Model Comparison
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(api_elements))
w = 0.25
bars1 = ax.bar(x - w, qwen, w, label='Qwen2.5-7B', color='#888880', edgecolor='white')
bars2 = ax.bar(x, mini, w, label='GPT-4o-mini', color='#1baf7a', edgecolor='white')
bars3 = ax.bar(x + w, gpt4o, w, label='GPT-4o', color='#2a78d6', edgecolor='white')

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        val = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, val + 1,
                f'{val:.0f}', ha='center', va='bottom', fontsize=8)

ax.set_ylabel('PASS Rate (%)', fontsize=13)
ax.set_ylim(0, 115)
ax.set_xticks(x)
ax.set_xticklabels(api_elements)
ax.set_title('Model Comparison (Same Framework, n=50)',
             fontsize=13, fontweight='bold', pad=15)
ax.legend(loc='upper left', framealpha=0.9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{output_dir}/fig3_model_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("Fig. 3 done")


# ============================================================
# Fig. 4: Domain Overall Comparison
# ============================================================
fig, ax = plt.subplots(figsize=(14, 5))
sorted_pairs = sorted(zip(domains, overalls), key=lambda x: x[1], reverse=True)
sorted_domains = [p[0] + '\n(50)' for p in sorted_pairs]
sorted_overalls = [p[1] for p in sorted_pairs]

colors = []
for v in sorted_overalls:
    if v >= 70:
        colors.append('#2a78d6')
    elif v >= 40:
        colors.append('#1baf7a')
    else:
        colors.append('#c44e52')

bars = ax.bar(sorted_domains, sorted_overalls, color=colors, width=0.6, edgecolor='white')
for bar, val in zip(bars, sorted_overalls):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
            f'{val}%', ha='center', va='bottom', fontweight='bold', fontsize=11)

avg = sum(overalls) / len(overalls)
ax.axhline(y=avg, color='#333', linestyle='--', alpha=0.5, linewidth=1)
ax.text(len(sorted_domains)-0.5, avg + 2, f'Avg: {avg:.1f}%',
        ha='right', fontsize=9, color='#555')

ax.set_ylabel('Overall PASS Rate (%)', fontsize=13)
ax.set_ylim(0, 100)
ax.set_title('Domain Independence Test (9 Domains, n=50 each, No Prompt Modification)',
             fontsize=13, fontweight='bold', pad=15)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{output_dir}/fig4_domain_overall.png', dpi=300, bbox_inches='tight')
plt.close()
print("Fig. 4 done")


# ============================================================
# Fig. 5: Domain Heatmap
# ============================================================
fig, ax = plt.subplots(figsize=(12, 5))
elements = ['Entity1', 'Relation', 'Entity2', 'Time', 'Location']

im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)

ax.set_xticks(np.arange(len(elements)))
ax.set_yticks(np.arange(len(domains)))
ax.set_xticklabels(elements)
ax.set_yticklabels(domains)

for i in range(len(domains)):
    for j in range(len(elements)):
        color = 'white' if heatmap_data[i, j] < 50 else 'black'
        ax.text(j, i, f'{heatmap_data[i, j]:.0f}%',
                ha='center', va='center', fontsize=10,
                fontweight='bold', color=color)

ax.set_title('Element-wise PASS Rate by Domain (%, n=50)',
             fontsize=13, fontweight='bold', pad=15)
fig.colorbar(im, ax=ax, shrink=0.8, label='PASS Rate (%)')
plt.tight_layout()
plt.savefig(f'{output_dir}/fig5_domain_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("Fig. 5 done")


# ============================================================
# Fig. 6: G-BERTScore by Domain
# ============================================================
fig, ax = plt.subplots(figsize=(12, 5))
sorted_bert = sorted(zip(domains, bert_f1), key=lambda x: x[1], reverse=True)
sorted_bd = [p[0] for p in sorted_bert]
sorted_bf = [p[1] for p in sorted_bert]

bars = ax.bar(sorted_bd, sorted_bf, color='#3498db', width=0.6, edgecolor='white')
for bar, val in zip(bars, sorted_bf):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
            f'{val}%', ha='center', va='bottom', fontweight='bold', fontsize=11)

avg_bert = sum(bert_f1) / len(bert_f1)
ax.axhline(y=avg_bert, color='#333', linestyle='--', alpha=0.5, linewidth=1)
ax.text(len(sorted_bd)-0.5, avg_bert + 0.3, f'Avg: {avg_bert:.1f}%',
        ha='right', fontsize=9, color='#555')

ax.set_ylabel('G-BERTScore F1 (%)', fontsize=13)
ax.set_ylim(94, 99)
ax.set_title('G-BERTScore F1 by Domain (Quintuple Text Similarity, n=50)',
             fontsize=13, fontweight='bold', pad=15)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{output_dir}/fig6_bertscore.png', dpi=300, bbox_inches='tight')
plt.close()
print("Fig. 6 done")


# ============================================================
# Fig. 7: Element Average vs Strict Match
# ============================================================
fig, ax = plt.subplots(figsize=(12, 5))

x = np.arange(len(domains))
w = 0.35
bars1 = ax.bar(x - w/2, elem_avg, w, label='Element Average',
               color='#3498db', edgecolor='white')
bars2 = ax.bar(x + w/2, strict, w, label='Strict Match',
               color='#e74c3c', edgecolor='white')

for bar, val in zip(bars1, elem_avg):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{val:.0f}', ha='center', va='bottom', fontsize=8)
for bar, val in zip(bars2, strict):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{val:.0f}', ha='center', va='bottom', fontsize=8)

ax.set_ylabel('Rate (%)', fontsize=13)
ax.set_ylim(0, 110)
ax.set_xticks(x)
ax.set_xticklabels(domains)
ax.set_title('Element Average vs Strict Match by Domain (n=50)',
             fontsize=13, fontweight='bold', pad=15)
ax.legend(loc='upper right', framealpha=0.9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{output_dir}/fig7_avg_vs_strict.png', dpi=300, bbox_inches='tight')
plt.close()
print("Fig. 7 done")


print(f"\nAll figures generated! -> {output_dir}/")