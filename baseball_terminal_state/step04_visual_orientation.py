"""
Step 4: Quick Visual Orientation
Compare pitch type distribution and first-pitch strike rate in
primary situation vs. overall season (2023).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pybaseball import cache
cache.enable()

# Load cached primary situation pitches from Step 2
primary = pd.read_csv('/Users/paul821/Desktop/baseball/baseball_terminal_state/primary_2023_pitches.csv')
print(f"Primary situation pitches: {len(primary):,}")

# ── Pull overall 2023 season sample for comparison ──
# Instead of loading all 720K pitches again, pull a one-month sample as a proxy
from pybaseball import statcast
print("Loading June 2023 as overall season proxy...")
season_sample = statcast(start_dt="2023-06-01", end_dt="2023-06-30")
print(f"Season sample pitches: {len(season_sample):,}")

# ── Figure 1: Pitch type distribution comparison ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Clean pitch types
primary_types = primary['pitch_type'].dropna().value_counts(normalize=True).head(8)
season_types = season_sample['pitch_type'].dropna().value_counts(normalize=True).head(8)

# Align categories
all_types = sorted(set(primary_types.index) | set(season_types.index))
comp = pd.DataFrame({
    'Primary Situation': primary_types.reindex(all_types, fill_value=0),
    'Season Overall': season_types.reindex(all_types, fill_value=0)
})

comp.plot(kind='bar', ax=axes[0], color=['#d62728', '#1f77b4'], alpha=0.85)
axes[0].set_title('Pitch Type Distribution\nBot 9, 2 Out, Bases Loaded vs Season Overall (2023)', fontsize=11)
axes[0].set_ylabel('Proportion')
axes[0].set_xlabel('Pitch Type')
axes[0].tick_params(axis='x', rotation=0)
axes[0].legend(fontsize=9)

# ── Figure 2: First-pitch strike rate comparison ──
# First pitch = where balls == 0 and strikes == 0
primary_first = primary[(primary['balls'] == 0) & (primary['strikes'] == 0)]
season_first = season_sample[(season_sample['balls'] == 0) & (season_sample['strikes'] == 0)]

# Strike on first pitch: called_strike, swinging_strike, foul, foul_tip, hit_into_play (in strike zone)
strike_descriptions = ['called_strike', 'swinging_strike', 'foul', 'foul_tip',
                       'swinging_strike_blocked', 'foul_bunt', 'missed_bunt']

primary_fps = primary_first['description'].isin(strike_descriptions).mean()
season_fps = season_first['description'].isin(strike_descriptions).mean()

# Also include in_play as strikes
primary_fps_with_play = (primary_first['description'].isin(strike_descriptions) |
                          primary_first['description'].str.contains('into_play', na=False)).mean()
season_fps_with_play = (season_first['description'].isin(strike_descriptions) |
                         season_first['description'].str.contains('into_play', na=False)).mean()

fps_data = pd.DataFrame({
    'Context': ['Primary\nSituation', 'Season\nOverall'],
    'First-Pitch Strike %\n(excl. in-play)': [primary_fps * 100, season_fps * 100],
    'First-Pitch Strike %\n(incl. in-play)': [primary_fps_with_play * 100, season_fps_with_play * 100]
})

x = np.arange(2)
width = 0.3
axes[1].bar(x - width/2, fps_data['First-Pitch Strike %\n(excl. in-play)'], width,
            label='Excl. in-play', color='#d62728', alpha=0.85)
axes[1].bar(x + width/2, fps_data['First-Pitch Strike %\n(incl. in-play)'], width,
            label='Incl. in-play', color='#1f77b4', alpha=0.85)
axes[1].set_xticks(x)
axes[1].set_xticklabels(fps_data['Context'])
axes[1].set_ylabel('First-Pitch Strike Rate (%)')
axes[1].set_title('First-Pitch Strike Rate\nBot 9, 2 Out, Bases Loaded vs Season Overall (2023)', fontsize=11)
axes[1].legend(fontsize=9)
axes[1].set_ylim(0, 80)

# Add value labels
for i, (v1, v2) in enumerate(zip(fps_data['First-Pitch Strike %\n(excl. in-play)'],
                                   fps_data['First-Pitch Strike %\n(incl. in-play)'])):
    axes[1].text(i - width/2, v1 + 1.5, f'{v1:.1f}%', ha='center', fontsize=9)
    axes[1].text(i + width/2, v2 + 1.5, f'{v2:.1f}%', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('/Users/paul821/Desktop/baseball/baseball_terminal_state/fig_step04_orientation.png', dpi=150)
plt.close()
print("\n✓ Figure saved to fig_step04_orientation.png")

# Print numeric summary
print(f"\n── Numeric Summary ──")
print(f"First-pitch strike rate (excl. in-play):")
print(f"  Primary situation: {primary_fps*100:.1f}% (n={len(primary_first)})")
print(f"  Season overall:    {season_fps*100:.1f}% (n={len(season_first)})")
print(f"\nPitch type shares (top 5):")
for pt in comp.index[:5]:
    print(f"  {pt:4s}  Primary: {comp.loc[pt, 'Primary Situation']*100:5.1f}%  "
          f"Season: {comp.loc[pt, 'Season Overall']*100:5.1f}%  "
          f"Diff: {(comp.loc[pt, 'Primary Situation'] - comp.loc[pt, 'Season Overall'])*100:+5.1f}%")
