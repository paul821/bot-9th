"""
Session 1: Exploratory Data Orientation
Terminal Game State Strategy in Baseball

Steps:
1. Environment setup & caching
2. Statcast sample size validation (2023)
3. Retrosheet orientation
4. Quick visual orientation
5. Standings data check
6. Summary report generation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ── Step 1: Enable pybaseball caching ──────────────────────────────────────
from pybaseball import cache
cache.enable()
print("✓ pybaseball cache enabled")

# ── Step 2: Statcast sample size validation ────────────────────────────────
# Pull 2023 Statcast data in monthly chunks to avoid timeouts
from pybaseball import statcast

print("\n── Step 2: Pulling 2023 Statcast data ──")
chunks = []
date_ranges = [
    ("2023-03-30", "2023-04-30"),
    ("2023-05-01", "2023-05-31"),
    ("2023-06-01", "2023-06-30"),
    ("2023-07-01", "2023-07-31"),
    ("2023-08-01", "2023-08-31"),
    ("2023-09-01", "2023-10-01"),
]

for start, end in date_ranges:
    print(f"  Pulling {start} to {end}...")
    chunk = statcast(start_dt=start, end_dt=end)
    chunks.append(chunk)
    print(f"    → {len(chunk):,} pitches")

sc_2023 = pd.concat(chunks, ignore_index=True)
print(f"\n  Total 2023 pitches: {len(sc_2023):,}")
print(f"  Columns ({len(sc_2023.columns)}): {list(sc_2023.columns[:20])}...")

# ── Primary situation filter ───────────────────────────────────────────────
# Bot 9, 2 outs, bases loaded
primary = sc_2023[
    (sc_2023['inning'] == 9) &
    (sc_2023['inning_topbot'] == 'Bot') &
    (sc_2023['outs_when_up'] == 2) &
    (sc_2023['on_1b'].notna()) &
    (sc_2023['on_2b'].notna()) &
    (sc_2023['on_3b'].notna())
]
print(f"\n── Primary situation (Bot 9, 2 out, bases loaded) ──")
print(f"  Total pitches: {len(primary):,}")

# Unique PAs = unique (game_pk, at_bat_number)
primary_pa = primary.groupby(['game_pk', 'at_bat_number']).last().reset_index()
print(f"  Unique plate appearances: {len(primary_pa):,}")

# ── Secondary situation filter ─────────────────────────────────────────────
# Bot 9, 2 outs, tying run on base
# (fld_score - bat_score) <= number of occupied bases
bot9_2out = sc_2023[
    (sc_2023['inning'] == 9) &
    (sc_2023['inning_topbot'] == 'Bot') &
    (sc_2023['outs_when_up'] == 2)
].copy()

# Count runners on base
bot9_2out['runners_on'] = (
    bot9_2out['on_1b'].notna().astype(int) +
    bot9_2out['on_2b'].notna().astype(int) +
    bot9_2out['on_3b'].notna().astype(int)
)

# At least one runner on base and tying run is on base
# i.e., deficit <= number of runners
bot9_2out['score_deficit'] = bot9_2out['fld_score'] - bot9_2out['bat_score']

secondary = bot9_2out[
    (bot9_2out['runners_on'] >= 1) &
    (bot9_2out['score_deficit'] <= bot9_2out['runners_on'])
]

secondary_pa = secondary.groupby(['game_pk', 'at_bat_number']).last().reset_index()
print(f"\n── Secondary situation (Bot 9, 2 out, tying run on base) ──")
print(f"  Total pitches: {len(secondary):,}")
print(f"  Unique plate appearances: {len(secondary_pa):,}")

# Show score deficit distribution in secondary
print(f"\n  Score deficit distribution in secondary situation PAs:")
print(secondary_pa['score_deficit'].value_counts().sort_index())

# ── Step 2 data quality checks ────────────────────────────────────────────
print(f"\n── Key field availability check ──")
key_fields = ['pitch_type', 'release_speed', 'pfx_x', 'pfx_z', 'plate_x', 'plate_z',
              'balls', 'strikes', 'on_1b', 'on_2b', 'on_3b', 'inning', 'inning_topbot',
              'outs_when_up', 'bat_score', 'fld_score', 'events', 'description',
              'game_pk', 'at_bat_number', 'batter', 'pitcher']
for f in key_fields:
    if f in sc_2023.columns:
        null_pct = sc_2023[f].isna().mean() * 100
        print(f"  {f:20s} ✓ present  ({null_pct:5.1f}% null)")
    else:
        print(f"  {f:20s} ✗ MISSING")

# Save primary situation data for Step 4
primary.to_csv('/Users/paul821/Desktop/baseball/baseball_terminal_state/primary_2023_pitches.csv', index=False)
secondary_pa.to_csv('/Users/paul821/Desktop/baseball/baseball_terminal_state/secondary_2023_pa.csv', index=False)
print("\n✓ Situation data cached to CSV")
