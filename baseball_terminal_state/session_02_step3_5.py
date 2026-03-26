"""
Session 2, Steps 3-5: Baseline construction and deviation computation
Runs AFTER session_02_baselines.py has cached all yearly data.
"""

import pandas as pd
import numpy as np
import os
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = "/Users/paul821/Desktop/baseball/baseball_terminal_state"

# ── Helper functions ────────────────────────────────────────────────────────

def is_strike(desc):
    strikes = {'called_strike', 'swinging_strike', 'foul', 'foul_tip',
               'swinging_strike_blocked', 'foul_bunt', 'missed_bunt',
               'hit_into_play', 'hit_into_play_no_out', 'hit_into_play_score'}
    return desc in strikes if isinstance(desc, str) else False

def is_in_zone(zone):
    try:
        z = float(zone)
        return 1 <= z <= 9
    except (ValueError, TypeError):
        return False

def is_swing(desc):
    swings = {'swinging_strike', 'foul', 'foul_tip', 'swinging_strike_blocked',
              'foul_bunt', 'missed_bunt', 'hit_into_play', 'hit_into_play_no_out',
              'hit_into_play_score', 'foul_pitchout', 'swinging_pitchout'}
    return desc in swings if isinstance(desc, str) else False

def is_contact(desc):
    contacts = {'foul', 'foul_tip', 'foul_bunt', 'hit_into_play',
                'hit_into_play_no_out', 'hit_into_play_score'}
    return desc in contacts if isinstance(desc, str) else False

def is_whiff(desc):
    whiffs = {'swinging_strike', 'swinging_strike_blocked',
              'missed_bunt', 'swinging_pitchout'}
    return desc in whiffs if isinstance(desc, str) else False

HARD_TYPES = {'FF', 'SI', 'FC'}
OFFSPEED_TYPES = {'CH', 'CU'}
PITCH_TYPES = {'FF', 'SI', 'FC', 'SL', 'CH', 'CU'}

MIN_PA = 50  # Minimum PA for inclusion in baseline


# ═══════════════════════════════════════════════════════════════════════════
# STEP 3: Build pitcher baseline metrics (Statcast era)
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print(" STEP 3: Build pitcher baseline metrics")
print("=" * 70)

def compute_pitcher_metrics(df, pitcher_id=None):
    """Compute pitcher metrics from a set of pitches.
    Returns a dict of metric values, or None if insufficient data.
    """
    if pitcher_id is not None:
        df = df[df['pitcher'] == pitcher_id]

    if len(df) == 0:
        return None

    # Count PAs (rows where events is non-null)
    pa_count = df['events'].notna().sum()

    total = len(df)
    typed = df[df['pitch_type'].isin(PITCH_TYPES)]
    typed_total = len(typed)

    if typed_total == 0:
        return None

    # Pitch type shares
    type_counts = typed['pitch_type'].value_counts()
    ff_pct = type_counts.get('FF', 0) / typed_total
    si_pct = type_counts.get('SI', 0) / typed_total
    fc_pct = type_counts.get('FC', 0) / typed_total
    sl_pct = type_counts.get('SL', 0) / typed_total
    ch_pct = type_counts.get('CH', 0) / typed_total
    cu_pct = type_counts.get('CU', 0) / typed_total
    hard_pct = ff_pct + si_pct + fc_pct
    offspeed_pct = ch_pct + cu_pct

    # First-pitch strike rate: pitch 1 of each PA
    first_pitches = df[(df['balls'] == 0) & (df['strikes'] == 0)]
    fps_rate = first_pitches['description'].apply(is_strike).mean() if len(first_pitches) > 0 else np.nan

    # Zone rate
    zone_valid = df[df['zone'].notna()]
    zone_rate = zone_valid['zone'].apply(is_in_zone).mean() if len(zone_valid) > 0 else np.nan

    # Chase rate induced: swings on pitches outside zone / pitches outside zone
    outside = df[df['zone'].apply(lambda z: not is_in_zone(z)) & df['zone'].notna()]
    if len(outside) > 0:
        chase_rate = outside['description'].apply(is_swing).mean()
    else:
        chase_rate = np.nan

    return {
        'pa_count': pa_count,
        'pitch_count': total,
        'ff_pct': ff_pct, 'si_pct': si_pct, 'fc_pct': fc_pct,
        'sl_pct': sl_pct, 'ch_pct': ch_pct, 'cu_pct': cu_pct,
        'hard_pct': hard_pct, 'offspeed_pct': offspeed_pct,
        'fps_rate': fps_rate, 'zone_rate': zone_rate,
        'chase_rate_induced': chase_rate,
    }


all_pitcher_baselines = []

for year in range(2015, 2025):
    baseline_path = f"{BASE_DIR}/data/statcast/baseline_pitcher_{year}.csv"
    primary_path = f"{BASE_DIR}/data/statcast/primary_{year}.csv"
    secondary_path = f"{BASE_DIR}/data/statcast/secondary_{year}.csv"

    if not os.path.exists(baseline_path):
        print(f"  {year}: no baseline file, skipping")
        continue

    baseline = pd.read_csv(baseline_path, low_memory=False)
    primary = pd.read_csv(primary_path, low_memory=False)
    secondary = pd.read_csv(secondary_path, low_memory=False)

    # Get pitchers who appear in situation sample
    sit_pitchers = set(primary['pitcher'].unique()) | set(secondary['pitcher'].unique())

    print(f"  {year}: {len(sit_pitchers)} situation pitchers, {len(baseline):,} baseline pitches")

    for pid in sit_pitchers:
        pid_baseline = baseline[baseline['pitcher'] == pid]

        # Check minimum PA threshold
        pa_count = pid_baseline['events'].notna().sum()
        if pa_count < MIN_PA:
            continue

        metrics = compute_pitcher_metrics(pid_baseline)
        if metrics is None:
            continue

        metrics['pitcher_id'] = pid
        metrics['season'] = year
        all_pitcher_baselines.append(metrics)

    del baseline
    print(f"    → {len([b for b in all_pitcher_baselines if b['season'] == year])} pitchers with baselines")

pitcher_baselines = pd.DataFrame(all_pitcher_baselines)
pitcher_baselines.to_csv(f"{BASE_DIR}/data/baselines/pitcher_baselines_statcast.csv", index=False)
print(f"\n  Total pitcher-season baselines: {len(pitcher_baselines)}")
print(f"  Unique pitchers: {pitcher_baselines['pitcher_id'].nunique()}")
print(f"  Mean baseline PA per pitcher: {pitcher_baselines['pa_count'].mean():.0f}")


# ═══════════════════════════════════════════════════════════════════════════
# STEP 4: Build batter baseline metrics (Statcast era)
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(" STEP 4: Build batter baseline metrics")
print("=" * 70)

def compute_batter_metrics(df, batter_id=None):
    """Compute batter metrics from a set of pitches."""
    if batter_id is not None:
        df = df[df['batter'] == batter_id]

    if len(df) == 0:
        return None

    total_pitches = len(df)
    # PAs = unique (game_pk, at_bat_number) with a terminal event
    pa_rows = df[df['events'].notna()]
    pa_count = len(pa_rows.groupby(['game_pk', 'at_bat_number']))

    if pa_count < MIN_PA:
        return None

    # K% and BB%
    events = pa_rows['events']
    k_pct = (events == 'strikeout').sum() / pa_count if pa_count > 0 else np.nan
    bb_pct = (events == 'walk').sum() / pa_count if pa_count > 0 else np.nan

    # Swing rate
    swing_count = df['description'].apply(is_swing).sum()
    swing_pct = swing_count / total_pitches if total_pitches > 0 else np.nan

    # Chase rate (O-swing%): swings outside zone / pitches outside zone
    outside = df[df['zone'].apply(lambda z: not is_in_zone(z)) & df['zone'].notna()]
    chase_pct = outside['description'].apply(is_swing).mean() if len(outside) > 0 else np.nan

    # Contact rate: contacts / swings
    swings = df[df['description'].apply(is_swing)]
    contact_pct = swings['description'].apply(is_contact).mean() if len(swings) > 0 else np.nan

    # First-pitch swing rate
    first_pitches = df[(df['balls'] == 0) & (df['strikes'] == 0)]
    fps_swing_pct = first_pitches['description'].apply(is_swing).mean() if len(first_pitches) > 0 else np.nan

    # Zone swing rate (Z-swing%)
    in_zone = df[df['zone'].apply(is_in_zone) & df['zone'].notna()]
    zone_swing_pct = in_zone['description'].apply(is_swing).mean() if len(in_zone) > 0 else np.nan

    return {
        'pa_count': pa_count,
        'pitch_count': total_pitches,
        'k_pct': k_pct, 'bb_pct': bb_pct,
        'swing_pct': swing_pct, 'chase_pct': chase_pct,
        'contact_pct': contact_pct,
        'fps_swing_pct': fps_swing_pct,
        'zone_swing_pct': zone_swing_pct,
    }


all_batter_baselines = []

for year in range(2015, 2025):
    baseline_path = f"{BASE_DIR}/data/statcast/baseline_batter_{year}.csv"
    primary_path = f"{BASE_DIR}/data/statcast/primary_{year}.csv"
    secondary_path = f"{BASE_DIR}/data/statcast/secondary_{year}.csv"

    if not os.path.exists(baseline_path):
        print(f"  {year}: no baseline file, skipping")
        continue

    baseline = pd.read_csv(baseline_path, low_memory=False)
    primary = pd.read_csv(primary_path, low_memory=False)
    secondary = pd.read_csv(secondary_path, low_memory=False)

    sit_batters = set(primary['batter'].unique()) | set(secondary['batter'].unique())

    print(f"  {year}: {len(sit_batters)} situation batters, {len(baseline):,} baseline pitches")

    for bid in sit_batters:
        metrics = compute_batter_metrics(baseline, batter_id=bid)
        if metrics is None:
            continue

        metrics['batter_id'] = bid
        metrics['season'] = year
        all_batter_baselines.append(metrics)

    del baseline
    print(f"    → {len([b for b in all_batter_baselines if b['season'] == year])} batters with baselines")

batter_baselines = pd.DataFrame(all_batter_baselines)
batter_baselines.to_csv(f"{BASE_DIR}/data/baselines/batter_baselines_statcast.csv", index=False)
print(f"\n  Total batter-season baselines: {len(batter_baselines)}")
print(f"  Unique batters: {batter_baselines['batter_id'].nunique()}")
print(f"  Mean baseline PA per batter: {batter_baselines['pa_count'].mean():.0f}")


# ═══════════════════════════════════════════════════════════════════════════
# STEP 5: Compute deviation metrics
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(" STEP 5: Compute Statcast era deviation metrics")
print("=" * 70)

# ── 5a: Pitcher deviations ──────────────────────────────────────────────

PITCHER_METRICS = ['ff_pct', 'si_pct', 'fc_pct', 'sl_pct', 'ch_pct', 'cu_pct',
                   'hard_pct', 'offspeed_pct', 'fps_rate', 'zone_rate', 'chase_rate_induced']

print("\n  5a: Pitcher deviations (primary situation)")

all_pitcher_devs = []

for year in range(2015, 2025):
    primary_path = f"{BASE_DIR}/data/statcast/primary_{year}.csv"
    if not os.path.exists(primary_path):
        continue

    primary = pd.read_csv(primary_path, low_memory=False)
    year_baselines = pitcher_baselines[pitcher_baselines['season'] == year]

    # Group primary pitches by pitcher
    for pid, group in primary.groupby('pitcher'):
        baseline_row = year_baselines[year_baselines['pitcher_id'] == pid]
        if len(baseline_row) == 0:
            continue

        sit_metrics = compute_pitcher_metrics(group)
        if sit_metrics is None:
            continue

        baseline_row = baseline_row.iloc[0]

        dev = {
            'pitcher_id': pid,
            'season': year,
            'sit_pitches': sit_metrics['pitch_count'],
            'sit_pa': sit_metrics['pa_count'],
            'baseline_pa': baseline_row['pa_count'],
        }

        for m in PITCHER_METRICS:
            dev[f'sit_{m}'] = sit_metrics.get(m, np.nan)
            dev[f'base_{m}'] = baseline_row.get(m, np.nan)
            sit_val = sit_metrics.get(m, np.nan)
            base_val = baseline_row.get(m, np.nan)
            if pd.notna(sit_val) and pd.notna(base_val):
                dev[f'dev_{m}'] = sit_val - base_val
            else:
                dev[f'dev_{m}'] = np.nan

        all_pitcher_devs.append(dev)

pitcher_dev_df = pd.DataFrame(all_pitcher_devs)
pitcher_dev_df.to_csv(f"{BASE_DIR}/data/deviations/statcast_pitcher_deviations.csv", index=False)

print(f"  Pitcher deviation rows: {len(pitcher_dev_df)} (pitcher-seasons)")
print(f"  Covering {pitcher_dev_df['season'].nunique()} seasons")

# Aggregate mean deviations with CIs
print("\n  Aggregate pitcher deviations (pooled across all years):")
print(f"  {'Metric':<22} {'Mean Dev':>10} {'95% CI':>20} {'t-stat':>8} {'p-value':>10} {'Sig':>5}")
print("  " + "-" * 77)

n_tests_p = len(PITCHER_METRICS)
pitcher_agg = {}

for m in PITCHER_METRICS:
    col = f'dev_{m}'
    vals = pitcher_dev_df[col].dropna()
    if len(vals) < 5:
        continue

    mean_dev = vals.mean()
    sem = vals.sem()
    ci_lo = mean_dev - 1.96 * sem
    ci_hi = mean_dev + 1.96 * sem
    t_stat, p_val = stats.ttest_1samp(vals, 0)
    # Bonferroni correction
    sig = "***" if p_val < 0.001 / n_tests_p else "**" if p_val < 0.01 / n_tests_p else "*" if p_val < 0.05 / n_tests_p else ""

    pitcher_agg[m] = {'mean': mean_dev, 'ci_lo': ci_lo, 'ci_hi': ci_hi,
                      't': t_stat, 'p': p_val, 'n': len(vals), 'sig': sig}

    print(f"  {m:<22} {mean_dev:>+10.4f} [{ci_lo:>+8.4f}, {ci_hi:>+8.4f}] {t_stat:>8.2f} {p_val:>10.4f} {sig:>5}")


# ── 5b: Batter deviations ──────────────────────────────────────────────

BATTER_METRICS = ['k_pct', 'bb_pct', 'swing_pct', 'chase_pct', 'contact_pct',
                  'fps_swing_pct', 'zone_swing_pct']

print(f"\n  5b: Batter deviations (primary situation)")

all_batter_devs = []

for year in range(2015, 2025):
    primary_path = f"{BASE_DIR}/data/statcast/primary_{year}.csv"
    if not os.path.exists(primary_path):
        continue

    primary = pd.read_csv(primary_path, low_memory=False)
    year_baselines = batter_baselines[batter_baselines['season'] == year]

    for bid, group in primary.groupby('batter'):
        baseline_row = year_baselines[year_baselines['batter_id'] == bid]
        if len(baseline_row) == 0:
            continue

        sit_metrics = compute_batter_metrics(group, batter_id=bid)
        # For situation sample, relax PA threshold — may have only 1 PA
        if sit_metrics is None:
            # Recompute without threshold
            sub = group[group['batter'] == bid] if bid is not None else group
            pa_rows = sub[sub['events'].notna()]
            pa_count = len(pa_rows.groupby(['game_pk', 'at_bat_number'])) if len(pa_rows) > 0 else 0
            if pa_count == 0:
                continue
            # Manually compute with no threshold
            total_pitches = len(sub)
            events = pa_rows['events']
            k_pct = (events == 'strikeout').sum() / pa_count
            bb_pct = (events == 'walk').sum() / pa_count
            swing_count = sub['description'].apply(is_swing).sum()
            swing_pct = swing_count / total_pitches if total_pitches > 0 else np.nan
            outside = sub[sub['zone'].apply(lambda z: not is_in_zone(z)) & sub['zone'].notna()]
            chase_pct = outside['description'].apply(is_swing).mean() if len(outside) > 0 else np.nan
            swings_df = sub[sub['description'].apply(is_swing)]
            contact_pct = swings_df['description'].apply(is_contact).mean() if len(swings_df) > 0 else np.nan
            first_pitches = sub[(sub['balls'] == 0) & (sub['strikes'] == 0)]
            fps_swing_pct = first_pitches['description'].apply(is_swing).mean() if len(first_pitches) > 0 else np.nan
            in_zone = sub[sub['zone'].apply(is_in_zone) & sub['zone'].notna()]
            zone_swing_pct = in_zone['description'].apply(is_swing).mean() if len(in_zone) > 0 else np.nan

            sit_metrics = {
                'pa_count': pa_count, 'pitch_count': total_pitches,
                'k_pct': k_pct, 'bb_pct': bb_pct, 'swing_pct': swing_pct,
                'chase_pct': chase_pct, 'contact_pct': contact_pct,
                'fps_swing_pct': fps_swing_pct, 'zone_swing_pct': zone_swing_pct,
            }

        baseline_row = baseline_row.iloc[0]

        dev = {
            'batter_id': bid,
            'season': year,
            'sit_pitches': sit_metrics['pitch_count'],
            'sit_pa': sit_metrics['pa_count'],
            'baseline_pa': baseline_row['pa_count'],
        }

        for m in BATTER_METRICS:
            dev[f'sit_{m}'] = sit_metrics.get(m, np.nan)
            dev[f'base_{m}'] = baseline_row.get(m, np.nan)
            sit_val = sit_metrics.get(m, np.nan)
            base_val = baseline_row.get(m, np.nan)
            if pd.notna(sit_val) and pd.notna(base_val):
                dev[f'dev_{m}'] = sit_val - base_val
            else:
                dev[f'dev_{m}'] = np.nan

        all_batter_devs.append(dev)

batter_dev_df = pd.DataFrame(all_batter_devs)
batter_dev_df.to_csv(f"{BASE_DIR}/data/deviations/statcast_batter_deviations.csv", index=False)

print(f"  Batter deviation rows: {len(batter_dev_df)} (batter-seasons)")
print(f"  Covering {batter_dev_df['season'].nunique()} seasons")

# Aggregate mean deviations with CIs
print("\n  Aggregate batter deviations (pooled across all years):")
print(f"  {'Metric':<22} {'Mean Dev':>10} {'95% CI':>20} {'t-stat':>8} {'p-value':>10} {'Sig':>5}")
print("  " + "-" * 77)

n_tests_b = len(BATTER_METRICS)
batter_agg = {}

for m in BATTER_METRICS:
    col = f'dev_{m}'
    vals = batter_dev_df[col].dropna()
    if len(vals) < 5:
        continue

    mean_dev = vals.mean()
    sem = vals.sem()
    ci_lo = mean_dev - 1.96 * sem
    ci_hi = mean_dev + 1.96 * sem
    t_stat, p_val = stats.ttest_1samp(vals, 0)
    sig = "***" if p_val < 0.001 / n_tests_b else "**" if p_val < 0.01 / n_tests_b else "*" if p_val < 0.05 / n_tests_b else ""

    batter_agg[m] = {'mean': mean_dev, 'ci_lo': ci_lo, 'ci_hi': ci_hi,
                     't': t_stat, 'p': p_val, 'n': len(vals), 'sig': sig}

    print(f"  {m:<22} {mean_dev:>+10.4f} [{ci_lo:>+8.4f}, {ci_hi:>+8.4f}] {t_stat:>8.2f} {p_val:>10.4f} {sig:>5}")


# ── 5c: Secondary situation comparison ──────────────────────────────────
print(f"\n  5c: Secondary situation (robustness check)")

sec_pitcher_devs = []
sec_batter_devs = []

for year in range(2015, 2025):
    secondary_path = f"{BASE_DIR}/data/statcast/secondary_{year}.csv"
    if not os.path.exists(secondary_path):
        continue

    secondary = pd.read_csv(secondary_path, low_memory=False)
    year_pit_baselines = pitcher_baselines[pitcher_baselines['season'] == year]
    year_bat_baselines = batter_baselines[batter_baselines['season'] == year]

    # Pitcher deviations for secondary
    for pid, group in secondary.groupby('pitcher'):
        baseline_row = year_pit_baselines[year_pit_baselines['pitcher_id'] == pid]
        if len(baseline_row) == 0:
            continue
        sit_metrics = compute_pitcher_metrics(group)
        if sit_metrics is None:
            continue
        baseline_row = baseline_row.iloc[0]
        dev = {'pitcher_id': pid, 'season': year}
        for m in PITCHER_METRICS:
            sit_val = sit_metrics.get(m, np.nan)
            base_val = baseline_row.get(m, np.nan)
            dev[f'dev_{m}'] = sit_val - base_val if pd.notna(sit_val) and pd.notna(base_val) else np.nan
        sec_pitcher_devs.append(dev)

    # Batter deviations for secondary
    for bid, group in secondary.groupby('batter'):
        baseline_row = year_bat_baselines[year_bat_baselines['batter_id'] == bid]
        if len(baseline_row) == 0:
            continue
        sub = group
        pa_rows = sub[sub['events'].notna()]
        pa_count = len(pa_rows.groupby(['game_pk', 'at_bat_number'])) if len(pa_rows) > 0 else 0
        if pa_count == 0:
            continue
        total_pitches = len(sub)
        events = pa_rows['events']
        dev = {'batter_id': bid, 'season': year}

        sit_k = (events == 'strikeout').sum() / pa_count
        sit_bb = (events == 'walk').sum() / pa_count
        sit_swing = sub['description'].apply(is_swing).sum() / total_pitches if total_pitches > 0 else np.nan

        baseline_row = baseline_row.iloc[0]
        dev['dev_k_pct'] = sit_k - baseline_row['k_pct'] if pd.notna(baseline_row['k_pct']) else np.nan
        dev['dev_bb_pct'] = sit_bb - baseline_row['bb_pct'] if pd.notna(baseline_row['bb_pct']) else np.nan
        dev['dev_swing_pct'] = sit_swing - baseline_row['swing_pct'] if pd.notna(sit_swing) and pd.notna(baseline_row['swing_pct']) else np.nan
        sec_batter_devs.append(dev)

sec_pit_df = pd.DataFrame(sec_pitcher_devs) if sec_pitcher_devs else pd.DataFrame()
sec_bat_df = pd.DataFrame(sec_batter_devs) if sec_batter_devs else pd.DataFrame()

print(f"  Secondary pitcher deviations: {len(sec_pit_df)} pitcher-seasons")
print(f"  Secondary batter deviations: {len(sec_bat_df)} batter-seasons")

if len(sec_pit_df) > 0:
    print("\n  Secondary pitcher deviations (key metrics):")
    for m in ['hard_pct', 'offspeed_pct', 'fps_rate', 'zone_rate']:
        col = f'dev_{m}'
        if col in sec_pit_df.columns:
            vals = sec_pit_df[col].dropna()
            if len(vals) >= 5:
                print(f"    {m}: mean={vals.mean():+.4f}, n={len(vals)}")

if len(sec_bat_df) > 0:
    print("\n  Secondary batter deviations (key metrics):")
    for m in ['k_pct', 'bb_pct', 'swing_pct']:
        col = f'dev_{m}'
        if col in sec_bat_df.columns:
            vals = sec_bat_df[col].dropna()
            if len(vals) >= 5:
                print(f"    {m}: mean={vals.mean():+.4f}, n={len(vals)}")


print(f"\n{'='*70}")
print(" STEPS 3-5 COMPLETE")
print("=" * 70)
print(f"\nOutput files:")
print(f"  data/baselines/pitcher_baselines_statcast.csv")
print(f"  data/baselines/batter_baselines_statcast.csv")
print(f"  data/deviations/statcast_pitcher_deviations.csv")
print(f"  data/deviations/statcast_batter_deviations.csv")
