"""
Session 2, Step 6: Parse Retrosheet data and compute batter baselines.
Handles the actual directory structure from Session 1.
"""

import pandas as pd
import numpy as np
import os
import glob
import subprocess
from io import StringIO
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = "/Users/paul821/Desktop/baseball/baseball_terminal_state"
RETRO_DIR = f"{BASE_DIR}/retrosheet_data"
os.makedirs(f"{BASE_DIR}/data/retrosheet", exist_ok=True)
os.makedirs(f"{BASE_DIR}/data/baselines", exist_ok=True)

# ── cwevent column headers (fields 0-96) — from `cwevent -n` output ────
CWEVENT_HEADERS = [
    'GAME_ID', 'AWAY_TEAM_ID', 'INN_CT', 'BAT_HOME_ID', 'OUTS_CT',       # 0-4
    'BALLS_CT', 'STRIKES_CT', 'PITCH_SEQ_TX', 'AWAY_SCORE_CT', 'HOME_SCORE_CT',  # 5-9
    'BAT_ID', 'BAT_HAND_CD', 'RESP_BAT_ID', 'RESP_BAT_HAND_CD', 'PIT_ID',  # 10-14
    'PIT_HAND_CD', 'RESP_PIT_ID', 'RESP_PIT_HAND_CD', 'POS2_FLD_ID', 'POS3_FLD_ID',  # 15-19
    'POS4_FLD_ID', 'POS5_FLD_ID', 'POS6_FLD_ID', 'POS7_FLD_ID', 'POS8_FLD_ID',  # 20-24
    'POS9_FLD_ID', 'BASE1_RUN_ID', 'BASE2_RUN_ID', 'BASE3_RUN_ID', 'EVENT_TX',  # 25-29
    'LEADOFF_FL', 'PH_FL', 'BAT_FLD_CD', 'BAT_LINEUP_ID', 'EVENT_CD',     # 30-34
    'BAT_EVENT_FL', 'AB_FL', 'H_CD', 'SH_FL', 'SF_FL',                    # 35-39
    'EVENT_OUTS_CT', 'DP_FL', 'TP_FL', 'RBI_CT', 'WP_FL',                 # 40-44
    'PB_FL', 'FLD_CD', 'BATTEDBALL_CD', 'BUNT_FL', 'FOUL_FL',             # 45-49
    'BATTEDBALL_LOC_TX', 'ERR_CT', 'ERR1_FLD_CD', 'ERR1_CD', 'ERR2_FLD_CD',  # 50-54
    'ERR2_CD', 'ERR3_FLD_CD', 'ERR3_CD', 'BAT_DEST_ID', 'RUN1_DEST_ID',  # 55-59
    'RUN2_DEST_ID', 'RUN3_DEST_ID', 'BAT_PLAY_TX', 'RUN1_PLAY_TX', 'RUN2_PLAY_TX',  # 60-64
    'RUN3_PLAY_TX', 'RUN1_SB_FL', 'RUN2_SB_FL', 'RUN3_SB_FL', 'RUN1_CS_FL',  # 65-69
    'RUN2_CS_FL', 'RUN3_CS_FL', 'RUN1_PK_FL', 'RUN2_PK_FL', 'RUN3_PK_FL',  # 70-74
    'RUN1_RESP_PIT_ID', 'RUN2_RESP_PIT_ID', 'RUN3_RESP_PIT_ID',           # 75-77
    'GAME_NEW_FL', 'GAME_END_FL',                                          # 78-79
    'PR_RUN1_FL', 'PR_RUN2_FL', 'PR_RUN3_FL',                             # 80-82
    'REMOVED_FOR_PR_RUN1_ID', 'REMOVED_FOR_PR_RUN2_ID', 'REMOVED_FOR_PR_RUN3_ID',  # 83-85
    'REMOVED_FOR_PH_BAT_ID', 'REMOVED_FOR_PH_BAT_FLD_CD',                 # 86-87
    'PO1_FLD_CD', 'PO2_FLD_CD', 'PO3_FLD_CD',                            # 88-90
    'ASS1_FLD_CD', 'ASS2_FLD_CD', 'ASS3_FLD_CD', 'ASS4_FLD_CD', 'ASS5_FLD_CD',  # 91-95
    'EVENT_ID',                                                             # 96
]

# Event codes
EVENT_K = 3
EVENT_BB = 14
EVENT_IBB = 15
EVENT_HBP = 16
EVENT_SINGLE = 20
EVENT_DOUBLE = 21
EVENT_TRIPLE = 22
EVENT_HR = 23
CONTACT_EVENTS = {EVENT_SINGLE, EVENT_DOUBLE, EVENT_TRIPLE, EVENT_HR}

# Pitch sequence characters
SWING_CHARS = set('SFXDELMOPQRTU')
STRIKE_CHARS = set('CSFKLMOPQRT')


def parse_retrosheet_year(year):
    """Parse all event files for a year using cwevent."""
    cache_path = f"{BASE_DIR}/data/retrosheet/events_{year}.csv"
    if os.path.exists(cache_path):
        print(f"  Loading cached {cache_path}")
        return pd.read_csv(cache_path, low_memory=False)

    # Find event files — check subdirectory first, then main dir
    subdir = f"{RETRO_DIR}/{year}"
    if os.path.isdir(subdir):
        ev_files = sorted(glob.glob(f"{subdir}/{year}*.EV*"))
        work_dir = subdir
    else:
        ev_files = sorted(glob.glob(f"{RETRO_DIR}/{year}*.EV*"))
        work_dir = RETRO_DIR

    if not ev_files:
        print(f"  No event files found for {year}")
        return None

    print(f"  Found {len(ev_files)} event files in {work_dir}")

    all_rows = []
    for evf in ev_files:
        result = subprocess.run(
            ['cwevent', '-y', str(year), '-f', '0-96', os.path.basename(evf)],
            capture_output=True, text=True, cwd=work_dir
        )
        if result.returncode == 0 and result.stdout.strip():
            chunk = pd.read_csv(StringIO(result.stdout), header=None, quotechar='"')
            all_rows.append(chunk)

    if not all_rows:
        print(f"  cwevent produced no output for {year}")
        return None

    df = pd.concat(all_rows, ignore_index=True)

    # Assign headers
    if len(df.columns) <= len(CWEVENT_HEADERS):
        df.columns = CWEVENT_HEADERS[:len(df.columns)]
    else:
        extra = [f'FIELD_{i}' for i in range(len(CWEVENT_HEADERS), len(df.columns))]
        df.columns = CWEVENT_HEADERS + extra

    # Clean string columns — remove quotes
    for col in ['GAME_ID', 'BAT_ID', 'PIT_ID', 'BASE1_RUN_ID', 'BASE2_RUN_ID',
                'BASE3_RUN_ID', 'PITCH_SEQ_TX', 'BAT_EVENT_FL']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip('"').str.strip()

    df.to_csv(cache_path, index=False)
    print(f"  Parsed {len(df):,} events → {cache_path}")
    return df


def filter_primary_retro(df):
    """Bot 9, 2 out, bases loaded."""
    mask = (
        (df['INN_CT'] == 9) &
        (df['BAT_HOME_ID'] == 1) &
        (df['OUTS_CT'] == 2) &
        (df['BASE1_RUN_ID'].notna()) & (df['BASE1_RUN_ID'] != '') & (df['BASE1_RUN_ID'] != 'nan') &
        (df['BASE2_RUN_ID'].notna()) & (df['BASE2_RUN_ID'] != '') & (df['BASE2_RUN_ID'] != 'nan') &
        (df['BASE3_RUN_ID'].notna()) & (df['BASE3_RUN_ID'] != '') & (df['BASE3_RUN_ID'] != 'nan')
    )
    return df[mask].copy()


def compute_retro_batter_metrics(df):
    """Compute batter metrics from a Retrosheet event subset (already filtered to one batter)."""
    # Only batting events
    bat = df[df['BAT_EVENT_FL'] == 'T'] if 'BAT_EVENT_FL' in df.columns else df
    pa_count = len(bat)
    if pa_count == 0:
        return None

    events = bat['EVENT_CD'].astype(float)

    k_pct = (events == EVENT_K).sum() / pa_count
    bb_pct = (events == EVENT_BB).sum() / pa_count
    ibb_pct = (events == EVENT_IBB).sum() / pa_count
    hbp_pct = (events == EVENT_HBP).sum() / pa_count
    contact_pct = events.isin(CONTACT_EVENTS).sum() / pa_count

    result = {
        'pa_count': pa_count,
        'k_pct': k_pct, 'bb_pct': bb_pct, 'ibb_pct': ibb_pct,
        'hbp_pct': hbp_pct, 'contact_pct': contact_pct,
    }

    # Pitch sequence metrics
    if 'PITCH_SEQ_TX' in bat.columns:
        seqs = bat['PITCH_SEQ_TX'].dropna()
        seqs = seqs[(seqs != '') & (seqs != 'nan')]

        if len(seqs) > 0:
            result['pitch_seq_coverage'] = len(seqs) / pa_count

            # First-pitch strike rate
            first_chars = seqs.str[0]
            fps_strike = first_chars.isin(list(STRIKE_CHARS)).sum() / len(first_chars)
            result['fps_strike_pct'] = fps_strike

            # Swing rate
            total_pitches = 0
            total_swings = 0
            for seq in seqs:
                pitches = [c for c in seq if c.isalpha()]
                total_pitches += len(pitches)
                total_swings += sum(1 for c in pitches if c in SWING_CHARS)

            result['swing_pct'] = total_swings / total_pitches if total_pitches > 0 else np.nan
        else:
            result['pitch_seq_coverage'] = 0.0
            result['fps_strike_pct'] = np.nan
            result['swing_pct'] = np.nan
    else:
        result['pitch_seq_coverage'] = 0.0
        result['fps_strike_pct'] = np.nan
        result['swing_pct'] = np.nan

    return result


# ═══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print(" STEP 6: Retrosheet batter baseline pipeline")
print("=" * 70)

MIN_PA = 50
VALIDATION_YEARS = [2000, 1990, 1960]

for year in VALIDATION_YEARS:
    print(f"\n{'─'*60}")
    print(f"  {year}")
    print(f"{'─'*60}")

    df = parse_retrosheet_year(year)
    if df is None:
        continue

    print(f"  Total events: {len(df):,}")

    # Verify key columns
    for col in ['INN_CT', 'BAT_HOME_ID', 'OUTS_CT', 'BAT_ID', 'EVENT_CD',
                'BASE1_RUN_ID', 'BASE2_RUN_ID', 'BASE3_RUN_ID', 'PITCH_SEQ_TX']:
        present = col in df.columns
        if not present:
            print(f"  WARNING: {col} missing!")

    # Filter to primary situation
    primary = filter_primary_retro(df)
    # Only batting events in primary
    primary_bat = primary[primary['BAT_EVENT_FL'] == 'T'] if 'BAT_EVENT_FL' in primary.columns else primary
    print(f"  Primary situation batting events: {len(primary_bat)}")

    if len(primary_bat) == 0:
        continue

    sit_batters = primary_bat['BAT_ID'].unique()
    print(f"  Unique batters in primary situation: {len(sit_batters)}")

    # Build situation mask for exclusion
    sit_mask = (
        (df['INN_CT'] == 9) & (df['BAT_HOME_ID'] == 1) & (df['OUTS_CT'] == 2) &
        (df['BASE1_RUN_ID'].notna()) & (df['BASE1_RUN_ID'] != '') & (df['BASE1_RUN_ID'] != 'nan') &
        (df['BASE2_RUN_ID'].notna()) & (df['BASE2_RUN_ID'] != '') & (df['BASE2_RUN_ID'] != 'nan') &
        (df['BASE3_RUN_ID'].notna()) & (df['BASE3_RUN_ID'] != '') & (df['BASE3_RUN_ID'] != 'nan')
    )
    baseline_all = df[~sit_mask]
    print(f"  Baseline events (excluding situation): {len(baseline_all):,}")

    # Per-batter baselines
    baselines = []
    for bid in sit_batters:
        batter_data = baseline_all[baseline_all['BAT_ID'] == bid]
        metrics = compute_retro_batter_metrics(batter_data)
        if metrics is None or metrics['pa_count'] < MIN_PA:
            continue
        metrics['batter_id'] = bid
        metrics['season'] = year
        baselines.append(metrics)

    if not baselines:
        print(f"  No batters met {MIN_PA} PA threshold")
        continue

    baseline_df = pd.DataFrame(baselines)
    baseline_df.to_csv(f"{BASE_DIR}/data/baselines/batter_baselines_retrosheet_{year}.csv", index=False)
    print(f"  Baselines: {len(baseline_df)} batters (mean PA: {baseline_df['pa_count'].mean():.0f})")
    print(f"  Pitch seq coverage: {baseline_df['pitch_seq_coverage'].mean():.1%}")

    # Compute deviations
    deviations = []
    for bid in baseline_df['batter_id']:
        base = baseline_df[baseline_df['batter_id'] == bid].iloc[0]
        sit_data = primary_bat[primary_bat['BAT_ID'] == bid]
        sit = compute_retro_batter_metrics(sit_data)
        if sit is None or sit['pa_count'] == 0:
            continue

        dev = {'batter_id': bid, 'season': year,
               'sit_pa': sit['pa_count'], 'baseline_pa': base['pa_count']}

        for m in ['k_pct', 'bb_pct', 'ibb_pct', 'contact_pct', 'fps_strike_pct', 'swing_pct']:
            dev[f'sit_{m}'] = sit.get(m, np.nan)
            dev[f'base_{m}'] = base.get(m, np.nan)
            s, b = sit.get(m, np.nan), base.get(m, np.nan)
            dev[f'dev_{m}'] = s - b if pd.notna(s) and pd.notna(b) else np.nan

        deviations.append(dev)

    if deviations:
        dev_df = pd.DataFrame(deviations)
        print(f"  Deviations: {len(dev_df)} batters")

        print(f"\n  Aggregate batter deviations ({year}):")
        print(f"  {'Metric':<20} {'Mean Dev':>10} {'N':>5} {'Range':>25}")
        print(f"  {'-'*62}")
        for m in ['k_pct', 'bb_pct', 'ibb_pct', 'contact_pct', 'fps_strike_pct', 'swing_pct']:
            col = f'dev_{m}'
            if col in dev_df.columns:
                vals = dev_df[col].dropna()
                if len(vals) >= 1:
                    print(f"  {m:<20} {vals.mean():>+10.4f} {len(vals):>5} [{vals.min():>+.4f}, {vals.max():>+.4f}]")

        dev_df.to_csv(f"{BASE_DIR}/data/deviations/retrosheet_batter_deviations_{year}.csv", index=False)

print(f"\n{'='*70}")
print(" STEP 6 COMPLETE")
print("=" * 70)
