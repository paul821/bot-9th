"""
Session 2, Step 6: Retrosheet era batter baseline pipeline
Validates on 1990 and 2000 (years already parsed in Session 1).
"""

import pandas as pd
import numpy as np
import os
import glob
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = "/Users/paul821/Desktop/baseball/baseball_terminal_state"
RETRO_DIR = f"{BASE_DIR}/retrosheet_data"
os.makedirs(f"{BASE_DIR}/data/baselines", exist_ok=True)
os.makedirs(f"{BASE_DIR}/data/retrosheet", exist_ok=True)

# ── Retrosheet event code definitions ───────────────────────────────────
EVENT_K = 3
EVENT_BB = 14
EVENT_IBB = 15
EVENT_HBP = 16
EVENT_SINGLE = 20
EVENT_DOUBLE = 21
EVENT_TRIPLE = 22
EVENT_HR = 23
CONTACT_EVENTS = {EVENT_SINGLE, EVENT_DOUBLE, EVENT_TRIPLE, EVENT_HR}
BATTING_EVENTS = {EVENT_K, EVENT_BB, EVENT_IBB, EVENT_HBP,
                  EVENT_SINGLE, EVENT_DOUBLE, EVENT_TRIPLE, EVENT_HR,
                  2, 19}  # 2=generic out, 19=fielder's choice

# Pitch sequence characters that indicate swings
SWING_CHARS = set('SFXDELMOPQRTU')
STRIKE_CHARS = set('CSFKLMOPQRT')  # C=called, S=swing, F=foul, etc.


def load_retrosheet_year(year):
    """Load parsed Retrosheet event files for a given year."""
    # Check for pre-parsed CSV
    csv_path = f"{BASE_DIR}/data/retrosheet/events_{year}.csv"
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path, low_memory=False)

    # Look for raw event files from cwevent output
    patterns = [
        f"{RETRO_DIR}/{year}*.csv",
        f"{RETRO_DIR}/events_{year}.csv",
        f"{RETRO_DIR}/{year}*EV*.csv",
    ]

    for pat in patterns:
        files = glob.glob(pat)
        if files:
            dfs = [pd.read_csv(f, low_memory=False) for f in files]
            df = pd.concat(dfs, ignore_index=True)
            return df

    # Try parsing from .EVA/.EVN files using cwevent
    ev_files = glob.glob(f"{RETRO_DIR}/{year}*.EV*")
    ros_file = glob.glob(f"{RETRO_DIR}/TEAM{year}")

    if ev_files and ros_file:
        import subprocess
        all_rows = []
        for evf in ev_files:
            result = subprocess.run(
                ['cwevent', '-y', str(year), '-f', '0-96', evf],
                capture_output=True, text=True, cwd=RETRO_DIR
            )
            if result.returncode == 0 and result.stdout.strip():
                from io import StringIO
                chunk = pd.read_csv(StringIO(result.stdout), header=None)
                all_rows.append(chunk)

        if all_rows:
            df = pd.concat(all_rows, ignore_index=True)
            # Add cwevent column headers
            df = add_cwevent_headers(df)
            df.to_csv(csv_path, index=False)
            return df

    return None


def add_cwevent_headers(df):
    """Add standard cwevent column names (fields 0-96)."""
    # Standard cwevent field names for fields 0-96
    headers = [
        'GAME_ID', 'AWAY_TEAM_ID', 'INN_CT', 'BAT_HOME_ID', 'OUTS_CT',
        'BALLS_CT', 'STRIKES_CT', 'PITCH_SEQ_TX', 'AWAY_SCORE_CT', 'HOME_SCORE_CT',
        'BAT_ID', 'BAT_HAND_CD', 'RES_BAT_ID', 'RES_BAT_HAND_CD', 'PIT_ID',
        'PIT_HAND_CD', 'RES_PIT_ID', 'RES_PIT_HAND_CD', 'POS2_FLD_ID', 'POS3_FLD_ID',
        'POS4_FLD_ID', 'POS5_FLD_ID', 'POS6_FLD_ID', 'POS7_FLD_ID', 'POS8_FLD_ID',
        'POS9_FLD_ID', 'BASE1_RUN_ID', 'BASE2_RUN_ID', 'BASE3_RUN_ID', 'BAT_LINEUP_ID',
        'BAT_FLD_CD', 'EVENT_TX', 'LEADOFF_FL', 'PH_FL', 'BAT_EVENT_FL',
        'AB_FL', 'H_FL', 'SH_FL', 'SF_FL', 'EVENT_OUTS_CT',
        'DP_FL', 'TP_FL', 'RBI_CT', 'WP_FL', 'PB_FL',
        'FLD_CD', 'BATTEDBALL_CD', 'BUNT_FL', 'FOUL_FL', 'BATTEDBALL_LOC_TX',
        'ERR_CT', 'ERR1_FLD_CD', 'ERR1_CD', 'ERR2_FLD_CD', 'ERR2_CD',
        'ERR3_FLD_CD', 'ERR3_CD', 'BAT_DEST_ID', 'RUN1_DEST_ID', 'RUN2_DEST_ID',
        'RUN3_DEST_ID', 'BAT_PLAY_TX', 'RUN1_PLAY_TX', 'RUN2_PLAY_TX', 'RUN3_PLAY_TX',
        'RUN1_SB_FL', 'RUN2_SB_FL', 'RUN3_SB_FL', 'RUN1_CS_FL', 'RUN2_CS_FL',
        'RUN3_CS_FL', 'PO1_FL', 'PO2_FL', 'PO3_FL', 'ASS_CT',
        'PO_CT', 'A_CT', 'E_CT', 'BAT_TEAM_ID', 'FLD_TEAM_ID',
        'HALF_INN_ID', 'START_BAT_SCORE_CT', 'INN_RUNS_CT', 'GAME_NEW_FL', 'GAME_END_FL',
        'PR_RUN1_FL', 'PR_RUN2_FL', 'PR_RUN3_FL', 'REMOVED_FOR_PR_RUN1_ID',
        'REMOVED_FOR_PR_RUN2_ID', 'REMOVED_FOR_PR_RUN3_ID',
        'REMOVED_FOR_PH_BAT_ID', 'REMOVED_FOR_PH_BAT_FLD_CD',
        'PO1_FLD_CD', 'PO2_FLD_CD', 'PO3_FLD_CD',
        'EVENT_CD', 'BAT_SAFE_ERR_FL',
    ]

    # Pad or trim headers to match actual column count
    if len(df.columns) <= len(headers):
        df.columns = headers[:len(df.columns)]
    else:
        # More columns than expected — name what we can
        extra = [f'FIELD_{i}' for i in range(len(headers), len(df.columns))]
        df.columns = headers + extra

    return df


def filter_retrosheet_primary(df):
    """Filter to primary situation: Bot 9, 2 out, bases loaded."""
    return df[
        (df['INN_CT'] == 9) &
        (df['BAT_HOME_ID'] == 1) &
        (df['OUTS_CT'] == 2) &
        (df['BASE1_RUN_ID'].notna()) & (df['BASE1_RUN_ID'] != '') &
        (df['BASE2_RUN_ID'].notna()) & (df['BASE2_RUN_ID'] != '') &
        (df['BASE3_RUN_ID'].notna()) & (df['BASE3_RUN_ID'] != '')
    ].copy()


def compute_retro_batter_metrics(df, batter_id=None):
    """Compute batter outcome metrics from Retrosheet event data."""
    if batter_id is not None:
        df = df[df['BAT_ID'] == batter_id]

    # Only batting events
    if 'BAT_EVENT_FL' in df.columns:
        bat_events = df[df['BAT_EVENT_FL'] == 'T'].copy()
    else:
        bat_events = df.copy()

    pa_count = len(bat_events)
    if pa_count == 0:
        return None

    event_col = 'EVENT_CD'
    if event_col not in bat_events.columns:
        return None

    events = bat_events[event_col].astype(float)

    k_pct = (events == EVENT_K).sum() / pa_count
    bb_pct = (events == EVENT_BB).sum() / pa_count
    ibb_pct = (events == EVENT_IBB).sum() / pa_count
    hbp_pct = (events == EVENT_HBP).sum() / pa_count
    contact_pct = events.isin(CONTACT_EVENTS).sum() / pa_count

    result = {
        'pa_count': pa_count,
        'k_pct': k_pct,
        'bb_pct': bb_pct,
        'ibb_pct': ibb_pct,
        'hbp_pct': hbp_pct,
        'contact_pct': contact_pct,
    }

    # Pitch sequence metrics (where available)
    if 'PITCH_SEQ_TX' in bat_events.columns:
        pitch_seqs = bat_events['PITCH_SEQ_TX'].dropna()
        pitch_seqs = pitch_seqs[pitch_seqs.str.len() > 0]

        if len(pitch_seqs) > 0:
            result['pitch_seq_coverage'] = len(pitch_seqs) / pa_count

            # First-pitch strike rate
            first_chars = pitch_seqs.str[0]
            fps_strike = first_chars.isin(list(STRIKE_CHARS)).sum() / len(first_chars)
            result['fps_strike_pct'] = fps_strike

            # Swing rate (approximate from pitch sequence)
            total_pitches = 0
            total_swings = 0
            for seq in pitch_seqs:
                # Filter out non-pitch characters (>, +, ., 1, 2, 3, etc.)
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
# STEP 6: Retrosheet pipeline validation
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print(" STEP 6: Retrosheet batter baseline pipeline (validation)")
print("=" * 70)

VALIDATION_YEARS = [1990, 2000, 1960]
MIN_PA_RETRO = 50

for year in VALIDATION_YEARS:
    print(f"\n  ── {year} ──────────────────────────────────────────")

    df = load_retrosheet_year(year)
    if df is None:
        print(f"  No data found for {year}")
        continue

    print(f"  Total events: {len(df):,}")
    print(f"  Columns available: {len(df.columns)}")

    # Check key columns exist
    required = ['INN_CT', 'BAT_HOME_ID', 'OUTS_CT', 'BAT_ID',
                'BASE1_RUN_ID', 'BASE2_RUN_ID', 'BASE3_RUN_ID', 'EVENT_CD']
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"  WARNING: missing columns: {missing}")
        # Try to identify columns by position if headers are wrong
        print(f"  First 10 column names: {list(df.columns[:10])}")
        continue

    # Filter to primary situation
    primary = filter_retrosheet_primary(df)
    print(f"  Primary situation events: {len(primary)}")

    if len(primary) == 0:
        print("  No primary situation events found!")
        continue

    # Get batters in situation
    sit_batters = primary['BAT_ID'].unique()
    print(f"  Unique batters in situation: {len(sit_batters)}")

    # Build baselines: all batting events EXCLUDING situation events
    situation_mask = (
        (df['INN_CT'] == 9) &
        (df['BAT_HOME_ID'] == 1) &
        (df['OUTS_CT'] == 2) &
        (df['BASE1_RUN_ID'].notna()) & (df['BASE1_RUN_ID'] != '') &
        (df['BASE2_RUN_ID'].notna()) & (df['BASE2_RUN_ID'] != '') &
        (df['BASE3_RUN_ID'].notna()) & (df['BASE3_RUN_ID'] != '')
    )

    all_batting = df
    if 'BAT_EVENT_FL' in df.columns:
        all_batting = df[df['BAT_EVENT_FL'] == 'T']

    baseline_data = all_batting[~situation_mask]
    print(f"  Baseline batting events: {len(baseline_data):,}")

    # Compute per-batter baselines
    baselines = []
    for bid in sit_batters:
        metrics = compute_retro_batter_metrics(baseline_data, batter_id=bid)
        if metrics is None or metrics['pa_count'] < MIN_PA_RETRO:
            continue
        metrics['batter_id'] = bid
        metrics['season'] = year
        baselines.append(metrics)

    if not baselines:
        print(f"  No batters met minimum PA threshold ({MIN_PA_RETRO})")
        continue

    baseline_df = pd.DataFrame(baselines)
    baseline_df.to_csv(f"{BASE_DIR}/data/baselines/batter_baselines_retrosheet_{year}.csv", index=False)
    print(f"  Baselines computed: {len(baseline_df)} batters")

    # Pitch sequence coverage
    if 'pitch_seq_coverage' in baseline_df.columns:
        avg_cov = baseline_df['pitch_seq_coverage'].mean()
        print(f"  Pitch sequence coverage: {avg_cov:.1%}")

    # Compute situation metrics and deviations
    deviations = []
    for bid in baseline_df['batter_id']:
        base = baseline_df[baseline_df['batter_id'] == bid].iloc[0]
        sit = compute_retro_batter_metrics(primary, batter_id=bid)
        if sit is None or sit['pa_count'] == 0:
            continue

        dev = {
            'batter_id': bid, 'season': year,
            'sit_pa': sit['pa_count'], 'baseline_pa': base['pa_count'],
        }

        for m in ['k_pct', 'bb_pct', 'ibb_pct', 'contact_pct']:
            dev[f'sit_{m}'] = sit.get(m, np.nan)
            dev[f'base_{m}'] = base.get(m, np.nan)
            if pd.notna(sit.get(m)) and pd.notna(base.get(m)):
                dev[f'dev_{m}'] = sit[m] - base[m]
            else:
                dev[f'dev_{m}'] = np.nan

        for m in ['fps_strike_pct', 'swing_pct']:
            dev[f'sit_{m}'] = sit.get(m, np.nan)
            dev[f'base_{m}'] = base.get(m, np.nan)
            if pd.notna(sit.get(m)) and pd.notna(base.get(m)):
                dev[f'dev_{m}'] = sit[m] - base[m]
            else:
                dev[f'dev_{m}'] = np.nan

        deviations.append(dev)

    if deviations:
        dev_df = pd.DataFrame(deviations)
        print(f"  Deviations computed: {len(dev_df)} batter-situations")

        # Report aggregate deviations
        print(f"\n  Aggregate batter deviations ({year}):")
        for m in ['k_pct', 'bb_pct', 'ibb_pct', 'contact_pct', 'fps_strike_pct', 'swing_pct']:
            col = f'dev_{m}'
            if col in dev_df.columns:
                vals = dev_df[col].dropna()
                if len(vals) >= 3:
                    print(f"    {m}: mean={vals.mean():+.4f}, n={len(vals)}, range=[{vals.min():+.4f}, {vals.max():+.4f}]")
    else:
        print(f"  No deviations computed (no batters in both situation and baseline)")


print(f"\n{'='*70}")
print(" STEP 6 COMPLETE")
print("=" * 70)
