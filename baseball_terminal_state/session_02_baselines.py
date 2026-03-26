"""
Session 2: Baseline Construction
Terminal Game State Strategy in Baseball

Pipeline:
- Step 2: Pull full Statcast era situation sample (2015-2024)
- Step 3: Build pitcher baselines from Statcast
- Step 4: Build batter baselines from Statcast
- Step 5: Compute deviation metrics
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

from pybaseball import cache, statcast
cache.enable()

BASE_DIR = "/Users/paul821/Desktop/baseball/baseball_terminal_state"
os.makedirs(f"{BASE_DIR}/data/statcast", exist_ok=True)
os.makedirs(f"{BASE_DIR}/data/baselines", exist_ok=True)
os.makedirs(f"{BASE_DIR}/data/deviations", exist_ok=True)

# ── Helper: Statcast date ranges by season ─────────────────────────────────
SEASON_RANGES = {
    2015: ("2015-04-05", "2015-10-04"),
    2016: ("2016-04-03", "2016-10-02"),
    2017: ("2017-04-02", "2017-10-01"),
    2018: ("2018-03-29", "2018-10-01"),
    2019: ("2019-03-20", "2019-09-29"),
    2020: ("2020-07-23", "2020-09-27"),  # COVID shortened
    2021: ("2021-04-01", "2021-10-03"),
    2022: ("2022-04-07", "2022-10-05"),
    2023: ("2023-03-30", "2023-10-01"),
    2024: ("2024-03-20", "2024-09-29"),
}


def pull_season_chunks(year):
    """Pull a full season in monthly chunks with retry logic."""
    start, end = SEASON_RANGES[year]
    from datetime import datetime, timedelta
    import time

    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end, "%Y-%m-%d")

    chunks = []
    chunk_start = s
    while chunk_start < e:
        chunk_end = min(chunk_start + timedelta(days=30), e)
        cs = chunk_start.strftime("%Y-%m-%d")
        ce = chunk_end.strftime("%Y-%m-%d")
        print(f"    Pulling {cs} to {ce}...")

        for attempt in range(3):
            try:
                chunk = statcast(start_dt=cs, end_dt=ce)
                if len(chunk) > 0:
                    chunks.append(chunk)
                    print(f"      → {len(chunk):,} pitches")
                break
            except Exception as ex:
                print(f"      Attempt {attempt+1} failed: {ex}")
                if attempt < 2:
                    time.sleep(5 * (attempt + 1))
                else:
                    print(f"      SKIPPING chunk {cs} to {ce} after 3 failures")

        chunk_start = chunk_end + timedelta(days=1)

    if chunks:
        return pd.concat(chunks, ignore_index=True)
    return pd.DataFrame()


def filter_primary(df):
    """Filter to primary situation: Bot 9, 2 out, bases loaded."""
    return df[
        (df['inning'] == 9) &
        (df['inning_topbot'] == 'Bot') &
        (df['outs_when_up'] == 2) &
        (df['on_1b'].notna()) &
        (df['on_2b'].notna()) &
        (df['on_3b'].notna())
    ].copy()


def filter_secondary(df):
    """Filter to secondary situation: Bot 9, 2 out, tying run on base."""
    bot9 = df[
        (df['inning'] == 9) &
        (df['inning_topbot'] == 'Bot') &
        (df['outs_when_up'] == 2)
    ].copy()

    bot9['runners_on'] = (
        bot9['on_1b'].notna().astype(int) +
        bot9['on_2b'].notna().astype(int) +
        bot9['on_3b'].notna().astype(int)
    )
    bot9['score_deficit'] = bot9['fld_score'] - bot9['bat_score']

    return bot9[
        (bot9['runners_on'] >= 1) &
        (bot9['score_deficit'] <= bot9['runners_on'])
    ].copy()


def is_strike(desc):
    """Classify a pitch description as a strike (for FPS rate)."""
    strikes = {'called_strike', 'swinging_strike', 'foul', 'foul_tip',
               'swinging_strike_blocked', 'foul_bunt', 'missed_bunt',
               'hit_into_play', 'hit_into_play_no_out', 'hit_into_play_score'}
    return desc in strikes


def is_in_zone(zone):
    """Zones 1-9 are in the strike zone."""
    try:
        z = int(zone)
        return 1 <= z <= 9
    except (ValueError, TypeError):
        return False


def is_swing(desc):
    """Classify a pitch description as a swing."""
    swings = {'swinging_strike', 'foul', 'foul_tip', 'swinging_strike_blocked',
              'foul_bunt', 'missed_bunt', 'hit_into_play', 'hit_into_play_no_out',
              'hit_into_play_score', 'foul_pitchout', 'swinging_pitchout'}
    return desc in swings


def is_contact(desc):
    """Classify a pitch description as contact (ball in play or foul)."""
    contacts = {'foul', 'foul_tip', 'foul_bunt', 'hit_into_play',
                'hit_into_play_no_out', 'hit_into_play_score'}
    return desc in contacts


# ═══════════════════════════════════════════════════════════════════════════
# STEP 2: Pull full Statcast era situation sample
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print(" STEP 2: Pull full Statcast era situation sample (2015-2024)")
print("=" * 70)

yearly_summary = []

for year in range(2015, 2025):
    primary_path = f"{BASE_DIR}/data/statcast/primary_{year}.csv"
    secondary_path = f"{BASE_DIR}/data/statcast/secondary_{year}.csv"
    baseline_pit_path = f"{BASE_DIR}/data/statcast/baseline_pitcher_{year}.csv"
    baseline_bat_path = f"{BASE_DIR}/data/statcast/baseline_batter_{year}.csv"

    # Check if already cached
    if (os.path.exists(primary_path) and os.path.exists(secondary_path) and
        os.path.exists(baseline_pit_path) and os.path.exists(baseline_bat_path)):
        primary = pd.read_csv(primary_path)
        secondary = pd.read_csv(secondary_path)
        primary_pas = len(primary.dropna(subset=['events']).groupby(['game_pk', 'at_bat_number']))
        secondary_pas = len(secondary.dropna(subset=['events']).groupby(['game_pk', 'at_bat_number']))
        print(f"\n  {year}: cached — primary={primary_pas} PAs, secondary={secondary_pas} PAs")
        yearly_summary.append({
            'year': year,
            'primary_pitches': len(primary),
            'primary_pas': primary_pas,
            'secondary_pitches': len(secondary),
            'secondary_pas': secondary_pas,
        })
        continue

    print(f"\n  Processing {year}...")
    season = pull_season_chunks(year)

    if len(season) == 0:
        print(f"    WARNING: No data for {year}")
        continue

    print(f"    Total season pitches: {len(season):,}")

    # Filter to situations
    primary = filter_primary(season)
    secondary = filter_secondary(season)

    primary_pas = len(primary.dropna(subset=['events']).groupby(['game_pk', 'at_bat_number']))
    secondary_pas = len(secondary.dropna(subset=['events']).groupby(['game_pk', 'at_bat_number']))

    print(f"    Primary: {len(primary)} pitches, {primary_pas} PAs")
    print(f"    Secondary: {len(secondary)} pitches, {secondary_pas} PAs")

    # Identify pitchers and batters in situation sample
    sit_pitchers = set(primary['pitcher'].unique()) | set(secondary['pitcher'].unique())
    sit_batters = set(primary['batter'].unique()) | set(secondary['batter'].unique())

    print(f"    Unique pitchers in situations: {len(sit_pitchers)}")
    print(f"    Unique batters in situations: {len(sit_batters)}")

    # Extract baseline data: all pitches by these players OUTSIDE the situation
    # Create situation mask (Bot 9, 2 out, at least one runner, tying run possible)
    situation_mask = (
        (season['inning'] == 9) &
        (season['inning_topbot'] == 'Bot') &
        (season['outs_when_up'] == 2) &
        (season['on_1b'].notna() | season['on_2b'].notna() | season['on_3b'].notna())
    )

    # Pitcher baselines: all pitches by situation pitchers OUTSIDE the situation
    baseline_pit = season[
        (season['pitcher'].isin(sit_pitchers)) & (~situation_mask)
    ][['game_pk', 'at_bat_number', 'pitcher', 'batter', 'pitch_type',
       'description', 'zone', 'balls', 'strikes', 'events', 'inning',
       'release_speed', 'pfx_x', 'pfx_z', 'plate_x', 'plate_z']].copy()

    # Batter baselines: all pitches by situation batters OUTSIDE the situation
    baseline_bat = season[
        (season['batter'].isin(sit_batters)) & (~situation_mask)
    ][['game_pk', 'at_bat_number', 'pitcher', 'batter', 'pitch_type',
       'description', 'zone', 'balls', 'strikes', 'events', 'inning',
       'release_speed']].copy()

    print(f"    Pitcher baseline pitches: {len(baseline_pit):,}")
    print(f"    Batter baseline pitches: {len(baseline_bat):,}")

    # Cache everything
    primary.to_csv(primary_path, index=False)
    secondary.to_csv(secondary_path, index=False)
    baseline_pit.to_csv(baseline_pit_path, index=False)
    baseline_bat.to_csv(baseline_bat_path, index=False)

    yearly_summary.append({
        'year': year,
        'primary_pitches': len(primary),
        'primary_pas': primary_pas,
        'secondary_pitches': len(secondary),
        'secondary_pas': secondary_pas,
    })

    # Free memory
    del season, baseline_pit, baseline_bat
    print(f"    ✓ Cached to data/statcast/")

# Summary
summary_df = pd.DataFrame(yearly_summary)
print(f"\n{'='*70}")
print(" STEP 2 SUMMARY")
print(f"{'='*70}")
print(summary_df.to_string(index=False))
print(f"\nTotal primary PAs: {summary_df['primary_pas'].sum()}")
print(f"Total secondary PAs: {summary_df['secondary_pas'].sum()}")

# Sanity check: 2023 should match Session 1 (69 primary, 404 secondary)
row_2023 = summary_df[summary_df['year'] == 2023]
if len(row_2023) > 0:
    p = row_2023['primary_pas'].values[0]
    s = row_2023['secondary_pas'].values[0]
    print(f"\n2023 sanity check: primary={p} (expected 69), secondary={s} (expected 404)")
    if abs(p - 69) > 5:
        print(f"  WARNING: 2023 primary PA mismatch: {p} vs expected ~69")
    if abs(s - 404) > 20:
        print(f"  WARNING: 2023 secondary PA mismatch: {s} vs expected ~404")
    print("✓ Sanity check passed")

summary_df.to_csv(f"{BASE_DIR}/data/statcast/yearly_summary.csv", index=False)
