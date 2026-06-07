import json
from collections import defaultdict
from datetime import datetime

# Read hypotheses
hypotheses = []
with open(r'C:\Users\point\projects\polymarketClaude\hermes-trading\state\hypotheses.jsonl', 'r') as f:
    for line in f:
        if line.strip():
            hypotheses.append(json.loads(line))

# Determine source file
import os
default_path = r'C:\Users\point\projects\polymarketClaude\hermes-trading\state\trades.jsonl'
live_path = r'C:\Users\point\projects\polymarketClaude\hermes-trading\hermes_trades_live.jsonl'
source_path = live_path if os.path.exists(live_path) else default_path

# Read trades — deduplicate by (trade_id, symbol), keeping last record
# This handles historical trade_id collisions where different symbols shared the same ID
trade_records = {}
with open(source_path, 'r') as f:
    for line in f:
        if line.strip():
            t = json.loads(line)
            key = (t.get('trade_id'), t.get('symbol'))
            trade_records[key] = t

trades = list(trade_records.values())
closed_trades = [t for t in trades if t['status'] == 'closed']
open_trades = [t for t in trades if t['status'] == 'open']

print(f"Total trades: {len(trades)}")
print(f"Closed trades: {len(closed_trades)}")
print(f"Open trades: {len(open_trades)}")
print()

# Fee-aware vs legacy trade performance split
print("## 1. Fee-Aware vs Legacy Trade Performance Split")
print()

fee_aware = [t for t in closed_trades if 'fee_usd' in t]
legacy = [t for t in closed_trades if 'fee_usd' not in t]

def print_split_section(label, dataset):
    if not dataset:
        print(f"No {label} trades.\n")
        return
    count = len(dataset)
    gross = sum(t.get('pnl_usd_gross', t.get('pnl_usd', 0)) for t in dataset)
    fees = sum(t.get('fee_usd', 0) for t in dataset)
    net = sum(t.get('pnl_usd', 0) for t in dataset)
    wins = sum(1 for t in dataset if t.get('pnl_usd', 0) > 0)
    wr = wins / count * 100 if count else 0
    avg = net / count if count else 0
    max_loss = min(t.get('pnl_usd', 0) for t in dataset)
    max_win = max(t.get('pnl_usd', 0) for t in dataset)
    print(f"### {label} ({count} trades)")
    print(f"  Gross P&L:  ${gross:+.2f}")
    print(f"  Fees:       ${fees:+.2f}")
    print(f"  Net P&L:    ${net:+.2f}")
    print(f"  Win Rate:   {wins}/{count} = {wr:.1f}%")
    print(f"  Avg P&L:    ${avg:+.2f}")
    print(f"  Max Loss:   ${max_loss:+.2f}")
    print(f"  Max Win:    ${max_win:+.2f}")
    print()

print_split_section("Legacy (no fee data)", legacy)
print_split_section("Fee-aware (post-Jun 6)", fee_aware)
print()

# 1. Score trajectory over versions
print("## 1. Score Trajectory Over Versions")
print()
print("| Version | score_before | Reflector | Variable Changed | Trades Considered |")
print("|---------|-------------|-----------|------------------|-------------------|")
for h in hypotheses:
    print(f"| v{h['from_version']} | {h['score_before']:.4f} | {h['reflector']} | {h['variable_changed']} | {h['num_trades_considered']} |")

print()
print("Note: score_before values refer to the score BEFORE the reflection that created the next version.")
print("Missing scores for some versions indicate gaps in the hypothesis log (not a bug, just incomplete logging).")
print()


# 2. Cumulative P&L over time
closed_sorted = sorted(closed_trades, key=lambda x: x['exit_time'])
cumulative = 0
cum_pnl = []
for t in closed_sorted:
    cumulative += t['pnl_usd']
    cum_pnl.append((t['exit_time'], cumulative, t['pnl_usd'], t['strategy_version']))

print("## 2. Cumulative Realized P&L Over Time")
print()
print("| # | Exit Time | Version | Trade P&L | Cumulative P&L |")
print("|---|-----------|---------|-----------|----------------|")
for i, (et, cum, pnl, ver) in enumerate(cum_pnl, 1):
    print(f"| {i} | {et[:19]} | {ver} | ${pnl:+.2f} | ${cum:+.2f} |")
print()

# 3. Rolling win rate (window of 5)
print("## 3. Rolling Win Rate (Window of 5 Trades)")
print()
wins = [1 if t['pnl_usd'] > 0 else 0 for t in closed_sorted]
print("| Window | Trades | Wins | Win Rate | Versions in Window |")
print("|--------|--------|------|----------|--------------------|")
for i in range(len(wins)):
    if i < 4:
        continue
    window = wins[i-4:i+1]
    w = sum(window)
    vers = [t['strategy_version'] for t in closed_sorted[i-4:i+1]]
    print(f"| {i-3} | {i-4+1}-{i+1} | {w}/5 | {w*100/5:.1f}% | {', '.join(set(vers))} |")
print()

# Overall win rate
overall_wr = sum(wins) / len(wins) * 100
print(f"Overall win rate: {sum(wins)}/{len(wins)} = {overall_wr:.1f}%")
print()

# 4. Version-by-version comparison
print("## 4. Version-by-Version Comparison (Closed Trades Only)")
print()
ver_stats = defaultdict(lambda: {'count':0, 'wins':0, 'total_pnl':0, 'avg_pnl':0, 'trades':[]})
for t in closed_sorted:
    v = t['strategy_version']
    ver_stats[v]['count'] += 1
    if t['pnl_usd'] > 0:
        ver_stats[v]['wins'] += 1
    ver_stats[v]['total_pnl'] += t['pnl_usd']
    ver_stats[v]['trades'].append(t['pnl_usd'])

print("| Version | Closed Trades | Wins | Win Rate | Total P&L | Avg P&L/Trade | Max Loss | Max Win |")
print("|---------|---------------|------|----------|-----------|---------------|----------|---------|")
for v in sorted(ver_stats.keys()):
    s = ver_stats[v]
    wr = s['wins']/s['count']*100 if s['count'] else 0
    avg = s['total_pnl']/s['count'] if s['count'] else 0
    max_loss = min(s['trades'])
    max_win = max(s['trades'])
    print(f"| {v} | {s['count']} | {s['wins']}/{s['count']} | {wr:.1f}% | ${s['total_pnl']:+.2f} | ${avg:+.2f} | ${max_loss:+.2f} | ${max_win:+.2f} |")
print()

# 5. AI vs Fallback comparison
print("## 5. AI (claude) vs Fallback Reflector Comparison")
print()
# Map versions to reflector
version_to_reflector = {}
for h in hypotheses:
    version_to_reflector[h['to_version']] = h['reflector']
# v01 is baseline (no reflector), v02 from fallback, v03 unknown (not in hypotheses, but it was after v02),
# v04 from claude, v05 unknown, v06 unknown, v07 from claude, v08 unknown, v09 unknown
# Actually from hypotheses: v02=fallback, v04=claude, v07=claude
# We can't know v03, v05, v06, v08, v09 reflector from this file alone.
# But we can compare versions we know.
ai_versions = ['04', '07']
fb_versions = ['02']
unknown_versions = ['01', '03', '05', '06', '08', '09']

def aggregate_versions(versions):
    count = 0
    wins = 0
    total = 0
    for v in versions:
        if v in ver_stats:
            count += ver_stats[v]['count']
            wins += ver_stats[v]['wins']
            total += ver_stats[v]['total_pnl']
    return count, wins, total

ai_count, ai_wins, ai_total = aggregate_versions(ai_versions)
fb_count, fb_wins, fb_total = aggregate_versions(fb_versions)
unk_count, unk_wins, unk_total = aggregate_versions(unknown_versions)

print("| Reflector | Versions | Trades | Wins | Win Rate | Total P&L | Avg P&L |")
print("|-----------|----------|--------|------|----------|-----------|---------|")
if fb_count:
    print(f"| fallback | {', '.join(fb_versions)} | {fb_count} | {fb_wins}/{fb_count} | {fb_wins/fb_count*100:.1f}% | ${fb_total:+.2f} | ${fb_total/fb_count:+.2f} |")
if ai_count:
    print(f"| claude | {', '.join(ai_versions)} | {ai_count} | {ai_wins}/{ai_count} | {ai_wins/ai_count*100:.1f}% | ${ai_total:+.2f} | ${ai_total/ai_count:+.2f} |")
if unk_count:
    print(f"| unknown | {', '.join(unknown_versions)} | {unk_count} | {unk_wins}/{unk_count} | {unk_wins/unk_count*100:.1f}% | ${unk_total:+.2f} | ${unk_total/unk_count:+.2f} |")
print()

# 6. Directional bias analysis
print("## 6. Directional Bias Analysis (Long vs Short)")
print()
long_trades = [t for t in closed_sorted if t.get('direction', 'long') == 'long']
short_trades = [t for t in closed_sorted if t.get('direction') == 'short']

def analyze_direction(trades, name):
    if not trades:
        return
    count = len(trades)
    wins = sum(1 for t in trades if t['pnl_usd'] > 0)
    total = sum(t['pnl_usd'] for t in trades)
    avg = total/count
    wr = wins/count*100
    print(f"{name}: {count} trades, {wins}/{count} wins ({wr:.1f}%), Total P&L ${total:+.2f}, Avg ${avg:+.2f}")

analyze_direction(long_trades, "Long")
analyze_direction(short_trades, "Short")
print()

# Break down by version and direction
print("| Version | Direction | Trades | Wins | Win Rate | Total P&L | Avg P&L |")
print("|---------|-----------|--------|------|----------|-----------|---------|")
for v in sorted(ver_stats.keys()):
    longs = [t for t in closed_sorted if t['strategy_version'] == v and t.get('direction','long') == 'long']
    shorts = [t for t in closed_sorted if t['strategy_version'] == v and t.get('direction') == 'short']
    for trades, name in [(longs, 'long'), (shorts, 'short')]:
        if trades:
            count = len(trades)
            wins = sum(1 for t in trades if t['pnl_usd'] > 0)
            total = sum(t['pnl_usd'] for t in trades)
            avg = total/count
            wr = wins/count*100
            print(f"| {v} | {name} | {count} | {wins}/{count} | {wr:.1f}% | ${total:+.2f} | ${avg:+.2f} |")
print()

# 7. Exit reason analysis
print("## 7. Exit Reason Analysis")
print()
exit_reasons = defaultdict(lambda: {'count':0, 'wins':0, 'total':0})
for t in closed_sorted:
    r = t.get('exit_reason', 'unknown')
    exit_reasons[r]['count'] += 1
    if t['pnl_usd'] > 0:
        exit_reasons[r]['wins'] += 1
    exit_reasons[r]['total'] += t['pnl_usd']

print("| Exit Reason | Count | Wins | Win Rate | Total P&L | Avg P&L |")
print("|-------------|-------|------|----------|-----------|---------|")
for r in sorted(exit_reasons.keys()):
    s = exit_reasons[r]
    wr = s['wins']/s['count']*100
    avg = s['total']/s['count']
    print(f"| {r} | {s['count']} | {s['wins']}/{s['count']} | {wr:.1f}% | ${s['total']:+.2f} | ${avg:+.2f} |")
print()

# Symbol analysis
print("## Symbol Performance")
print()
sym_stats = defaultdict(lambda: {'count':0, 'wins':0, 'total':0})
for t in closed_sorted:
    sym = t['symbol']
    sym_stats[sym]['count'] += 1
    if t['pnl_usd'] > 0:
        sym_stats[sym]['wins'] += 1
    sym_stats[sym]['total'] += t['pnl_usd']

print("| Symbol | Trades | Wins | Win Rate | Total P&L | Avg P&L |")
print("|--------|--------|------|----------|-----------|---------|")
for sym in sorted(sym_stats.keys()):
    s = sym_stats[sym]
    wr = s['wins']/s['count']*100
    avg = s['total']/s['count']
    print(f"| {sym} | {s['count']} | {s['wins']}/{s['count']} | {wr:.1f}% | ${s['total']:+.2f} | ${avg:+.2f} |")
print()

# Open trades summary
print("## Open Trades Summary")
print(f"Open trades: {len(open_trades)}")
for t in open_trades:
    print(f"  {t['trade_id']}: {t['symbol']} {t.get('direction','long')} at {t['entry_price']} (v{t['strategy_version']})")
