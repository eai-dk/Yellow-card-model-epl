# 🟨 EPL Yellow Card Betting Model

**Locked Baseline v1.0** - 17th January 2026

A data-driven model for predicting EPL yellow cards with a proven edge.

## 🎯 The Edge

**Bookies set player YC odds based on historical rates but DON'T adjust for referee strictness. We exploit that.**

| Player Type | Hist Rate | Bookie Odds | Actual Rate (Strict Ref) | Edge |
|-------------|-----------|-------------|--------------------------|------|
| High-rate player | 35% | 2.5 | ~40% | 0% |
| **Low-rate DEF/MID** | 10% | **9.0** | ~15% | **+35%** 🔥 |

## 📊 Backtested Results

- **4 seasons analysed**: 26,000+ player-games
- **TIER 1 (Very Strict Refs)**: +25% ROI
- **TIER 2 (Strict Refs + Away)**: +12% ROI

### 17th Jan 2026 Live Test:
- 6/20 hits at avg odds 6.5
- **+£193 profit** on £200 staked
- **+97% ROI**

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Get today's picks
python predict.py 2026-01-18

# Or run with default (today's date)
python predict.py
```

## 📁 Project Structure

```
├── predict.py           # Main prediction script
├── data/
│   ├── complete_yc_data.csv    # Historical player data
│   └── referee_stats.csv       # Referee strictness ratings
├── config.py            # API keys and settings
└── requirements.txt     # Python dependencies
```

## ⚙️ Configuration

Set your API-Football key in `config.py`:

```python
API_KEY = "your-api-football-key"
```

## 🎰 Strategy Tiers

### TIER 1 - Very Strict Refs (5+ yellows/game)
- **Refs**: Kavanagh, Tim Robinson, S. Barrott, S. Allison
- **Target**: ALL DEF/MID from both teams
- **Expected ROI**: +25%

### TIER 2 - Strict Refs (4+ yellows/game)
- **Refs**: John Brooks, Stuart Attwell, Michael Salisbury, etc.
- **Target**: DEF/MID from AWAY team only
- **Expected ROI**: +12%

## 📱 Output Example

```
🔒 LOCKED BASELINE STRATEGY - 2026-01-17
================================================================================

🔥🔥 TIER 1 - VERY STRICT REFS (+25% ROI expected)
--------------------------------------------------------------------------------
⚽ 15:00 Leeds vs Fulham
   Ref: Chris Kavanagh 🔥🔥 VERY STRICT
   • Kenny Tete                (D) - 24%
   • Antonee Robinson          (D) - 19%
   • Harrison Reed             (M) - 16%
```

## 🔮 Future Improvements (Not Yet Implemented)

- [ ] Player vs specific opponent matchups
- [ ] Team aggression profiles
- [ ] Rivalry/derby game adjustments
- [ ] Player vs player foul history

## ⚠️ Disclaimer

This is for educational purposes. Gamble responsibly.

## 📄 License

MIT

