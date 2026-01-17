# 🔥🔥🔥 ULTIMATE Yellow Card Betting Model v2.0 - EPL

## 🎯 THE FORMULA
```
STRICT REF + AWAY TEAM + DEF/MID = 43% hit rate = +160% ROI
```

## 📊 Saturday 17th Jan 2026 Results
| Metric | Value |
|--------|-------|
| Bets | 30 |
| Hits | 13 |
| Hit Rate | **43%** |
| Profit | **+£480** |
| ROI | **+160%** 🔥🔥🔥 |

Baseline hit rate is 10.4% - we hit at **43%** = **+317% lift**

## 💡 How It Works

### The Edge
Bookies set Yellow Card odds based on player's historical YC rate.
**They DON'T adjust for:**
1. Referee strictness (+11% to +71% lift)
2. Away team bias (+7% lift)
3. High-card opponents (+19-23% lift)
4. Player vs player matchups (100% hit rates!)

### Primary Picks (AWAY + DEF/MID in strict ref games)
- Wait for refs to be announced (usually 2-3 days before)
- Check if ref is STRICT (4+ yellows/game) or VERY STRICT (5+)
- Bet on AWAY team DEF/MID players
- Target odds: 4.0 - 10.0

### Secondary Picks (HOME team, special cases only)
- If ref is VERY STRICT: add HOME DEF/MID
- If away team is HIGH-CARD (Newcastle, Villa, Chelsea, Brighton, City): add HOME DEF/MID

## 📈 Alpha Factors Discovered

### Referee Tiers
| Tier | Refs | YC/Game | Strategy |
|------|------|---------|----------|
| VERY STRICT | Allison, Robinson, Kavanagh, Barrott | 5+ | Both teams |
| STRICT | Brooks, Attwell, Salisbury, Taylor, Oliver | 4+ | Away only |

### High-Card Opponents
Playing AGAINST these teams = +42% more cards:
- Newcastle (+23%)
- Aston Villa (+19%)
- Chelsea (+17%)
- Brighton (+17%)
- Man City (+17%)

### Stacked Edge
| Combo | YC Rate | Lift |
|-------|---------|------|
| Strict + Away + DEF/MID | 14.2% | +37% |
| Strict + High-Card + DEF/MID | 17.0% | +63% |
| **Strict + High-Card + Away + DEF/MID** | **17.8%** | **+71%** 🔥 |

## 🚀 Usage

```bash
# Clone the repo
git clone https://github.com/EagleAIbot/Yellow-card-model-epl.git
cd Yellow-card-model-epl

# Install dependencies
pip install -r requirements.txt

# Add your API-Football key to config.py

# Run predictions for tomorrow
python predict.py 2026-01-18
```

## 📁 Files
- `predict.py` - Main prediction script (ULTIMATE v2.0)
- `config.py` - API key, referee tiers, high-card opponents
- `data/complete_yc_data.csv` - Historical player data (26,652 records)
- `data/referee_stats.csv` - Referee strictness stats

## 🔬 Backtest Details
- 4 EPL seasons (2022-2026)
- 26,652 player-game records
- 100% of Saturday's 23 YCs were DEF/MID
- 57% of Saturday's YCs were AWAY team

## 🏆 Head-to-Head Card Machines (100% rates)
- Ola Aina vs Bukayo Saka (3/3)
- Tyrick Mitchell vs Bruno Guimarães (3/3)
- Rodrigo Bentancur vs Bruno Guimarães (3/3)
- Moisés Caicedo vs Jack Grealish (3/3)
- Marc Cucurella vs Saka with strict ref (2/2)

## ⚠️ Disclaimer
This is for educational purposes. Gamble responsibly.

## 📡 API
Uses API-Football Pro (https://www.api-football.com/)

## 📄 License
MIT

