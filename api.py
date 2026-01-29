"""
PitchPredict Yellow Card API
FastAPI application for serving YC predictions from Supabase
With current squad validation to filter out transferred players
"""

import os
import json
import unicodedata
import requests
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Yellow Card Model API",
    description="EPL Yellow Card predictions with edge calculation",
    version="2.2.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Supabase config
SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://kijtxzvbvhgswpahmvua.supabase.co")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")

# Load current squad data
SQUAD_DATA = {}
SQUAD_LOOKUP = {}  # normalized_name -> team

def load_squad_data():
    """Load squad data from JSON file and build lookup"""
    global SQUAD_DATA, SQUAD_LOOKUP
    try:
        squad_path = os.path.join(os.path.dirname(__file__), "data", "epl-squads.json")
        with open(squad_path, "r", encoding="utf-8") as f:
            SQUAD_DATA = json.load(f)
        
        # Build normalized lookup: player_name -> team
        for team, players in SQUAD_DATA.items():
            for player in players:
                name = player.get("name", "")
                normalized = normalize_name(name)
                SQUAD_LOOKUP[normalized] = team
        print(f"Loaded {len(SQUAD_LOOKUP)} players from squad data")
    except Exception as e:
        print(f"Warning: Could not load squad data: {e}")

def normalize_name(name):
    """Normalize player name for comparison"""
    if not name:
        return ""
    name = unicodedata.normalize("NFKD", name).encode("ASCII", "ignore").decode("ASCII")
    name = name.lower().strip()
    parts = name.split()
    if len(parts) > 1 and len(parts[0]) <= 2 and parts[0].endswith("."):
        name = " ".join(parts[1:])
    return name

def is_player_in_current_squad(player_name, team):
    """Check if player is in the current squad for the given team"""
    if not SQUAD_LOOKUP:
        return True
    
    normalized = normalize_name(player_name)
    current_team = SQUAD_LOOKUP.get(normalized)
    
    if current_team is None:
        for squad_name, squad_team in SQUAD_LOOKUP.items():
            if normalized in squad_name or squad_name in normalized:
                current_team = squad_team
                break
    
    if current_team is None:
        return False
    
    team_normalized = team.lower().strip()
    current_team_normalized = current_team.lower().strip()
    
    return team_normalized == current_team_normalized

load_squad_data()

def get_supabase_headers():
    return {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json"
    }

@app.get("/")
def root():
    return {
        "status": "ok",
        "model": "Yellow Card v2.2 (Squad Validated)",
        "timestamp": datetime.utcnow().isoformat(),
        "squad_players_loaded": len(SQUAD_LOOKUP)
    }

@app.get("/health")
def health():
    return {"status": "healthy", "model": "yellow-card"}

@app.get("/picks/weekend")
async def weekend_picks(days_ahead: int = 7):
    """Get weekend value picks from Supabase with squad validation"""
    today = datetime.now().strftime("%Y-%m-%d")
    end_date = (datetime.now() + timedelta(days=days_ahead)).strftime("%Y-%m-%d")

    supabase_url = f"{SUPABASE_URL}/rest/v1/yc_predictions"
    params = f"?fixture_date=gte.{today}&fixture_date=lte.{end_date}&order=edge.desc"

    try:
        resp = requests.get(supabase_url + params, headers=get_supabase_headers(), timeout=10)

        if resp.status_code != 200:
            raise HTTPException(status_code=500, detail=f"Supabase error: {resp.text}")

        predictions = resp.json()
        picks = []
        filtered_count = 0

        for pred in predictions:
            player_name = pred.get("player_name")
            team = pred.get("team")
            
            if not is_player_in_current_squad(player_name, team):
                filtered_count += 1
                continue
            
            picks.append({
                "player_name": player_name,
                "team": team,
                "position": pred.get("position"),
                "fixture": pred.get("fixture"),
                "referee": pred.get("referee"),
                "probability": float(pred.get("model_probability", 0)),
                "odds": float(pred.get("odds", 0)) if pred.get("odds") else None,
                "implied_probability": float(pred.get("implied_probability", 0)) if pred.get("implied_probability") else None,
                "edge": float(pred.get("edge", 0)) if pred.get("edge") else None,
                "tier": pred.get("tier"),
                "fixture_date": pred.get("fixture_date"),
            })

        return {
            "model": "yellow-card",
            "total_picks": len(picks),
            "picks": picks,
            "filtered_transfers": filtered_count,
            "strategy": "STRICT REF + AWAY DEF/MID = 43% hit rate",
            "generated_at": predictions[0].get("generated_at") if predictions else None
        }

    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="Database timeout")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 5000))
    uvicorn.run(app, host="0.0.0.0", port=port)