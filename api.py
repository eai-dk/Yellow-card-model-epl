"""
PitchPredict Yellow Card API
FastAPI application for serving YC predictions from Supabase
"""

import os
import requests
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(
    title="Yellow Card Model API",
    description="EPL Yellow Card predictions with edge calculation",
    version="2.1.0"
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
        "model": "Yellow Card v2.1 (Supabase)",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/health")
def health():
    return {"status": "healthy", "model": "yellow-card"}

@app.get("/picks/weekend")
async def weekend_picks(days_ahead: int = 7):
    """Get weekend value picks from Supabase"""
    if not SUPABASE_KEY:
        raise HTTPException(status_code=500, detail="Supabase key not configured")
    
    today = datetime.utcnow().date()
    end_date = today + timedelta(days=days_ahead)
    
    url = f"{SUPABASE_URL}/rest/v1/yc_predictions"
    params = {
        "select": "*",
        "date": f"gte.{today.isoformat()}",
        "date": f"lte.{end_date.isoformat()}",
        "order": "edge.desc"
    }
    
    try:
        resp = requests.get(url, headers=get_supabase_headers(), params=params, timeout=30)
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=f"Supabase error: {resp.text}")
        
        predictions = resp.json()
        
        # Format response
        picks = []
        for pred in predictions:
            picks.append({
                "player_name": pred.get("player", ""),
                "team": pred.get("team", ""),
                "position": pred.get("pos", ""),
                "fixture": pred.get("game", ""),
                "probability": pred.get("model_prob", 0),
                "odds": pred.get("odds", 0),
                "implied_probability": pred.get("implied_prob", 0),
                "edge": pred.get("edge", 0),
                "tier": pred.get("tier", ""),
                "fixture_date": pred.get("date", "")
            })
        
        return {
            "model": "yellow-card",
            "total_picks": len(picks),
            "picks": picks
        }
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="Supabase request timed out")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
