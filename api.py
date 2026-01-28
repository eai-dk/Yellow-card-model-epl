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
    today = datetime.now().strftime("%Y-%m-%d")
    end_date = (datetime.now() + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
    
    # Read from Supabase
    supabase_url = f"{SUPABASE_URL}/rest/v1/yc_predictions"
    params = f"?fixture_date=gte.{today}&fixture_date=lte.{end_date}&order=edge.desc"
    
    try:
        resp = requests.get(
            supabase_url + params,
            headers=get_supabase_headers(),
            timeout=10
        )
        
        if resp.status_code != 200:
            raise HTTPException(status_code=500, detail=f"Supabase error: {resp.text}")
        
        predictions = resp.json()
        
        # Format for frontend
        picks = []
        for pred in predictions:
            picks.append({
                "player_name": pred.get("player_name"),
                "team": pred.get("team"),
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
            "strategy": "STRICT REF + AWAY DEF/MID = 43% hit rate",
            "generated_at": predictions[0].get("generated_at") if predictions else None
        }
        
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="Database timeout")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predictions/{date}")
def predictions(date: str):
    """Get predictions for a specific date"""
    supabase_url = f"{SUPABASE_URL}/rest/v1/yc_predictions"
    params = f"?fixture_date=eq.{date}&order=edge.desc"
    
    try:
        resp = requests.get(
            supabase_url + params,
            headers=get_supabase_headers(),
            timeout=10
        )
        
        if resp.status_code != 200:
            raise HTTPException(status_code=500, detail=f"Supabase error: {resp.text}")
        
        predictions = resp.json()
        
        return {
            "date": date,
            "total_picks": len(predictions),
            "picks": predictions,
            "strategy": "STRICT REF + AWAY DEF/MID = 43% hit rate"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 5000))
    uvicorn.run(app, host="0.0.0.0", port=port)