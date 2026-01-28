"""
PitchPredict Yellow Card API
FastAPI application for serving YC predictions from CSV file
"""

import os
import csv
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

@app.get("/")
def root():
    return {
        "status": "ok",
        "model": "Yellow Card v2.2 (CSV)",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/health")
def health():
    return {"status": "healthy", "model": "yellow-card"}

@app.get("/picks/weekend")
async def weekend_picks(days_ahead: int = 7):
    """Get weekend value picks from CSV file"""
    try:
        picks = []
        csv_path = "weekend_yc_picks.csv"
        
        if not os.path.exists(csv_path):
            return {"model": "yellow-card", "total_picks": 0, "picks": []}
        
        today = datetime.now().date()
        end_date = today + timedelta(days=days_ahead)
        
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    pick_date = datetime.strptime(row.get("date", ""), "%Y-%m-%d").date()
                    if today <= pick_date <= end_date:
                        picks.append({
                            "player_name": row.get("player", ""),
                            "team": row.get("team", ""),
                            "position": row.get("pos", ""),
                            "fixture": row.get("game", ""),
                            "referee": row.get("ref", ""),
                            "probability": float(row.get("model_prob", 0)),
                            "odds": float(row.get("odds", 0)),
                            "implied_probability": float(row.get("implied_prob", 0)),
                            "edge": float(row.get("edge", 0)),
                            "tier": row.get("tier", ""),
                            "fixture_date": row.get("date", "")
                        })
                except (ValueError, KeyError):
                    continue
        
        # Sort by edge descending
        picks.sort(key=lambda x: x.get("edge", 0), reverse=True)
        
        return {
            "model": "yellow-card",
            "total_picks": len(picks),
            "picks": picks
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
