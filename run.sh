#!/bin/bash
# KisaanVaani — Quick Start
# Usage: bash run.sh

echo "🌾 Starting KisaanVaani..."
echo ""

# Check API key
if [ -z "sk_4wzr70fm_HdvSZ2wLPMC24fVVVAYqFnoX" ]; then
  if [ -f ".env" ]; then
    export $(cat .env | xargs)
  fi
fi

if [ -z "$SARVAM_API_KEY" ]; then
  echo "⚠️  WARNING: SARVAM_API_KEY not set!"
  echo "   Create a .env file:  echo 'SARVAM_API_KEY=your_key' > .env"
  echo ""
fi

echo "🚀 Server starting at http://localhost:8000"
echo "   Open this URL in your browser to use KisaanVaani"
echo ""
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
