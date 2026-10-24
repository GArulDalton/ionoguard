import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
RES = os.path.join(ROOT, 'results'); FIG = os.path.join(ROOT, 'figures')
os.makedirs(os.path.join(RES, 'nights'), exist_ok=True); os.makedirs(FIG, exist_ok=True)
QUICK = '--quick' in sys.argv          # smoke test: short nights
T = 1800 if QUICK else 21600
