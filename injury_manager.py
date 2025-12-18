# injury_manager.py
import os
import json
import time
from datetime import datetime, time as dt_time
import pytz
import scrape_injuries  # Your existing scraper file

# --- CONFIGURATION ---
DATA_FOLDER = 'data'
CACHE_FILE = os.path.join(DATA_FOLDER, 'injury_cache.json')
TIMEZONE = pytz.timezone('US/Eastern')

# The 4 Daily Schedule Times (Hours, Minutes)
SCHEDULE_TIMES = [
    (8, 0),   # 8:00 AM EST
    (12, 30), # 12:30 PM EST
    (15, 30), # 3:30 PM EST
    (19, 30)  # 7:30 PM EST
]

def get_last_scheduled_checkpoint():
    """
    Determines the most recent scheduled run time relative to NOW.
    Example: If it's 2:00 PM, the last checkpoint was 12:30 PM.
    """
    now = datetime.now(TIMEZONE)
    current_time = now.time()
    
    # Find the latest schedule time that has already passed today
    last_checkpoint = None
    for h, m in SCHEDULE_TIMES:
        sched_t = dt_time(h, m)
        if current_time >= sched_t:
            last_checkpoint = now.replace(hour=h, minute=m, second=0, microsecond=0)
    
    # If NO scheduled time has passed today (e.g., it's 7:00 AM), 
    # the last checkpoint was yesterday at 19:30.
    if last_checkpoint is None:
        last_h, last_m = SCHEDULE_TIMES[-1]
        yesterday = now.date().toordinal() - 1
        last_checkpoint = datetime.fromordinal(yesterday).replace(
            hour=last_h, minute=last_m, second=0, microsecond=0, tzinfo=TIMEZONE
        )
        # Re-attach timezone if replace dropped it (datetime quirk handling)
        if last_checkpoint.tzinfo is None:
            last_checkpoint = TIMEZONE.localize(last_checkpoint)
            
    return last_checkpoint

def get_injury_data():
    """
    The main entry point. Returns the injury dictionary.
    Decides whether to Load from Cache or Scrape Fresh.
    """
    # 1. Ensure Data Folder Exists
    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)
        
    # 2. Check if Cache Exists
    needs_update = False
    
    if not os.path.exists(CACHE_FILE):
        print("   [Injury Manager] No cache found. Scraping needed.")
        needs_update = True
    else:
        # 3. Check Timestamp
        # Get file modification time (in system local, convert to EST)
        mtime = os.path.getmtime(CACHE_FILE)
        file_time = datetime.fromtimestamp(mtime).astimezone(TIMEZONE)
        
        last_checkpoint = get_last_scheduled_checkpoint()
        
        # DEBUG:
        # print(f"DEBUG: File Time: {file_time.strftime('%H:%M')}")
        # print(f"DEBUG: Last Checkpoint: {last_checkpoint.strftime('%H:%M')}")
        
        if file_time < last_checkpoint:
            print(f"   [Injury Manager] Cache is stale (Last update: {file_time.strftime('%H:%M')} < Schedule: {last_checkpoint.strftime('%H:%M')}).")
            needs_update = True
        else:
            print("   [Injury Manager] Cache is fresh. Loading from file.")
    
    # 4. Execute Logic
    if needs_update:
        return run_scraper_and_save()
    else:
        return load_from_cache()

def run_scraper_and_save():
    print("   [Injury Manager] 🔄 RUNNING SCRAPER (This happens 4x/day)...")
    try:
        # Call your existing scraper function
        data = scrape_injuries.scrape_cbs_injuries()
        
        # Save to JSON
        with open(CACHE_FILE, 'w') as f:
            json.dump(data, f, indent=4)
            
        print(f"   [Injury Manager] ✅ Data saved to {CACHE_FILE}")
        return data
    except Exception as e:
        print(f"   [Injury Manager] ❌ Scraping Failed: {e}")
        # If scraping fails, try to load old cache as fallback
        return load_from_cache()

def load_from_cache():
    if not os.path.exists(CACHE_FILE):
        return {}
        
    try:
        with open(CACHE_FILE, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"   [Injury Manager] Error reading cache: {e}")
        return {}

# Test block
if __name__ == "__main__":
    data = get_injury_data()
    print(f"Loaded {len(data)} teams.")
