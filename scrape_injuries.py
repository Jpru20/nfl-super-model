import requests
from bs4 import BeautifulSoup

def scrape_cbs_injuries():
    """
    Scrapes CBS Sports for NFL injuries.
    Returns: {'TEAM_ABBR': {'Player Name': 'Status', 'Player Name': 'Status'}}
    """
    url = "https://www.cbssports.com/nfl/injuries/"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    print("   [Scraper] Connecting to CBS Sports...")
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        team_map = {
            "Arizona": "ARI", "Atlanta": "ATL", "Baltimore": "BAL", "Buffalo": "BUF",
            "Carolina": "CAR", "Chicago": "CHI", "Cincinnati": "CIN", "Cleveland": "CLE",
            "Dallas": "DAL", "Denver": "DEN", "Detroit": "DET", "Green Bay": "GB",
            "Houston": "HOU", "Indianapolis": "IND", "Jacksonville": "JAX", "Kansas City": "KC",
            "Las Vegas": "LV", "L.A. Chargers": "LAC", "Los Angeles Chargers": "LAC",
            "L.A. Rams": "LA", "Los Angeles Rams": "LA", "Miami": "MIA", "Minnesota": "MIN",
            "New England": "NE", "New Orleans": "NO", "N.Y. Giants": "NYG", "New York Giants": "NYG",
            "N.Y. Jets": "NYJ", "New York Jets": "NYJ", "Philadelphia": "PHI", "Pittsburgh": "PIT",
            "San Francisco": "SF", "Seattle": "SEA", "Tampa Bay": "TB", "Tennessee": "TEN",
            "Washington": "WAS",
            "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL",
            "Buffalo Bills": "BUF", "Carolina Panthers": "CAR", "Chicago Bears": "CHI",
            "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE", "Dallas Cowboys": "DAL",
            "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
            "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX",
            "Kansas City Chiefs": "KC", "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC",
            "Los Angeles Rams": "LA", "Miami Dolphins": "MIA", "Minnesota Vikings": "MIN",
            "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
            "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT",
            "San Francisco": "SF", "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB",
            "Tennessee Titans": "TEN", "Washington Commanders": "WAS"
        }

        injury_dict = {}
        teams_found = 0

        for team_link in soup.find_all('a', href=True):
            if '/nfl/teams/' in team_link['href']:
                raw_name = team_link.text.strip()
              if raw_name in team_map:
                    abbr = team_map[raw_name]
                    if abbr not in injury_dict:
                        injury_dict[abbr] = {} # Changed from list to dict
                        teams_found += 1

                    table = team_link.find_next('table')

                    if table:
                        rows = table.find_all('tr')
                        for row in rows:
                            cols = row.find_all('td')
                            if len(cols) >= 2:
                                # FIX: Find the <a> tag inside the cell to get the clean name
                                player_link = cols[0].find('a')
                                if player_link:
                                    player_name = player_link.text.strip()
                                else:
                                    # Fallback: Split by newline if <a> is missing
                                    player_name = cols[0].text.strip().split('\n')[0]

                                # Check Status
                                status_text = row.text.lower()

                                # Determine specific status for weighting
                                current_status = None
                                if any(x in status_text for x in ['out', 'injured reserve', 'ir', 'suspended']):
                                    current_status = 'OUT'
                                elif 'doubtful' in status_text:
                                    current_status = 'DOUBTFUL'
                                elif any(x in status_text for x in ['questionable', 'did not participate', 'dnp', 'no practice']):
                                    current_status = 'QUESTIONABLE'

                                if current_status:
                                    injury_dict[abbr][player_name] = current_status

        print(f"   [Scraper] Successfully identified {teams_found} teams.")
        total_injuries = sum(len(v) for v in injury_dict.values())
        print(f"   [Scraper] Found {total_injuries} flagged players.")
        return injury_dict

    except Exception as e:
        print(f"   [Scraper Error] {e}")
        return {}

if __name__ == "__main__":
    data = scrape_cbs_injuries()
    print("\n--- SAMPLE DATA (DETROIT) ---")
    print(data.get('DET', 'No injuries found'))
                    abbr = team_map[raw_name]
