# SmartMusicAI


A Python-based music recommendation system that creates personalized playlists using machine learning.

## Features

- **Content-Based Filtering:** recommends tracks using artist, title, and genre information.
- **Collaborative Filtering:** finds listening patterns shared by similar users.
- **Hybrid Recommendations:** combines both methods when enough user interaction data is available.
- **CSV Export:** saves the recommended playlist in the `output/` folder.
- **ListenBrainz Import:** imports recent public listening history from ListenBrainz.

## Technologies

- Python
- Pandas and NumPy
- Scikit-learn
- SciPy
- ListenBrainz API

## Installation

```bash
python -m venv .venv
```

Activate the environment on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run

Generate five recommendations using the included demo data:

```bash
python main.py --user-id listener_1 --n 5
```

You can also provide music preferences:

```bash
python main.py --user-id new_listener --genres "hip-hop,r&b" --artists "Drake,The Weeknd" --n 20
```

The generated playlist is saved as a CSV file in `output/`.

## Import Listening Data

You can import recent public listening history from ListenBrainz:

```bash
python import_listenbrainz.py --usernames "username1,username2,username3" --count 500
```

Replace the example usernames with real public ListenBrainz usernames. **This import replaces** `data/tracks.csv` and `data/interactions.csv`, so back up the demo files first if you want to keep them.

## Data

The project includes small synthetic demo data to let you test the recommender immediately. Demo tracks and users are not real. ListenBrainz imports provide listening history; genre information may be missing from those records.

## Current Status

This is an early project version. It generates recommendations locally and exports them to CSV. Spotify playlist creation and account integration are not implemented yet.
