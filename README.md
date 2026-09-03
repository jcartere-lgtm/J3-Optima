# Solar Energy Planner

An easy-to-use Streamlit website that finds a solar panel's recommended tilt and facing direction, strongest daily collection window, seven-day energy forecast, and projected future energy totals.

## Upload these files to GitHub

- `app.py`
- `requirements.txt`
- `README.md` (optional, but recommended)

## Run locally

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Important note

The app uses free OpenStreetMap Nominatim location search and Open-Meteo forecast data. Monthly, quarterly, biannual, and annual figures are forecast-based projections from the current seven-day weather outlook. They are planning estimates, not an engineering or installation design.
