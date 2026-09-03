"""Solar Energy Planner - run with: python -m streamlit run app.py"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from scipy.optimize import differential_evolution


st.set_page_config(page_title="Solar Energy Planner", page_icon="☀️", layout="wide")
st.markdown("""
<style>
  .block-container {padding-top: 2rem;}
  div[data-testid="stMetric"] {background: #fff9ed; border: 1px solid #f4c96b; border-radius: 14px; padding: 16px;}
  div[data-testid="stMetricLabel"] {font-size: .95rem; font-weight: 700; color: #334155;}
  div[data-testid="stMetricValue"] {font-size: 2rem; font-weight: 800; color: #0f172a;}
</style>
""", unsafe_allow_html=True)

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
HEADERS = {"User-Agent": "SolarEnergyPlanner/1.0 (student project)"}
ALBEDO = 0.25
SYSTEM_EFFICIENCY = 0.20  # Assumed 20% panel conversion efficiency.


def location_query(country: str, state: str, city: str, postal_code: str) -> str:
    """Create a location string while allowing any of the fields to be blank."""
    return ", ".join(value.strip() for value in [city, state, postal_code, country] if value.strip())


@st.cache_data(ttl=3600, show_spinner=False)
def geocode(query: str) -> tuple[float, float, str]:
    response = requests.get(
        NOMINATIM_URL,
        params={"q": query, "format": "json", "limit": 1},
        headers=HEADERS,
        timeout=15,
    )
    response.raise_for_status()
    results = response.json()
    if not results:
        raise ValueError("We could not find that location. Check the city, state, ZIP, and country.")
    result = results[0]
    return float(result["lat"]), float(result["lon"]), result["display_name"]


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_weather(latitude: float, longitude: float) -> tuple[pd.DataFrame, str]:
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": "direct_normal_irradiance,diffuse_radiation,shortwave_radiation,cloud_cover,temperature_2m",
        "forecast_days": 7,
        "timezone": "auto",
    }
    response = requests.get(FORECAST_URL, params=params, timeout=20)
    response.raise_for_status()
    payload = response.json()
    if "hourly" not in payload:
        raise ValueError("The weather provider did not return an hourly forecast.")
    weather = pd.DataFrame(payload["hourly"])
    weather["time"] = pd.to_datetime(weather["time"])
    return weather, payload.get("timezone_abbreviation", "local time")


def demo_weather() -> pd.DataFrame:
    """A labelled fallback so the interface remains useful during an API outage."""
    times = pd.date_range(pd.Timestamp.now().floor("h"), periods=168, freq="h")
    daylight = np.clip(np.sin(np.pi * (times.hour.to_numpy() - 6) / 12), 0, None)
    return pd.DataFrame({
        "time": times,
        "direct_normal_irradiance": 800 * daylight,
        "diffuse_radiation": 110 * daylight,
        "shortwave_radiation": 920 * daylight,
        "cloud_cover": np.full(168, 15.0),
        "temperature_2m": 25 + 8 * daylight,
    })


def solar_position(times: pd.Series, latitude: float, longitude: float) -> tuple[np.ndarray, np.ndarray]:
    """Approximate local solar altitude and bearing; bearing is 0=south, -east, +west."""
    day = times.dt.dayofyear.to_numpy()
    local_hour = times.dt.hour.to_numpy() + times.dt.minute.to_numpy() / 60
    declination = np.radians(23.45 * np.sin(np.radians(360 * (284 + day) / 365)))
    solar_hour = local_hour + longitude / 15 - np.round(longitude / 15)
    hour_angle = np.radians(15 * (solar_hour - 12))
    phi = np.radians(latitude)
    altitude = np.arcsin(np.sin(phi) * np.sin(declination) + np.cos(phi) * np.cos(declination) * np.cos(hour_angle))
    bearing = np.arctan2(np.sin(hour_angle), np.cos(hour_angle) * np.sin(phi) - np.tan(declination) * np.cos(phi))
    return altitude, bearing


def plane_irradiance(tilt: float, azimuth: float, weather: pd.DataFrame, latitude: float, longitude: float) -> np.ndarray:
    """Calculate weather-adjusted irradiance on a tilted panel using a Liu-Jordan-style model."""
    altitude, sun_bearing = solar_position(weather["time"], latitude, longitude)
    tilt_r = np.radians(tilt)
    incidence = np.sin(altitude) * np.cos(tilt_r) + np.cos(altitude) * np.sin(tilt_r) * np.cos(sun_bearing - np.radians(azimuth))
    beam_ratio = np.maximum(incidence, 0) / np.maximum(np.sin(altitude), 0.065)
    dni = weather["direct_normal_irradiance"].to_numpy()
    diffuse = weather["diffuse_radiation"].to_numpy()
    ground = np.maximum(weather["shortwave_radiation"].to_numpy() - dni * np.maximum(np.sin(altitude), 0) - diffuse, 0)
    plane = dni * beam_ratio + diffuse * (1 + np.cos(tilt_r)) / 2 + ground * ALBEDO * (1 - np.cos(tilt_r)) / 2
    cloud_factor = 1 - weather["cloud_cover"].to_numpy() / 100 * 0.12
    soiling_factor = 1 - 0.12 * np.exp(-0.18 * tilt)
    return np.maximum(plane * cloud_factor * soiling_factor, 0)


def optimize(weather: pd.DataFrame, latitude: float, longitude: float) -> tuple[float, float]:
    def objective(values: np.ndarray) -> float:
        return -float(plane_irradiance(values[0], values[1], weather, latitude, longitude).sum())

    result = differential_evolution(objective, bounds=[(10, 60), (-45, 45)], seed=7, polish=True)
    return float(result.x[0]), float(result.x[1])


def direction_text(azimuth: float) -> str:
    if abs(azimuth) < 1:
        return "Face due south"
    side = "west" if azimuth > 0 else "east"
    return f"Face {abs(azimuth):.0f}° {side} of south"


def energy_table(irradiance: np.ndarray, weather: pd.DataFrame, area: float) -> pd.DataFrame:
    result = weather[["time"]].copy()
    result["panel_irradiance_wm2"] = irradiance
    result["energy_kwh"] = irradiance * area * SYSTEM_EFFICIENCY / 1000
    result["date"] = result["time"].dt.date
    return result


st.title("☀️ Solar Energy Planner")
st.caption("Find the best panel angle, strongest solar-collection hours, and energy outlook for any location.")

with st.sidebar:
    st.header("1. Location")
    country = st.text_input("Country", "United States")
    state = st.text_input("State / province", "Arizona")
    city = st.text_input("City", "Phoenix")
    postal_code = st.text_input("ZIP / postal code", "")
    st.divider()
    st.header("2. Panel and plan")
    panel_area = st.slider("Panel area (m²)", 1.0, 100.0, 15.0, 0.5)
    interval = st.selectbox("Optimization plan", ["Daily", "Monthly", "Quarterly", "Biannual", "Annual"], help="Uses live seven-day forecast data. Longer plans are forecast-based projections, not historical simulations.")
    run = st.button("Find my best solar plan", type="primary", use_container_width=True)
    st.caption("Energy estimate assumes a 20% efficient panel. Change the code later if your panel's efficiency differs.")

if "run" not in st.session_state:
    st.session_state.run = True
if run:
    st.session_state.run = True

if st.session_state.run:
    query = location_query(country, state, city, postal_code)
    if not query:
        st.error("Please enter at least a country, city, state/province, or ZIP/postal code.")
        st.stop()

    try:
        with st.spinner("Checking the location and loading the seven-day solar forecast..."):
            latitude, longitude, matched_location = geocode(query)
            weather, timezone = fetch_weather(latitude, longitude)
        source = f"Live Open-Meteo forecast ({timezone})"
    except (requests.RequestException, ValueError, KeyError) as error:
        latitude, longitude = 33.4484, -112.0740
        matched_location, timezone = "Phoenix, Arizona (demo location)", "local time"
        weather = demo_weather()
        source = "Demo forecast - live weather was unavailable"
        st.warning(f"{source}: {error}")

    optimization_weather = weather.iloc[:24].copy() if interval == "Daily" else weather
    best_tilt, best_azimuth = optimize(optimization_weather, latitude, longitude)
    optimized_irradiance = plane_irradiance(best_tilt, best_azimuth, weather, latitude, longitude)
    latitude_tilt = float(np.clip(abs(latitude), 10, 60))
    baseline_irradiance = plane_irradiance(latitude_tilt, 0, weather, latitude, longitude)
    optimized = energy_table(optimized_irradiance, weather, panel_area)
    baseline = energy_table(baseline_irradiance, weather, panel_area)

    daily_energy = optimized.groupby("date", as_index=False)["energy_kwh"].sum()
    predicted_daily = float(daily_energy.iloc[0]["energy_kwh"])
    baseline_daily = float(baseline.groupby("date")["energy_kwh"].sum().iloc[0])
    gain = (predicted_daily / baseline_daily - 1) * 100 if baseline_daily else 0
    days_in_plan = {"Daily": 1, "Monthly": 30, "Quarterly": 91, "Biannual": 182, "Annual": 365}[interval]
    projected_total = predicted_daily if interval == "Daily" else float(daily_energy["energy_kwh"].mean() * days_in_plan)

    today = optimized.iloc[:24].copy()
    peak_row = today.loc[today["panel_irradiance_wm2"].idxmax()]
    prime = today[today["panel_irradiance_wm2"] >= peak_row["panel_irradiance_wm2"] * 0.75]
    if peak_row["panel_irradiance_wm2"] > 0 and not prime.empty:
        prime_start, prime_end = prime.iloc[0]["time"], prime.iloc[-1]["time"] + pd.Timedelta(hours=1)
        prime_window = f"{prime_start.strftime('%I:%M %p').lstrip('0')} - {prime_end.strftime('%I:%M %p').lstrip('0')}"
        peak_time = peak_row["time"].strftime("%I:%M %p").lstrip("0")
    else:
        prime_start = prime_end = None
        prime_window, peak_time = "No strong solar window forecast", "No daylight forecast"

    st.success(f"Location found: {matched_location} | {latitude:.4f}°, {longitude:.4f}° | {source}")
    if interval != "Daily":
        st.info(f"{interval} total is a {days_in_plan}-day projection based on the average of this seven-day forecast.")

    st.subheader("Your recommended setup")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Recommended tilt", f"{best_tilt:.1f}°", "Measured up from horizontal")
    c2.metric("Recommended facing direction", direction_text(best_azimuth), "Azimuth orientation")
    c3.metric("Predicted daily energy", f"{predicted_daily:.1f} kWh")
    c4.metric("Gain vs. latitude tilt", f"{gain:+.1f}%")
    st.metric(f"Estimated {interval.lower()} total", f"{projected_total:.1f} kWh", f"Based on a {panel_area:.1f} m² panel area")

    st.subheader("Best time to collect solar energy")
    timer_left, timer_right = st.columns(2)
    timer_left.metric("Best collection window", prime_window)
    timer_right.metric("Peak solar time", peak_time, f"{peak_row['panel_irradiance_wm2']:.0f} W/m² at the panel")
    st.caption("The best collection window includes hours forecast to reach at least 75% of the day’s peak solar input.")

    chart_left, chart_right = st.columns(2)
    hourly = pd.DataFrame({"Time": weather.iloc[:24]["time"], "Optimized": optimized_irradiance[:24], "Latitude tilt": baseline_irradiance[:24]}).melt("Time", var_name="Setup", value_name="Irradiance (W/m²)")
    with chart_left:
        st.subheader("Today’s solar capture")
        line = px.line(hourly, x="Time", y="Irradiance (W/m²)", color="Setup", template="plotly_white")
        if prime_start is not None:
            line.add_vrect(x0=prime_start, x1=prime_end, fillcolor="#fbbf24", opacity=0.2, line_width=0, annotation_text="Best collection window")
        st.plotly_chart(line, use_container_width=True)
    with chart_right:
        st.subheader("7-day energy forecast")
        daily_energy["date"] = daily_energy["date"].astype(str)
        bar = px.bar(daily_energy, x="date", y="energy_kwh", labels={"date": "Date", "energy_kwh": "Energy (kWh)"}, template="plotly_white", color_discrete_sequence=["#f59e0b"])
        st.plotly_chart(bar, use_container_width=True)

    st.subheader("Future energy prediction")
    projection = pd.DataFrame({"Period": ["Daily", "Monthly", "Quarterly", "Biannual", "Annual"], "Estimated energy (kWh)": [predicted_daily, predicted_daily * 30, predicted_daily * 91, predicted_daily * 182, predicted_daily * 365]})
    st.plotly_chart(px.bar(projection, x="Period", y="Estimated energy (kWh)", template="plotly_white", color_discrete_sequence=["#0ea5e9"]), use_container_width=True)

    download = optimized.copy()
    download["recommended_tilt_deg"] = best_tilt
    download["recommended_azimuth_deg"] = best_azimuth
    download["location"] = matched_location
    st.download_button("Download hourly solar plan (CSV)", download.to_csv(index=False).encode("utf-8"), "solar_energy_plan.csv", "text/csv")
    st.caption("Planning estimate only. Verify roof structure, shading, electrical design, permits, and manufacturer requirements with a qualified solar professional.")
