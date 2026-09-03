"""Dynamic solar panel tilt and azimuth optimizer.

Run with: streamlit run app.py
"""
from __future__ import annotations

from datetime import date
from io import StringIO
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from scipy.optimize import differential_evolution, minimize


st.set_page_config(page_title="Solar Tilt Optimizer", page_icon="☀️", layout="wide")

NOMINATIM = "https://nominatim.openstreetmap.org/search"
OPEN_METEO = "https://api.open-meteo.com/v1/forecast"
HEADERS = {"User-Agent": "SolarTiltOptimizer/1.0 (educational solar planning tool)"}
ALBEDO = 0.25


@st.cache_data(ttl=60 * 60, show_spinner=False)
def geocode(location: str) -> tuple[float, float, str]:
    """Resolve a city, address, or ZIP code using OpenStreetMap Nominatim."""
    response = requests.get(
        NOMINATIM,
        params={"q": location, "format": "json", "limit": 1},
        headers=HEADERS,
        timeout=12,
    )
    response.raise_for_status()
    results = response.json()
    if not results:
        raise ValueError("Location not found. Try city, state/country, or coordinates.")
    result = results[0]
    return float(result["lat"]), float(result["lon"]), result["display_name"]


@st.cache_data(ttl=60 * 60, show_spinner=False)
def fetch_forecast(latitude: float, longitude: float) -> tuple[pd.DataFrame, str]:
    """Fetch seven days of hourly solar and weather forecasts without an API key."""
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": "direct_normal_irradiance,diffuse_radiation,shortwave_radiation,cloud_cover,temperature_2m",
        "forecast_days": 7,
        "timezone": "auto",
    }
    response = requests.get(OPEN_METEO, params=params, timeout=15)
    response.raise_for_status()
    payload = response.json()
    hourly = payload.get("hourly")
    if not hourly:
        raise ValueError("The weather service returned no hourly forecast data.")
    frame = pd.DataFrame(hourly)
    frame["time"] = pd.to_datetime(frame["time"])
    return frame, payload.get("timezone_abbreviation", payload.get("timezone", "local"))


def fallback_forecast() -> pd.DataFrame:
    """Keep the experience usable if an external forecast service is unavailable."""
    times = pd.date_range(pd.Timestamp.now().floor("h"), periods=168, freq="h")
    hour = times.hour.to_numpy()
    solar = np.clip(np.sin(np.pi * (hour - 6) / 12), 0, None)
    return pd.DataFrame({
        "time": times,
        "direct_normal_irradiance": 820 * solar,
        "diffuse_radiation": 90 * solar,
        "shortwave_radiation": 900 * solar,
        "cloud_cover": np.full(len(times), 15.0),
        "temperature_2m": 27 + 8 * np.sin(np.pi * (hour - 8) / 12),
    })


def sun_geometry(times: pd.Series, latitude: float, longitude: float) -> tuple[np.ndarray, np.ndarray]:
    """Approximate solar altitude and bearing from local timestamps.

    Bearing uses the app convention: 0 = south, negative = east, positive = west.
    """
    day = times.dt.dayofyear.to_numpy()
    hour = times.dt.hour.to_numpy() + times.dt.minute.to_numpy() / 60
    declination = np.radians(23.45 * np.sin(np.radians(360 * (284 + day) / 365)))
    # Forecast times are local; longitude correction provides a practical planning estimate.
    solar_time = hour + longitude / 15 - np.round(longitude / 15)
    hour_angle = np.radians(15 * (solar_time - 12))
    phi = np.radians(latitude)
    altitude = np.arcsin(np.sin(phi) * np.sin(declination) + np.cos(phi) * np.cos(declination) * np.cos(hour_angle))
    # atan2 expression yields zero at south, negative east, positive west.
    bearing = np.arctan2(
        np.sin(hour_angle),
        np.cos(hour_angle) * np.sin(phi) - np.tan(declination) * np.cos(phi),
    )
    return altitude, bearing


def plane_irradiance(beta: float, gamma: float, weather: pd.DataFrame, latitude: float, longitude: float) -> np.ndarray:
    """Liu-Jordan plane-of-array irradiance with cloud and soiling effects."""
    altitude, sun_bearing = sun_geometry(weather["time"], latitude, longitude)
    beta_r, gamma_r = np.radians(beta), np.radians(gamma)
    cos_incidence = np.sin(altitude) * np.cos(beta_r) + np.cos(altitude) * np.sin(beta_r) * np.cos(sun_bearing - gamma_r)
    cos_zenith = np.maximum(np.sin(altitude), 0.065)
    rb = np.maximum(cos_incidence, 0) / cos_zenith
    dni = weather["direct_normal_irradiance"].to_numpy()
    diffuse = weather["diffuse_radiation"].to_numpy()
    ground = np.maximum(weather["shortwave_radiation"].to_numpy() - dni * np.maximum(np.sin(altitude), 0) - diffuse, 0)
    tilt_r = np.radians(beta)
    irradiance = dni * rb + diffuse * (1 + np.cos(tilt_r)) / 2 + ground * ALBEDO * (1 - np.cos(tilt_r)) / 2
    cloud_factor = 1 - weather["cloud_cover"].to_numpy() / 100 * 0.12
    soiling_loss = 0.12 * np.exp(-0.18 * beta)
    return np.maximum(irradiance * cloud_factor * (1 - soiling_loss), 0)


def optimize(weather: pd.DataFrame, latitude: float, longitude: float, initial: tuple[float, float] | None = None) -> tuple[float, float, float]:
    """Find the constrained configuration with the greatest forecast plane irradiance."""
    def objective(x: np.ndarray) -> float:
        return -float(plane_irradiance(x[0], x[1], weather, latitude, longitude).sum())

    # Differential evolution helps avoid a poor local result; SLSQP refines it.
    global_result = differential_evolution(objective, [(10, 60), (-45, 45)], seed=42, polish=False)
    result = minimize(objective, initial or global_result.x, method="SLSQP", bounds=[(10, 60), (-45, 45)])
    beta, gamma = result.x if result.success else global_result.x
    return float(beta), float(gamma), -objective(np.array([beta, gamma]))


def daily_energy(irradiance: np.ndarray, weather: pd.DataFrame, area: float, efficiency: float) -> pd.DataFrame:
    output = weather[["time"]].copy()
    output["plane_irradiance_wm2"] = irradiance
    output["energy_kwh"] = irradiance * area * efficiency / 1000
    output["date"] = output["time"].dt.date
    return output


st.title("☀️ Solar Tilt Optimizer")
st.caption("Forecast-informed panel-angle recommendations using Open-Meteo solar weather data.")

with st.sidebar:
    st.header("System inputs")
    location = st.text_input("Location", "Phoenix, AZ", help="City, ZIP/postal code, or a detailed address.")
    area = st.slider("Panel area (m²)", 1.0, 100.0, 15.0, 0.5)
    efficiency = st.slider("Panel efficiency (%)", 10.0, 30.0, 20.0, 0.5) / 100
    mode = st.radio("Adjustment mode", ["Daily optimal", "Quarterly plan", "Fixed annual"], help="Daily uses the next 24 hours; quarterly produces four seasonal recommendations; fixed annual uses the full forecast.")
    calculate = st.button("Calculate recommendation", type="primary", use_container_width=True)

if "calculated" not in st.session_state:
    st.session_state.calculated = True
if calculate:
    st.session_state.calculated = True

if st.session_state.calculated:
    try:
        with st.spinner("Locating site and retrieving the solar forecast..."):
            latitude, longitude, place = geocode(location)
            weather, timezone = fetch_forecast(latitude, longitude)
        source_note = f"Live Open-Meteo forecast ({timezone})"
    except (requests.RequestException, ValueError, KeyError) as exc:
        latitude, longitude, place = 33.4484, -112.0740, "Phoenix, Arizona (fallback location)"
        weather, timezone = fallback_forecast(), "local"
        source_note = "Demo forecast - live lookup was unavailable"
        st.warning(f"{source_note}: {exc}")

    if mode == "Daily optimal":
        active_weather = weather.iloc[:24].copy()
    else:
        active_weather = weather.copy()
    beta, gamma, objective_energy = optimize(active_weather, latitude, longitude)
    optimized = plane_irradiance(beta, gamma, weather, latitude, longitude)
    fixed_beta = float(np.clip(abs(latitude), 10, 60))
    fixed = plane_irradiance(fixed_beta, 0, weather, latitude, longitude)
    optimized_daily = daily_energy(optimized, weather, area, efficiency)
    fixed_daily = daily_energy(fixed, weather, area, efficiency)
    today_energy = optimized_daily.iloc[:24]["energy_kwh"].sum()
    fixed_today = fixed_daily.iloc[:24]["energy_kwh"].sum()
    gain = (today_energy / fixed_today - 1) * 100 if fixed_today else 0

    st.success(f"📍 {place} · {latitude:.4f}°, {longitude:.4f}° · {source_note}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Recommended tilt", f"{beta:.1f}°", f"{beta - abs(latitude):+.1f}° vs. latitude")
    c2.metric("Recommended azimuth", f"{gamma:.1f}°", "0° = south")
    c3.metric("Predicted energy today", f"{today_energy:.1f} kWh")
    c4.metric("Gain vs. latitude tilt", f"{gain:+.1f}%")

    left, right = st.columns(2)
    hourly_chart = pd.DataFrame({
        "Time": weather.iloc[:24]["time"],
        "Optimized": optimized[:24],
        "Latitude tilt": fixed[:24],
    }).melt("Time", var_name="Configuration", value_name="Irradiance (W/m²)")
    with left:
        st.subheader("Today’s plane-of-array irradiance")
        st.plotly_chart(px.line(hourly_chart, x="Time", y="Irradiance (W/m²)", color="Configuration", template="plotly_white"), use_container_width=True)
    with right:
        st.subheader("7-day energy forecast")
        weekly = optimized_daily.groupby("date", as_index=False)["energy_kwh"].sum()
        weekly["date"] = weekly["date"].astype(str)
        st.plotly_chart(px.bar(weekly, x="date", y="energy_kwh", labels={"date": "Date", "energy_kwh": "Energy (kWh)"}, template="plotly_white", color_discrete_sequence=["#f59e0b"]), use_container_width=True)

    st.subheader("Energy landscape")
    tilts = np.linspace(10, 60, 26)
    azimuths = np.linspace(-45, 45, 25)
    surface = np.array([[plane_irradiance(t, a, active_weather, latitude, longitude).sum() * area * efficiency / 1000 for t in tilts] for a in azimuths])
    heatmap = go.Figure(go.Heatmap(x=tilts, y=azimuths, z=surface, colorscale="YlOrRd", colorbar_title="kWh"))
    heatmap.add_scatter(x=[beta], y=[gamma], mode="markers", marker={"color": "#0f172a", "size": 11, "symbol": "x"}, name="Recommendation")
    heatmap.update_layout(template="plotly_white", xaxis_title="Tilt (degrees)", yaxis_title="Azimuth (degrees; east - / west +)", height=390)
    st.plotly_chart(heatmap, use_container_width=True)

    if mode == "Quarterly plan":
        st.subheader("Seasonal adjustment plan")
        quarters = np.array_split(weather, 4)
        plan = []
        previous_tilt = None
        for label, quarter in zip(["Q1", "Q2", "Q3", "Q4"], quarters):
            q_beta, q_gamma, _ = optimize(quarter, latitude, longitude)
            if previous_tilt is not None:
                q_beta = float(np.clip(q_beta, previous_tilt - 30, previous_tilt + 30))
            previous_tilt = q_beta
            plan.append({"Period": label, "Tilt (°)": round(q_beta, 1), "Azimuth (°)": round(q_gamma, 1)})
        st.dataframe(pd.DataFrame(plan), use_container_width=True, hide_index=True)

    export = optimized_daily.copy()
    export["recommended_tilt_deg"] = beta
    export["recommended_azimuth_deg"] = gamma
    export["location"] = place
    st.download_button("Download hourly recommendation CSV", export.to_csv(index=False).encode("utf-8"), "solar_tilt_recommendation.csv", "text/csv")
    st.caption("Planning estimate only. Confirm structural, electrical, shading, permitting, and manufacturer constraints with a qualified solar professional.")
