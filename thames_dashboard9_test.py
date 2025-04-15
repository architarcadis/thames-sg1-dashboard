# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 19:40:00 2025

@author: GeminiAdapter (Based on bansala4846)

Updated Thames Water Streamlit App with revised KPI definitions.
Includes fix for timedelta unit 'M' error and budget aggregation error.
Restructured tabs for storyline flow.
Enhanced Data Discovery tab based on KPI requirements and personas.
Simplified Landing Page placeholder and fixed rendering.
Refocused analytical tabs (Summary, Gap, Programme, Advanced) around the 7 KPIs.
Enhanced Executive Summary tab for more direct KPI insights.
Fixed SyntaxError in Advanced Analytics regression formula (removed backticks).
"""

# Full Code Block: Thames Water SG1 Insights Dashboard (Regression Fix)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.io as pio # Import for Plotly templates
from datetime import datetime, timedelta, date
from wordcloud import WordCloud
import base64 # For logo embedding (optional)
import statsmodels.api as sm # For forecasting/regression
from statsmodels.tsa.arima.model import ARIMA # For Forecasting
from sklearn.cluster import KMeans # For Clustering
from sklearn.preprocessing import StandardScaler # For Clustering scaling
import statsmodels.formula.api as smf # For Regression
import traceback # For detailed error reporting
import matplotlib.pyplot as plt # Added for WordCloud display

# -----------------------------
# App Config & Thames Water Theme
# -----------------------------
st.set_page_config(
    layout="wide",
    page_title="Thames Water SG1 Insights",
    page_icon="🌊" # Favicon emoji
)

# Thames Water Colors
THAMES_COLORS = {
    "Primary Blue": "#005670",
    "Secondary Blue": "#00A1D6",
    "Green": "#28A745",
    "Red": "#FF4B4B",
    "Amber": "#FFB107",
    "Background": "#F0F2F6", # Light grey background
    "Text": "#31333F",
    "Gray": "#6c757d" # Added a gray for subtitles/captions
}

# --- Define Custom Plotly Template ---
thames_template = go.layout.Template()
thames_template.layout.font = dict(family="Roboto, sans-serif", size=12, color=THAMES_COLORS['Text'])
thames_template.layout.title.font = dict(family="Roboto, sans-serif", size=16, color=THAMES_COLORS['Primary Blue'])
thames_template.layout.paper_bgcolor = 'rgba(0,0,0,0)' # Transparent background
thames_template.layout.plot_bgcolor = 'rgba(0,0,0,0)'
thames_template.layout.margin = dict(l=40, r=40, t=60, b=40)
thames_template.layout.colorway = [THAMES_COLORS['Secondary Blue'], THAMES_COLORS['Primary Blue'], THAMES_COLORS['Green'], THAMES_COLORS['Amber'], THAMES_COLORS['Red'], THAMES_COLORS['Gray']]
thames_template.layout.hoverlabel = dict(bgcolor="white", font_size=12, font_family="Roboto, sans-serif")
thames_template.layout.xaxis = dict(showgrid=False, linecolor=THAMES_COLORS['Gray'], tickcolor=THAMES_COLORS['Gray'], title_font_size=13, tickfont_size=11)
thames_template.layout.yaxis = dict(showgrid=True, gridcolor='#e1e1e1', linecolor=THAMES_COLORS['Gray'], tickcolor=THAMES_COLORS['Gray'], title_font_size=13, tickfont_size=11)
thames_template.layout.legend = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor='rgba(255,255,255,0.6)', font_size=11)
thames_template.layout.title.x = 0.5
thames_template.layout.title.xanchor = 'center'
pio.templates["thames_template"] = thames_template
pio.templates.default = "thames_template"


# Apply Custom CSS for Branding and Layout (Enhanced)
# IMPORTANT: Keep this CSS definition at the beginning
st.markdown(f"""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');

    /* General Styles */
    body, .stApp {{
        font-family: 'Roboto', sans-serif;
        color: {THAMES_COLORS['Text']};
        background-color: {THAMES_COLORS['Background']};
    }}
    .stApp > header {{
        background-color: {THAMES_COLORS['Primary Blue']};
        color: white;
    }}
    /* Metric Card Styles (Updated for KPIs) */
    .kpi-card {{
        background-color: white; padding: 1rem 1rem; border-radius: 8px;
        box-shadow: 0 4px 10px rgba(0, 86, 112, 0.1); text-align: center;
        border-left: 5px solid {THAMES_COLORS['Secondary Blue']}; margin-bottom: 1rem;
        min-height: 120px; display: flex; flex-direction: column; justify-content: space-between; /* Adjusted for KPI */
        transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out;
    }}
    .kpi-card:hover {{ transform: translateY(-3px); box-shadow: 0 6px 15px rgba(0, 86, 112, 0.15); }}
    .kpi-card h3 {{ font-size: 0.95rem; color: {THAMES_COLORS['Primary Blue']}; margin-bottom: 0.4rem; font-weight: bold; }}
    .kpi-card .value {{ font-size: 1.6rem; font-weight: bold; color: {THAMES_COLORS['Text']}; margin: 0.2rem 0; }}
    .kpi-card .status {{ font-size: 0.9rem; font-weight: bold; margin-top: 0.4rem; padding: 0.2rem 0.5rem; border-radius: 4px; display: inline-block; }}
    .kpi-card .status-green {{ background-color: #e9f5ea; color: {THAMES_COLORS['Green']}; }}
    .kpi-card .status-amber {{ background-color: #fff8e1; color: {THAMES_COLORS['Amber']}; }}
    .kpi-card .status-red {{ background-color: #ffebee; color: {THAMES_COLORS['Red']}; }}
    .kpi-card .status-none {{ color: {THAMES_COLORS['Gray']}; font-style: italic; font-size: 0.85rem;}} /* For KPIs with no RAG */
    /* Specific style for summary RAG counts */
    .kpi-card .rag-summary {{ font-size: 0.85rem; line-height: 1.4; margin-top: 0.4rem; }}
    .kpi-card .rag-summary span {{ margin: 0 0.3rem; }}


    .insight-box {{
        background-color: #e7f3f7; padding: 1rem 1.2rem; border-radius: 8px;
        border-left: 5px solid {THAMES_COLORS['Secondary Blue']}; margin-bottom: 1rem;
        font-size: 0.95rem; line-height: 1.6; box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
    }}
    /* Sidebar Styles */
    .stSidebar {{ background-color: {THAMES_COLORS['Primary Blue']} !important; }}
    .stSidebar > div:first-child {{ color: white; padding-top: 1rem; }}
    .stSidebar h1, .stSidebar h2, .stSidebar h3, .stSidebar h4, .stSidebar p, .stSidebar label, .stSidebar .stMarkdown, .stSidebar .stButton>button {{ color: white !important; }}
    .stSidebar .stButton button, .stSidebar a[href^="data:file/csv;base64"] {{ background-color: {THAMES_COLORS['Secondary Blue']}; color: white; border: none; border-radius: 5px; padding: 0.5rem 1rem; width: calc(100% - 2rem); margin: 0.5rem 1rem; text-align: center; display: inline-block; text-decoration: none; transition: background-color 0.2s ease-in-out, transform 0.1s ease; }}
    .stSidebar .stButton button:hover, .stSidebar a[href^="data:file/csv;base64"]:hover {{ background-color: #007bbb; transform: scale(1.02); }}
    .stSidebar .stButton button:active, .stSidebar a[href^="data:file/csv;base64"]:active {{ transform: scale(0.98); }}
    /* Tab Styles */
    .stTabs [data-baseweb="tab-list"] {{ background-color: {THAMES_COLORS['Primary Blue']}; padding: 0.5rem; border-radius: 5px 5px 0 0; }}
    .stTabs [data-baseweb="tab"] {{ height: 40px; background-color: transparent; color: white; border-radius: 5px; padding: 0.5rem 1rem; cursor: pointer; transition: background-color 0.2s ease;}}
    .stTabs [data-baseweb="tab"]:hover {{ background-color: rgba(0, 161, 214, 0.3); }}
    .stTabs [aria-selected="true"] {{ background-color: {THAMES_COLORS['Secondary Blue']}; color: white !important; font-weight: bold; }}
    /* Expander Styles */
    .stExpander {{ border: 1px solid #ddd; border-radius: 8px; background-color: #ffffff; margin-bottom: 1.5rem; box-shadow: 0 2px 4px rgba(0,0,0,0.05);}}
    .stExpander header {{ font-weight: bold; color: {THAMES_COLORS['Primary Blue']}; padding: 0.8rem 1rem;}}
    /* Divider */
    hr {{ margin-top: 2rem; margin-bottom: 2rem; border-top: 1px solid #ccc; }}
    /* Plotly Chart Hover Tooltips */
    .plotly .hovertext {{ font-family: 'Roboto', sans-serif !important; }}

    /* Landing Page Specific */
    .landing-container {{
        padding: 2rem;
        background-color: white;
        border-radius: 8px;
        box-shadow: 0 4px 10px rgba(0, 86, 112, 0.1);
        margin-top: 1rem;
    }}
    /* Style headings directly in markdown for landing page */
    .landing-container-markdown h1 {{
        color: {THAMES_COLORS['Primary Blue']};
        border-bottom: 2px solid {THAMES_COLORS['Secondary Blue']};
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
        font-size: 1.75rem; /* Adjust size if needed */
    }}
     .landing-container-markdown p {{
        line-height: 1.7;
        margin-bottom: 1rem;
     }}
     .landing-container-markdown ul {{
        list-style-position: inside;
        margin-left: 1rem;
        margin-bottom: 1rem; /* Add space below lists */
     }}
      .landing-container-markdown li {{
        margin-bottom: 0.5rem; /* Space between list items */
     }}

     /* Diagram Placeholder Styles (Simpler) */
     .diagram-placeholder table {{
         width: 100%;
         border-collapse: collapse;
         background-color: #fdfdff; /* Slightly off-white */
         border: 1px solid #e0e0e0; /* Lighter border */
         border-radius: 8px; /* Rounded corners for table */
         box-shadow: inset 0 1px 3px rgba(0,0,0,0.05);
         margin-top: 1.5rem;
     }}
     .diagram-placeholder th, .diagram-placeholder td {{
         border: 1px solid #eee; /* Lighter internal borders */
         padding: 12px; /* More padding */
         text-align: center;
         vertical-align: top;
     }}
     .diagram-placeholder th {{
         background-color: #e7f3f7; /* Light blue background */
         color: {THAMES_COLORS['Primary Blue']};
         font-weight: bold;
         font-size: 1.1rem;
         border-top-left-radius: 8px; /* Rounded corners for header */
         border-top-right-radius: 8px;
     }}
      .diagram-placeholder td {{
         background-color: #fff; /* White background for cells */
      }}
     .diagram-placeholder td ul {{
         list-style-type: none;
         padding: 0;
         margin: 0;
         text-align: left;
         font-size: 0.9rem;
     }}
     .diagram-placeholder td ul li {{
         margin-bottom: 0.4rem;
         padding: 0.3rem 0.5rem; /* Padding within list items */
         border-left: 3px solid {THAMES_COLORS['Secondary Blue']};
         background-color: #f8f9fa;
         border-radius: 3px;
     }}
     .diagram-placeholder .arrow {{
         font-size: 1.8rem; /* Slightly larger arrow */
         color: {THAMES_COLORS['Gray']};
         border: none; /* Remove border from arrow cell */
         text-align: center;
         vertical-align: middle;
         background-color: transparent; /* Ensure arrow cell is transparent */
     }}
     .update-note {{
         width: 100%;
         text-align: center;
         font-style: italic;
         color: {THAMES_COLORS['Gray']};
         margin-top: 1.5rem;
         font-size: 0.9rem;
     }}


    /* Data Discovery Specific */
    .discovery-section h3 {{
        color: {THAMES_COLORS['Primary Blue']};
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
        border-bottom: 1px solid #ddd;
        padding-bottom: 0.3rem;
    }}
     .discovery-section .stTextArea {{
        margin-bottom: 1rem; /* Add space below text areas */
     }}
     .persona-note {{
         font-size: 0.85rem;
         font-style: italic;
         color: {THAMES_COLORS['Gray']};
         margin-bottom: 0.5rem;
     }}

</style>
""", unsafe_allow_html=True)


# -----------------------------
# Mock Data Generation (Updated for new KPIs)
# -----------------------------
@st.cache_data(ttl=3600) # Cache data for 1 hour
def create_mock_data(num_sites=30):
    """Generates enhanced mock data including fields for new KPIs."""
    np.random.seed(42) # for reproducibility
    sites = [f"Site-{i+1:02d}" for i in range(num_sites)] # Use leading zeros for sorting
    programmes = ["Programme Alpha", "Programme Beta", "Programme Gamma"] * (num_sites // 3 + 1)
    programmes = programmes[:num_sites]
    np.random.shuffle(programmes)
    catchments = ["North Catchment", "South Catchment", "Central System"] * (num_sites // 3 + 1)
    catchments = catchments[:num_sites]
    np.random.shuffle(catchments)

    possible_risks = ["Operational Failure", "Data Breach", "Compliance Issue", "Aging Infrastructure", "Supply Chain Delay", "Resource Constraint", "Sensor Malfunction"]
    risk_details_list = np.random.choice(possible_risks, size=num_sites, replace=True)
    for i in range(num_sites // 4): # Add some more details for word cloud
        risk_details_list[np.random.randint(0, num_sites)] += "; Data Quality Concern"
        risk_details_list[np.random.randint(0, num_sites)] += "; Reporting Delay"

    today = date.today()
    baseline_dates = [today + timedelta(days=np.random.randint(60, 365)) for _ in range(num_sites)]
    forecast_dates = [d + timedelta(days=np.random.randint(-45, 60)) for d in baseline_dates] # Add variance around baseline
    resource_end_dates = [today + timedelta(days=np.random.randint(30, 450)) for _ in range(num_sites)] # For KPI 6

    # Create DataFrame
    data = pd.DataFrame({
        "Site": sites,
        "Programme Name": programmes,
        "Catchment": catchments,
        "Baseline SG1 Date": pd.to_datetime(baseline_dates),
        "Forecast SG1 Date": pd.to_datetime(forecast_dates),
        "Programme IBP Budget (£K)": np.random.choice([5000, 10000, 15000], size=num_sites), # Programme level budget (will be duplicated per site initially)
        "Project Forecast Outturn (£K)": np.random.uniform(150, 750, num_sites).round(1), # Site level forecast
        "Project Actual Outturn (£K)": [np.nan] * num_sites, # Initially NaN
        "Engineering Design Cost (£K)": np.random.uniform(5, 30, num_sites).round(1),
        "Engineering Budget (£K)": np.random.uniform(8, 40, num_sites).round(1), # Site level eng budget
        "Documentation Completion (%)": np.random.uniform(85, 100, num_sites).round(1),
        "Resource Allocation End Date": pd.to_datetime(resource_end_dates),
        "HS File Signed": np.random.choice([True, False], size=num_sites, p=[0.8, 0.2]),
        "CDM Checklist Complete": np.random.choice([True, False], size=num_sites, p=[0.75, 0.25]),
        # --- Existing/Supporting Columns ---
        "Data Completeness (%)": np.random.uniform(55, 99, num_sites).round(1),
        "SG1 Progress (%)": np.random.uniform(20, 95, num_sites).round(1), # Still useful for context
        "Risk Score": np.random.uniform(1.0, 9.9, num_sites).round(1), # Still useful for context
        "Carbon Impact (tCO2e)": np.random.uniform(100, 550, num_sites).round(1), # Still useful for context
        "Data Governance Score (%)": np.random.uniform(40, 95, num_sites).round(1), # Used in Gap Analysis
        "Data Architecture Score (%)": np.random.uniform(30, 90, num_sites).round(1), # Used in Gap Analysis
        "Compliance Score (%)": np.random.uniform(60, 99, num_sites).round(1), # Used in Gap Analysis
        "Cost Variance (%)": np.random.uniform(-15, 20, num_sites).round(1), # Superseded by KPI 2/3 but kept for potential use
        "Schedule Variance (%)": np.random.uniform(-10, 15, num_sites).round(1), # Superseded by KPI 1 but kept for potential use
        "Innovation Potential (%)": np.random.uniform(20, 85, num_sites).round(1), # Used in Advanced Analytics
        "Risk Details": risk_details_list,
    })

    # Add some actuals for projects likely past SG1 (based on forecast date)
    past_sg1_indices = data[data['Forecast SG1 Date'] < pd.to_datetime(today)].index
    for idx in past_sg1_indices:
        # Simulate actual outturn based on forecast with some variance
        data.loc[idx, 'Project Actual Outturn (£K)'] = data.loc[idx, 'Project Forecast Outturn (£K)'] * np.random.uniform(0.95, 1.1)

    # Ensure budgets are reasonable compared to costs
    data['Engineering Budget (£K)'] = data['Engineering Design Cost (£K)'] * np.random.uniform(1.1, 1.8) # Budget > Cost
    # Aggregate site forecasts to get a mock programme forecast (summing site forecasts)
    prog_forecast = data.groupby('Programme Name')['Project Forecast Outturn (£K)'].transform('sum')
    # Ensure programme budget is somewhat related to sum of site forecasts
    data['Programme IBP Budget (£K)'] = prog_forecast * np.random.uniform(0.9, 1.2)

    # Calculate TG1 Progress based on SG1 Progress (as before) - Keep for context if needed
    data["TG1 Progress (%)"] = (data["SG1 Progress (%)"] * np.random.uniform(0.9, 1.1)).clip(0, 100).round(1)
    # Calculate DAMA/Gap scores (as used in Gap Analysis tab)
    data["Data Quality Score (%)"] = (data["Data Completeness (%)"] * np.random.uniform(0.85, 1.05)).clip(0, 100).round(1)
    data["Data Quality Gap (%)"] = 100 - data["Data Quality Score (%)"]
    data["Governance Gap (%)"] = 100 - data["Data Governance Score (%)"]
    data["Architecture Gap (%)"] = 100 - data["Data Architecture Score (%)"]
    data["Gap Severity"] = (data["Data Quality Gap (%)"] * 0.4 +
                            data["Governance Gap (%)"] * 0.3 +
                            data["Architecture Gap (%)"] * 0.3) * (data["Risk Score"] / 10)

    return data

# -----------------------------
# Utility & KPI Calculation Functions (Robust Error Handling Added)
# -----------------------------

# --- KPI 1: Performance against baseline dates ---
def calculate_schedule_performance(df):
    """Calculates days ahead/behind schedule based on Forecast vs Baseline SG1 Date."""
    col_name = 'Schedule Deviation (Days)'
    try:
        if 'Forecast SG1 Date' in df.columns and 'Baseline SG1 Date' in df.columns:
            # Ensure columns are datetime objects, coerce errors to NaT
            forecast_dt = pd.to_datetime(df['Forecast SG1 Date'], errors='coerce')
            baseline_dt = pd.to_datetime(df['Baseline SG1 Date'], errors='coerce')
            # Calculate difference only where both dates are valid
            valid_dates = forecast_dt.notna() & baseline_dt.notna()
            # Days ahead (negative) or behind (positive)
            df[col_name] = np.nan # Initialize column
            df.loc[valid_dates, col_name] = (forecast_dt[valid_dates] - baseline_dt[valid_dates]).dt.days
        else:
            df[col_name] = np.nan # Column missing
    except Exception as e:
        st.error(f"Error in calculate_schedule_performance: {e}")
        df[col_name] = np.nan # Ensure column exists even on error
    # Ensure column always exists
    if col_name not in df.columns: df[col_name] = np.nan
    return df

# --- KPI 2 & 3: Programme budget performance (Refactored Aggregation - FIX) ---
def calculate_programme_budget_performance(master_df):
    """
    Calculates programme-level budget KPIs (Forecast & Actual) using groupby.agg.
    Returns a DataFrame with Programme Name as index and KPI columns.
    Handles potential errors and missing columns gracefully.
    """
    # Define expected columns for the output DataFrame
    expected_cols = ['Forecast Budget Variance (%)', 'Forecast Budget RAG', 'Actual Budget Variance (%)', 'Actual Budget RAG']
    prog_kpi_df = pd.DataFrame(columns=expected_cols) # Initialize empty df with cols
    prog_names_index = pd.Index([], name='Programme Name') # Initialize empty index

    try:
        # Check if necessary source columns exist
        required_source_cols = ['Programme Name', 'Project Forecast Outturn (£K)', 'Project Actual Outturn (£K)', 'Programme IBP Budget (£K)']
        if not all(col in master_df.columns for col in required_source_cols):
            st.warning("Missing columns required for programme budget KPI calculation.")
            # Ensure the returned DataFrame has the expected columns, filled with defaults
            prog_kpi_df['Forecast Budget RAG'] = 'Unknown'
            prog_kpi_df['Actual Budget RAG'] = 'Unknown'
            # Try to add Programme Name index if possible
            if 'Programme Name' in master_df.columns:
                 prog_names = master_df['Programme Name'].unique()
                 prog_kpi_df = prog_kpi_df.reindex(pd.Index(prog_names, name='Programme Name'))
            return prog_kpi_df.reset_index() # Return with Programme Name if available, else empty

        # --- Pre-convert columns to numeric ---
        df_numeric = master_df.copy()
        for col in ['Project Forecast Outturn (£K)', 'Project Actual Outturn (£K)', 'Programme IBP Budget (£K)']:
            df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce')

        # Aggregate necessary data safely
        prog_agg = df_numeric.groupby('Programme Name').agg(
            Total_Forecast_Outturn=('Project Forecast Outturn (£K)', 'sum'), # Sum will ignore NaN
            Total_Actual_Outturn=('Project Actual Outturn (£K)', lambda x: x.sum(skipna=True)), # Sum only non-NaN actuals
            IBP_Budget=('Programme IBP Budget (£K)', 'first') # Use 'first' to get the first non-NaN budget in the group
        ) # Keep rows even if budget is NaN for now, handle in calculation

        prog_names_index = prog_agg.index # Store index before potential drop

        if not prog_agg.empty:
            # --- Calculate Forecast Variance ---
            valid_forecast_budget = prog_agg['IBP_Budget'].notna() & (prog_agg['IBP_Budget'] > 0)
            prog_agg['Forecast Budget Variance (%)'] = np.nan
            prog_agg.loc[valid_forecast_budget, 'Forecast Budget Variance (%)'] = \
                ((prog_agg['Total_Forecast_Outturn'] - prog_agg['IBP_Budget']) / prog_agg['IBP_Budget']) * 100

            # --- Calculate Forecast RAG ---
            conditions_forecast = [
                prog_agg['Forecast Budget Variance (%)'] < -10,
                prog_agg['Forecast Budget Variance (%)'] <= 0
            ]
            choices_forecast = ["Green", "Amber"]
            prog_agg['Forecast Budget RAG'] = np.select(conditions_forecast, choices_forecast, default="Red")
            prog_agg.loc[prog_agg['Forecast Budget Variance (%)'].isna(), 'Forecast Budget RAG'] = "Unknown"

            # --- Calculate Actual Variance ---
            valid_actual_budget = valid_forecast_budget & (prog_agg['Total_Actual_Outturn'] > 0) # Need actuals > 0 and valid budget
            prog_agg['Actual Budget Variance (%)'] = np.nan
            prog_agg.loc[valid_actual_budget, 'Actual Budget Variance (%)'] = \
                ((prog_agg['Total_Actual_Outturn'] - prog_agg['IBP_Budget']) / prog_agg['IBP_Budget']) * 100

            # --- Calculate Actual RAG ---
            conditions_actual = [
                prog_agg['Actual Budget Variance (%)'] < -10,
                prog_agg['Actual Budget Variance (%)'] <= 0
            ]
            choices_actual = ["Green", "Amber"]
            prog_agg['Actual Budget RAG'] = np.select(conditions_actual, choices_actual, default="Red")
            prog_agg.loc[prog_agg['Actual Budget Variance (%)'].isna(), 'Actual Budget RAG'] = "N/A" # Use N/A if variance couldn't be calculated

            # Select only the final KPI columns
            prog_kpi_df = prog_agg[expected_cols]
        else:
             st.info("No programmes found or no valid budget data after grouping.")
             # Ensure the returned DataFrame has the expected columns, filled with defaults
             prog_kpi_df = pd.DataFrame(index=prog_names_index, columns=expected_cols)
             prog_kpi_df['Forecast Budget RAG'] = 'Unknown'
             prog_kpi_df['Actual Budget RAG'] = 'Unknown'


    except Exception as e:
        st.error(f"Error calculating programme budget performance: {e}")
        # Return DataFrame with expected columns and Error status on failure
        # Get unique programme names if possible to build the index
        prog_names = master_df['Programme Name'].unique() if 'Programme Name' in master_df.columns else []
        prog_kpi_df = pd.DataFrame(index=pd.Index(prog_names, name='Programme Name'), columns=expected_cols)
        prog_kpi_df['Forecast Budget RAG'] = 'Error'
        prog_kpi_df['Actual Budget RAG'] = 'Error'

    # Ensure the function returns a DataFrame with 'Programme Name' as a column
    prog_kpi_df = prog_kpi_df.reset_index() # Make Programme Name a column

    # Ensure all expected columns are present, filling with defaults if necessary
    for col in expected_cols:
        if col not in prog_kpi_df.columns:
            if 'RAG' in col: prog_kpi_df[col] = 'Error'
            else: prog_kpi_df[col] = np.nan
    if 'Programme Name' not in prog_kpi_df.columns:
         prog_kpi_df['Programme Name'] = "" # Add empty Programme Name if missing


    return prog_kpi_df


# --- KPI 4: Solution cost relative to engineering budget ---
def calculate_solution_cost_performance(df):
    """Calculates Solution Cost % of Engineering Budget and RAG status."""
    pct_col = 'Solution Cost (%)'
    rag_col = 'Solution Cost RAG'
    try:
        if 'Engineering Design Cost (£K)' in df.columns and 'Engineering Budget (£K)' in df.columns:
            eng_cost = pd.to_numeric(df['Engineering Design Cost (£K)'], errors='coerce')
            eng_budget = pd.to_numeric(df['Engineering Budget (£K)'], errors='coerce')

            valid_calc = eng_cost.notna() & eng_budget.notna() & (eng_budget > 0)

            df[pct_col] = np.nan # Initialize column
            df.loc[valid_calc, pct_col] = (eng_cost[valid_calc] / eng_budget[valid_calc]) * 100

            conditions = [
                df[pct_col] < 2,
                df[pct_col] < 2.5,
                df[pct_col] >= 2.5
            ]
            choices = ["Green", "Amber", "Red"]
            df[rag_col] = np.select(conditions, choices, default="Unknown")
            df.loc[df[pct_col].isna(), rag_col] = "Unknown" # Ensure Unknown if calc failed
        else:
            df[pct_col] = np.nan
            df[rag_col] = "Unknown"
    except Exception as e:
        st.error(f"Error in calculate_solution_cost_performance: {e}")
        df[pct_col] = np.nan
        df[rag_col] = "Error" # Indicate error
    # Ensure columns always exist
    if pct_col not in df.columns: df[pct_col] = np.nan
    if rag_col not in df.columns: df[rag_col] = "Error"
    return df

# --- KPI 5: Completion of documentation ---
def calculate_documentation_rag(df):
    """Calculates RAG status for Documentation Completion."""
    rag_col = 'Documentation RAG'
    pct_col = 'Documentation Completion (%)'
    try:
        if pct_col in df.columns:
            doc_pct = pd.to_numeric(df[pct_col], errors='coerce')
            conditions = [
                doc_pct == 100,
                doc_pct >= 95,
                doc_pct < 95
            ]
            choices = ["Green", "Amber", "Red"]
            df[rag_col] = np.select(conditions, choices, default="Unknown")
            df.loc[doc_pct.isna(), rag_col] = "Unknown"
        else:
            df[rag_col] = "Unknown"
    except Exception as e:
        st.error(f"Error in calculate_documentation_rag: {e}")
        df[rag_col] = "Error"
    # Ensure column always exists
    if rag_col not in df.columns: df[rag_col] = "Error"
    return df

# --- KPI 6: Resourcing of programme ---
def calculate_resourcing_rag(df):
    """Calculates RAG status based on Resource Allocation End Date."""
    months_col = 'Months Allocated'
    rag_col = 'Resourcing RAG'
    try:
        if 'Resource Allocation End Date' in df.columns:
            res_end_date = pd.to_datetime(df['Resource Allocation End Date'], errors='coerce')
            today_dt = pd.to_datetime(date.today())

            delta_days = (res_end_date - today_dt).dt.days
            avg_days_in_month = 30.4375
            df[months_col] = delta_days / avg_days_in_month
            df[months_col] = df[months_col].fillna(-1) # Handle NaT/errors as < 3 months

            conditions = [
                df[months_col] >= 12,
                df[months_col] >= 3,
                df[months_col] < 3
            ]
            choices = ["Green", "Amber", "Red"]
            df[rag_col] = np.select(conditions, choices, default="Red")
        else:
            df[months_col] = np.nan
            df[rag_col] = "Unknown"
    except Exception as e:
        st.error(f"Error in calculate_resourcing_rag: {e}")
        df[months_col] = np.nan
        df[rag_col] = "Error"
    # Ensure columns always exist
    if months_col not in df.columns: df[months_col] = np.nan
    if rag_col not in df.columns: df[rag_col] = "Error"
    return df


# --- KPI 7: Health and Safety ---
# No RAG, just calculate percentages later during display.

# --- Helper: Get RAG color ---
def get_rag_color(rag_status):
    if rag_status == "Green": return THAMES_COLORS['Green']
    elif rag_status == "Amber": return THAMES_COLORS['Amber']
    elif rag_status == "Red": return THAMES_COLORS['Red']
    else: return THAMES_COLORS['Gray']

# --- Helper: Format KPI Value ---
def format_kpi_value(value, unit="", decimals=1):
    if pd.isna(value) or value == 'Error': # Check for 'Error' status text
        return "N/A"
    try:
        # Convert value to numeric if it's not already (might be string 'N/A' etc.)
        numeric_value = pd.to_numeric(value, errors='coerce')
        if pd.isna(numeric_value):
             # Check if original value was a status string we should preserve
             if isinstance(value, str) and value in ["N/A", "Unknown"]: return value
             return "N/A" # Return N/A if conversion fails otherwise

        if unit == "days":
            # Special formatting for days ahead/behind
            if numeric_value == 0: return f"On Time" # Removed unit as it's clear from context
            elif numeric_value < 0: return f"{abs(numeric_value):.0f} Days Ahead"
            else: return f"{numeric_value:.0f} Days Behind"
        elif unit == "Months":
             # Formatting for calculated months
             return f"{numeric_value:.1f}{unit}" # Show one decimal place for months
        else:
            # General numeric formatting
            return f"{numeric_value:,.{decimals}f}{unit}"
    except (ValueError, TypeError):
        return str(value) # Return as string if formatting fails

# --- Helper: Create KPI Card HTML ---
def create_kpi_card(title, value, unit="", decimals=1, rag_status=None, status_text=None):
    """Generates HTML for a KPI card with optional RAG status."""
    formatted_val = format_kpi_value(value, unit, decimals)
    card_html = f"""<div class="kpi-card"><h3>{title}</h3><p class="value">{formatted_val}</p>"""

    # Determine the status text and class
    display_status = "N/A" # Default status text
    status_class = "status-none" # Default class

    if rag_status and rag_status not in ["Unknown", "N/A", "Error"]:
        status_class = f"status-{rag_status.lower()}"
        display_status = status_text if status_text else rag_status
    elif status_text: # Handle cases with specific text but no standard RAG
        display_status = status_text
        # Keep status_class as status-none unless specific styling is needed
    elif unit == "days": # Special case for KPI 1 - status is implied by value format
         display_status = "" # Value already says "Ahead/Behind/On Time"
    elif value == "Error" or rag_status == "Error": # Explicitly handle error case
         display_status = "Error"
         status_class = "status-red" # Make errors visible
    elif formatted_val == "N/A": # If value is N/A, use that as status
         display_status = "N/A"
    elif rag_status == "N/A": # If RAG is N/A
         display_status = "N/A"

    # Add the status span if there's something to display
    if display_status:
        # Use <br> for line breaks if status_text contains it (for H&S card)
        if "<br>" in str(display_status):
             card_html += f"""<span class="status {status_class}">{display_status}</span>"""
        else:
             card_html += f"""<span class="status {status_class}">{display_status}</span>"""

    else:
        # Add an empty span only if needed for layout consistency (e.g., for KPI 1)
        if unit == "days":
             card_html += f"""<span class="status status-none"></span>"""

    card_html += "</div>"
    return card_html


# --- Helper: Create RAG Summary HTML for KPI Cards ---
def create_rag_summary_html(rag_counts):
    """Creates HTML snippet for RAG counts."""
    summary = "<div class='rag-summary'>"
    colors = {'Green': THAMES_COLORS['Green'], 'Amber': THAMES_COLORS['Amber'], 'Red': THAMES_COLORS['Red'], 'Unknown': THAMES_COLORS['Gray'], 'N/A': THAMES_COLORS['Gray'], 'Error': THAMES_COLORS['Red']}
    order = ['Red', 'Amber', 'Green', 'N/A', 'Unknown', 'Error'] # Display order
    has_counts = False
    for rag in order:
        if rag in rag_counts and rag_counts[rag] > 0:
            summary += f"<span style='color:{colors.get(rag, THAMES_COLORS['Gray'])};'><b>{rag}:</b> {rag_counts[rag]}</span>"
            has_counts = True
    if not has_counts:
        summary += "<span>N/A</span>" # Show N/A if no counts
    summary += "</div>"
    return summary


# --- Enhanced Commentary Function ---
@st.cache_data(ttl=600) # Cache for 10 mins
def generate_executive_commentary_kpi(df, unique_programmes_df):
    """Generates commentary focused on KPI insights."""
    try:
        if df is None or df.empty:
            return "No data available for the selected filters to generate commentary."

        num_sites = len(df)
        num_programmes = len(unique_programmes_df) if unique_programmes_df is not None else 0
        commentary = f"📊 Overview for **{num_sites}** selected site(s) across **{num_programmes}** programme(s). "

        # KPI 1: Schedule
        if 'Schedule Deviation (Days)' in df.columns:
            schedule_numeric = pd.to_numeric(df['Schedule Deviation (Days)'], errors='coerce')
            if schedule_numeric.notna().any():
                avg_dev = schedule_numeric.mean()
                min_dev = schedule_numeric.min()
                max_dev = schedule_numeric.max()
                schedule_text = format_kpi_value(avg_dev, 'days')
                commentary += f"Average schedule position is **{schedule_text}** (ranging from {format_kpi_value(min_dev, 'days')} to {format_kpi_value(max_dev, 'days')}). "

        # KPI 2: Budget Forecast
        if 'Forecast Budget RAG' in unique_programmes_df.columns:
            forecast_rag_counts = unique_programmes_df['Forecast Budget RAG'].value_counts()
            if 'Red' in forecast_rag_counts and forecast_rag_counts['Red'] > 0:
                commentary += f"<span style='color:{THAMES_COLORS['Red']};'>**{forecast_rag_counts['Red']}** programme(s) forecast over budget.</span> "
            elif 'Amber' in forecast_rag_counts and forecast_rag_counts['Amber'] > 0:
                 commentary += "Some programmes are close to budget limits. "
            else:
                 commentary += "Programme forecast budgets appear under control. "

        # KPI 6: Resourcing
        if 'Resourcing RAG' in unique_programmes_df.columns:
            resourcing_rag_counts = unique_programmes_df['Resourcing RAG'].value_counts()
            if 'Red' in resourcing_rag_counts and resourcing_rag_counts['Red'] > 0:
                 commentary += f"<span style='color:{THAMES_COLORS['Red']};'>**{resourcing_rag_counts['Red']}** programme(s) have short-term resourcing (<3 months).</span> "

        # KPI 5: Documentation
        if 'Documentation RAG' in df.columns:
            doc_rag_counts = df['Documentation RAG'].value_counts()
            if 'Red' in doc_rag_counts and doc_rag_counts['Red'] > 0:
                 commentary += f"<span style='color:{THAMES_COLORS['Red']};'>**{doc_rag_counts['Red']}** site(s) have low documentation completion (<95%).</span> "

        # KPI 7: H&S
        hs_issues = 0
        if 'HS File Signed' in df.columns: hs_issues += len(df[df['HS File Signed'] == False])
        if 'CDM Checklist Complete' in df.columns: hs_issues += len(df[df['CDM Checklist Complete'] == False])
        if hs_issues > 0:
             commentary += f"<span style='color:{THAMES_COLORS['Red']};'>**{hs_issues}** potential H&S compliance issues noted (file/checklist).</span> "

        commentary += " Review the **KPI Dashboard** for full details."
        return commentary
    except Exception as e:
        return f"Error generating commentary: {e}"


# --- Other Utility Functions (Logo, Styling) ---
def add_logo(logo_path="assets/thames_logo.png", width=150):
    # (Keep original logic from user code)
    try:
        with open(logo_path, "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")
        st.sidebar.markdown(
            f"""
            <div style="text-align: center; margin-bottom: 20px;">
                <img src="data:image/png;base64,{data}" width="{width}">
            </div>
            """,
            unsafe_allow_html=True,
        )
    except FileNotFoundError:
        st.sidebar.warning(f"Logo file not found at {logo_path}. Skipping logo display.")

def style_dataframe(df):
    """Applies conditional styling to a DataFrame for display."""
    if df is None or df.empty:
        return df

    # Keep existing styling rules, add new ones if needed
    styled_df = df.style

    # Example: Highlight schedule deviation
    if 'Schedule Deviation (Days)' in df.columns:
        # Apply styling only to numeric values, ignore NaNs
        styled_df = styled_df.applymap(lambda v: 'background-color: #FFDDCC' if pd.notna(v) and isinstance(v, (int, float)) and v > 14 else ('background-color: #CCFFCC' if pd.notna(v) and isinstance(v, (int, float)) and v < -14 else ''), subset=['Schedule Deviation (Days)'])

    # Example: Highlight RAG columns with lighter background
    rag_cols_to_style = ['Solution Cost RAG', 'Documentation RAG', 'Resourcing RAG', 'Forecast Budget RAG', 'Actual Budget RAG']
    for col in rag_cols_to_style:
         if col in df.columns:
             styled_df = styled_df.applymap(lambda v: f'background-color: {get_rag_color(v)}33' if pd.notna(v) and v not in ["Unknown", "N/A", "Error"] else '', subset=[col]) # Use lighter background, check for N/A etc.

    # Keep existing highlights (Risk, Progress, Variance, Severity)
    if 'Risk Score' in df.columns:
        styled_df = styled_df.apply(lambda s: ['background-color: #FFCCCC' if pd.notna(v) and v > 7.5 else '' for v in s], subset=['Risk Score'])
    if 'SG1 Progress (%)' in df.columns:
        styled_df = styled_df.apply(lambda s: ['background-color: #FFDDCC' if pd.notna(v) and v < 50 else '' for v in s], subset=['SG1 Progress (%)'])
    if 'Cost Variance (%)' in df.columns: # Old variance, might be removed later
        styled_df = styled_df.apply(lambda s: ['background-color: #FFDDCC' if pd.notna(v) and v > 5 else 'background-color: #CCFFCC' if pd.notna(v) and v < -5 else '' for v in s], subset=['Cost Variance (%)'])
    if 'Schedule Variance (%)' in df.columns: # Old variance, might be removed later
        styled_df = styled_df.apply(lambda s: ['background-color: #FFDDCC' if pd.notna(v) and v > 5 else 'background-color: #CCFFCC' if pd.notna(v) and v < -5 else '' for v in s], subset=['Schedule Variance (%)'])
    if 'Gap Severity' in df.columns:
        # Calculate median only on non-NaN values
        valid_severity = pd.to_numeric(df['Gap Severity'], errors='coerce').dropna() # Ensure numeric
        median_severity = valid_severity.median() if not valid_severity.empty else 0
        styled_df = styled_df.apply(lambda s: ['background-color: #FFE0CC' if pd.notna(v) and v > median_severity else '' for v in s], subset=['Gap Severity'])

    # Format floats (adjust precision as needed)
    float_cols = df.select_dtypes(include='float').columns
    format_dict = {col: '{:.1f}' for col in float_cols}
    # Override specific formats
    if 'Schedule Deviation (Days)' in format_dict: format_dict['Schedule Deviation (Days)'] = '{:.0f}'
    if 'Solution Cost (%)' in format_dict: format_dict['Solution Cost (%)'] = '{:.1f}%'
    if 'Forecast Budget Variance (%)' in format_dict: format_dict['Forecast Budget Variance (%)'] = '{:.1f}%'
    if 'Actual Budget Variance (%)' in format_dict: format_dict['Actual Budget Variance (%)'] = '{:.1f}%'
    if 'Documentation Completion (%)' in format_dict: format_dict['Documentation Completion (%)'] = '{:.1f}%'
    if 'HS File Signed (%)' in format_dict: format_dict['HS File Signed (%)'] = '{:.1f}%' # Calculated later
    if 'CDM Checklist Complete (%)' in format_dict: format_dict['CDM Checklist Complete (%)'] = '{:.1f}%' # Calculated later
    if 'Months Allocated' in format_dict: format_dict['Months Allocated'] = '{:.1f}' # Show months with one decimal

    # Apply formatting, replacing NaN with "N/A"
    styled_df = styled_df.format(format_dict, na_rep="N/A")

    # Format boolean columns (H&S)
    if 'HS File Signed' in df.columns:
         styled_df = styled_df.format({'HS File Signed': lambda x: 'Yes' if x==True else ('No' if x==False else 'N/A')})
    if 'CDM Checklist Complete' in df.columns:
         styled_df = styled_df.format({'CDM Checklist Complete': lambda x: 'Yes' if x==True else ('No' if x==False else 'N/A')})


    return styled_df


# --- Initialize Session State ---
if 'persona' not in st.session_state:
    st.session_state['persona'] = 'Programme Manager' # Default persona
# Initialize Data Discovery inputs in session state
if 'discovery_inputs' not in st.session_state:
    st.session_state['discovery_inputs'] = {} # Resetting this for the new structure
if 'master_data' not in st.session_state:
     st.session_state['master_data'] = None # Initialize as None

# -----------------------------
# Sidebar and Data Loading/Filtering
# -----------------------------

# --- Sidebar ---
# add_logo() # Uncomment if you have a logo file in ./assets/
st.sidebar.title("🌊 Thames Water SG1")
st.sidebar.markdown("---")

# Persona Selector
persona_options = ["Programme Manager", "Asset Manager", "Compliance Officer", "Sustainability Lead", "Data Analyst"]
# Ensure default is valid if session state holds an old value
current_persona = st.session_state.get('persona', 'Programme Manager')
if current_persona not in persona_options:
    current_persona = 'Programme Manager' # Default to a valid option
st.session_state['persona'] = st.sidebar.selectbox(
    "Select Your Persona",
    persona_options,
    index=persona_options.index(current_persona),
    help="Select your role to tailor insights and discovery questions."
)
st.sidebar.markdown(f"*Viewing as: {st.session_state['persona']}*")
st.sidebar.markdown("---")

# File Uploader
uploaded_file = st.sidebar.file_uploader(
    "Upload Site Data (CSV)", type="csv",
    help="Upload CSV with required columns (e.g., Site, Programme Name, Baseline SG1 Date, Forecast SG1 Date, Budgets, Costs etc.)"
)

# --- Data Loading ---
master_df = None # Initialize master_df
if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        # --- Basic Validation ---
        # Check for a few essential columns from the new structure
        required_cols = ['Site', 'Programme Name', 'Baseline SG1 Date', 'Forecast SG1 Date', 'Project Forecast Outturn (£K)', 'Programme IBP Budget (£K)']
        if not all(col in data.columns for col in required_cols):
            st.sidebar.error(f"CSV missing required columns. Need at least: {', '.join(required_cols)}")
            if st.session_state['master_data'] is None: # Only load mock if no data exists at all
                st.session_state['master_data'] = create_mock_data()
                st.sidebar.info("Loaded Mock Data due to upload error.")
            master_df = st.session_state.get('master_data') # Use existing if available
        else:
            # --- Data Type Conversion & Validation ---
            date_cols = ['Baseline SG1 Date', 'Forecast SG1 Date', 'Resource Allocation End Date']
            for col in date_cols:
                if col in data.columns:
                    data[col] = pd.to_datetime(data[col], errors='coerce') # Convert dates, handle errors

            numeric_cols = ['Project Forecast Outturn (£K)', 'Programme IBP Budget (£K)',
                            'Project Actual Outturn (£K)', 'Engineering Design Cost (£K)',
                            'Engineering Budget (£K)', 'Documentation Completion (%)',
                            'Risk Score', 'Carbon Impact (tCO2e)', 'Cost Variance (%)',
                            'Schedule Variance (%)', 'Innovation Potential (%)',
                            'SG1 Progress (%)', 'Data Completeness (%)', # Add others from original
                            'Data Governance Score (%)', 'Data Architecture Score (%)',
                            'Compliance Score (%)', 'Gap Severity' # Ensure Gap Severity is numeric
                            ]
            for col in numeric_cols:
                 if col in data.columns:
                     data[col] = pd.to_numeric(data[col], errors='coerce') # Convert numerics, handle errors

            # Ensure boolean columns are boolean
            bool_cols = ['HS File Signed', 'CDM Checklist Complete']
            for col in bool_cols:
                if col in data.columns:
                    # Attempt conversion, map common strings, default to False if ambiguous
                    if data[col].dtype == 'object':
                        data[col] = data[col].str.lower().map({'true': True, 'yes': True, '1': True, 'false': False, 'no': False, '0': False}).fillna(False)
                    data[col] = data[col].astype(bool)


            st.session_state['master_data'] = data
            master_df = data # Use uploaded data
            st.sidebar.success("CSV Uploaded & Processed!")
            # Clear caches that depend on data
            generate_executive_commentary_kpi.clear() # Use the new commentary function name
            # Add other function clears if needed

    except Exception as e:
        st.sidebar.error(f"Error reading/processing CSV: {e}")
        if st.session_state['master_data'] is None:
            st.session_state['master_data'] = create_mock_data()
            st.sidebar.info("Loaded Mock Data due to upload error.")
        master_df = st.session_state.get('master_data') # Use existing if available
else:
    # Load mock data if no file uploaded and no data in session state yet
    if st.session_state.get('master_data') is None:
        st.session_state['master_data'] = create_mock_data()
        st.sidebar.info("Using Mock Data")
    master_df = st.session_state.get('master_data')

# --- Perform KPI Calculations on Master Data ---
# Crucially, perform calculations *before* filtering
if master_df is not None and not master_df.empty:
    calculation_errors = False
    error_messages = []
    master_df_processed = master_df.copy() # Work on a copy
    try:
        # Apply calculations that add columns to the dataframe
        master_df_processed = calculate_schedule_performance(master_df_processed)
        master_df_processed = calculate_solution_cost_performance(master_df_processed)
        master_df_processed = calculate_documentation_rag(master_df_processed)
        master_df_processed = calculate_resourcing_rag(master_df_processed) # This adds 'Months Allocated' and 'Resourcing RAG'

        # Programme level calculations (Refactored)
        if 'Programme Name' in master_df_processed.columns:
            try:
                prog_budget_kpis = calculate_programme_budget_performance(master_df_processed)
                # Merge programme level KPIs back to the main df
                # Ensure 'Programme Name' exists in prog_budget_kpis before merge
                if 'Programme Name' in prog_budget_kpis.columns:
                     master_df_processed = master_df_processed.merge(prog_budget_kpis, on='Programme Name', how='left')
                else:
                     # Handle case where budget calc failed and returned empty/wrong df
                     calculation_errors = True
                     error_messages.append("Programme budget KPI calculation failed to return expected structure.")
                     for col in ['Forecast Budget Variance (%)', 'Forecast Budget RAG', 'Actual Budget Variance (%)', 'Actual Budget RAG']:
                         if col not in master_df_processed.columns: master_df_processed[col] = np.nan if 'Variance' in col else 'Error'

            except Exception as agg_e:
                 calculation_errors = True
                 error_messages.append(f"Error during programme budget KPI calculation/merge: {agg_e}")
                 # Ensure columns exist even if merge fails
                 for col in ['Forecast Budget Variance (%)', 'Forecast Budget RAG', 'Actual Budget Variance (%)', 'Actual Budget RAG']:
                     if col not in master_df_processed.columns: master_df_processed[col] = np.nan if 'Variance' in col else 'Error'
        else:
            st.warning("Cannot calculate programme-level budget KPIs without 'Programme Name' column.")
            # Add placeholder columns if they don't exist to prevent errors later
            for col in ['Forecast Budget Variance (%)', 'Forecast Budget RAG', 'Actual Budget Variance (%)', 'Actual Budget RAG']:
                if col not in master_df_processed.columns: master_df_processed[col] = np.nan if 'Variance' in col else 'Unknown'

        # Ensure all potentially calculated columns exist, even if calculation failed
        expected_cols = ['Schedule Deviation (Days)', 'Solution Cost (%)', 'Solution Cost RAG',
                         'Documentation RAG', 'Months Allocated', 'Resourcing RAG',
                         'Forecast Budget Variance (%)', 'Forecast Budget RAG',
                         'Actual Budget Variance (%)', 'Actual Budget RAG']
        for col in expected_cols:
             if col not in master_df_processed.columns:
                 calculation_errors = True # Mark error if expected col is missing
                 error_messages.append(f"Column '{col}' was not created during KPI calculation.")
                 if 'RAG' in col or 'RAG' in col: master_df_processed[col] = 'Error' # Use Error status for RAG
                 else: master_df_processed[col] = np.nan

        # Store the potentially modified master_df
        st.session_state['master_data'] = master_df_processed
        master_df = master_df_processed # Use the processed df going forward

        # Display any calculation errors prominently if they occurred
        if calculation_errors:
             st.error("⚠️ Errors occurred during KPI calculations. Some KPIs may show 'Error' or 'N/A'. Please check data quality and calculation logic. Details logged above/in console.")
             # Optionally log detailed errors here if needed, e.g. st.expander("Show Errors")... st.error(msg)

    except Exception as e:
        st.error(f"Critical error during KPI calculation phase: {e}")
        st.code(traceback.format_exc())
        calculation_errors = True
        # Ensure master_df is still assigned even if calculations fail partially
        if 'master_data' not in st.session_state or st.session_state['master_data'] is None:
             st.session_state['master_data'] = create_mock_data() # Fallback to mock
        master_df = st.session_state.get('master_data')

else:
     st.warning("No data loaded. Cannot perform KPI calculations.")


# --- Sidebar Thresholds & Options (Keep Benchmarking for now) ---
st.sidebar.markdown("---")
st.sidebar.subheader("Benchmarking Options")
show_overall_benchmark = st.sidebar.checkbox("Show Overall Average on Charts", value=False, key='benchmark_overall')
show_catchment_benchmark = st.sidebar.checkbox("Show Catchment Average on Charts", value=False, key='benchmark_catchment')

# --- Global Filters ---
st.sidebar.markdown("---")
st.sidebar.subheader("Global Filters")

filtered_df = pd.DataFrame() # Initialize filtered_df

if master_df is not None and not master_df.empty:
    # Filter by Programme first (if column exists)
    programme_filtered_df = pd.DataFrame() # Intermediate df for programme filter
    if 'Programme Name' in master_df.columns:
        available_programmes = sorted(master_df['Programme Name'].unique())
        selected_programmes = st.sidebar.multiselect(
            "Filter by Programme", options=available_programmes, default=available_programmes, key="programme_filter"
        )
        if selected_programmes:
            programme_filtered_df = master_df[master_df['Programme Name'].isin(selected_programmes)].copy()
        else:
            programme_filtered_df = pd.DataFrame(columns=master_df.columns) # Empty if no programme selected
    else:
        programme_filtered_df = master_df.copy() # Use all data if no programme column

    # Filter by Catchment (on the already programme-filtered data)
    catchment_filtered_df = pd.DataFrame() # Intermediate df
    if 'Catchment' in programme_filtered_df.columns and not programme_filtered_df.empty:
        available_catchments = sorted(programme_filtered_df['Catchment'].unique())
        selected_catchments = st.sidebar.multiselect(
            "Filter by Catchment", options=available_catchments, default=available_catchments, key="catchment_filter"
        )
        if selected_catchments:
             catchment_filtered_df = programme_filtered_df[programme_filtered_df['Catchment'].isin(selected_catchments)].copy()
        else:
             # If catchments were available but none selected, result is empty
             if available_catchments:
                 catchment_filtered_df = pd.DataFrame(columns=master_df.columns)
             else: # No catchments were available in the first place
                  catchment_filtered_df = programme_filtered_df.copy()
    else:
         catchment_filtered_df = programme_filtered_df.copy() # Pass through if no catchment column


    # Filter by Site (on the already filtered data)
    if 'Site' in catchment_filtered_df.columns and not catchment_filtered_df.empty:
        available_sites = sorted(catchment_filtered_df['Site'].unique())
        selected_sites = st.sidebar.multiselect(
            "Filter by Site", options=available_sites, default=[], key="site_filter",
            help="Select specific sites. Leave blank for all sites in selected programmes/catchments."
        )
        if selected_sites:
            filtered_df = catchment_filtered_df[catchment_filtered_df['Site'].isin(selected_sites)].copy()
        else:
             filtered_df = catchment_filtered_df.copy() # No site filter applied
    else:
         filtered_df = catchment_filtered_df.copy() # Pass through if no site column


else:
    # If master_df itself is None or empty
    st.error("Master data is missing or empty. Cannot apply filters.")
    filtered_df = pd.DataFrame() # Ensure filtered_df is an empty DataFrame

# Check if filtered data is empty after filtering
if master_df is not None and not master_df.empty and filtered_df.empty:
    # Check if the emptiness was caused by filters
    programme_filter_active = 'Programme Name' in master_df.columns and selected_programmes != available_programmes
    catchment_filter_active = 'Catchment' in master_df.columns and 'Catchment' in programme_filtered_df.columns and selected_catchments != available_catchments
    site_filter_active = 'Site' in master_df.columns and 'Site' in catchment_filtered_df.columns and selected_sites

    if programme_filter_active or catchment_filter_active or site_filter_active:
        st.warning("No data matches the current filter criteria. Adjust filters in the sidebar.")


# --- Export Options ---
st.sidebar.markdown("---")
st.sidebar.subheader("Export Options")

def get_table_download_link(df, filename="thames_water_data.csv", text="Download Filtered Data as CSV"):
    """Generates a download link for a DataFrame."""
    if df is None or df.empty:
        return "<span style='color: white; margin: 0.5rem 1rem; display:block; text-align:center;'>No data to download.</span>"
    # Convert datetime columns to string to avoid timezone issues in CSV
    df_export = df.copy()
    for col in df_export.select_dtypes(include=['datetime64[ns]', 'datetime64[ns, UTC]']).columns:
         # Format dates, handle NaT gracefully
         df_export[col] = df_export[col].dt.strftime('%Y-%m-%d').replace('NaT', '')

    csv = df_export.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    # Using direct style for simplicity here:
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" style="display: inline-block; padding: 0.5rem 1rem; background-color: {THAMES_COLORS["Secondary Blue"]}; color: white; border-radius: 5px; text-decoration: none; width: calc(100% - 2rem); text-align: center; margin: 0.5rem 1rem; transition: background-color 0.2s ease, transform 0.1s ease;" onmouseover="this.style.backgroundColor=\'#007bbb\'; this.style.transform=\'scale(1.02)\';" onmouseout="this.style.backgroundColor=\'{THAMES_COLORS["Secondary Blue"]}\'; this.style.transform=\'scale(1)\';" onmousedown="this.style.transform=\'scale(0.98)\';" onmouseup="this.style.transform=\'scale(1.02)\';">{text}</a>'
    return href

st.sidebar.markdown(get_table_download_link(filtered_df), unsafe_allow_html=True)
st.sidebar.info("Note: PDF/PowerPoint export needs specific library setup.")

# --- Display Active Filters ---
active_filters_msg = "Filters Applied: "
if master_df is not None and not master_df.empty:
    filters_applied = False
    # Check Programme Filter
    if 'Programme Name' in master_df.columns:
        current_programmes = filtered_df['Programme Name'].unique() if not filtered_df.empty else []
        all_programmes = master_df['Programme Name'].unique()
        if len(current_programmes) < len(all_programmes):
            active_filters_msg += f"Programme(s): {', '.join(current_programmes)}; "
            filters_applied = True
        elif len(current_programmes) == 0 and len(all_programmes) > 0:
             active_filters_msg += "NO Programmes Selected; "
             filters_applied = True
        else:
             active_filters_msg += "All Programmes; "

    # Check Catchment Filter
    if 'Catchment' in master_df.columns:
        current_catchments = filtered_df['Catchment'].unique() if not filtered_df.empty else []
        # Determine available catchments based on *initially* selected programmes
        initial_programmes = master_df['Programme Name'].unique() if 'Programme Name' not in master_df.columns else selected_programmes
        all_available_catchments = master_df[master_df['Programme Name'].isin(initial_programmes)]['Catchment'].unique() if 'Programme Name' in master_df.columns else master_df['Catchment'].unique()

        if len(current_catchments) < len(all_available_catchments):
             active_filters_msg += f"Catchment(s): {', '.join(current_catchments)}; "
             filters_applied = True
        elif len(current_catchments) == 0 and len(all_available_catchments) > 0:
             active_filters_msg += "NO Catchments Selected; "
             filters_applied = True
        elif not filters_applied: # Only add 'All Catchments' if no programme filter was applied
             active_filters_msg += "All Catchments; "

    # Check Site Filter
    if 'Site' in master_df.columns:
        current_sites = filtered_df['Site'].unique() if not filtered_df.empty else []
        # Determine available sites based on *initially* selected programmes/catchments
        initial_programmes = master_df['Programme Name'].unique() if 'Programme Name' not in master_df.columns else selected_programmes
        initial_catchments = master_df['Catchment'].unique() if 'Catchment' not in master_df.columns else selected_catchments
        prog_filt = master_df['Programme Name'].isin(initial_programmes) if 'Programme Name' in master_df.columns else True
        catch_filt = master_df['Catchment'].isin(initial_catchments) if 'Catchment' in master_df.columns else True
        all_available_sites = master_df[prog_filt & catch_filt]['Site'].unique()

        if len(current_sites) < len(all_available_sites):
            active_filters_msg += f"Site(s): {', '.join(current_sites)}"
            filters_applied = True
        elif not filters_applied: # Only add 'All Sites' if no other filters applied
             active_filters_msg += "All Sites"

    if not filters_applied:
         active_filters_msg += "None"


if len(active_filters_msg) > len("Filters Applied: "):
    st.caption(f"```{active_filters_msg}```") # Use code block for visibility


# -----------------------------
# Main Dashboard Tabs (NEW ORDER CONFIRMED)
# -----------------------------
tab_landing, tab_summary, tab_kpi, tab_discovery, tab_gap, tab_prog, tab_adv = st.tabs([
    "👋 Landing Page",
    "🏠 Executive Summary",
    "🏆 KPI Dashboard",
    "🔍 Data Discovery",
    "📊 Gap Analysis",
    "📈 Programme Reporting",
    "📉 Advanced Analytics"
])

# =============================
# 👋 Tab Landing: Landing Page
# =============================
with tab_landing:
    st.header("👋 Welcome to the Thames Water SG1 Insights Dashboard")
    st.markdown("---")

    # Section 1: Purpose (Using standard Markdown)
    # Use a CSS class to apply container styling to markdown
    st.markdown("""
    <div class="landing-container landing-container-markdown">
        <h1>Purpose</h1>
        <p>
            This dashboard provides insights into the Stage Gate 1 (SG1) programme, focusing on key performance indicators (KPIs),
            data readiness, risk analysis, and overall programme health. It aims to support Programme Managers, Asset Managers,
            Compliance Officers, and other stakeholders in monitoring progress and making informed decisions.
        </p>
        <p>
            Navigate through the tabs at the top to explore different aspects of the SG1 programme.
            Use the sidebar to filter data or upload your own dataset.
        </p>
    </div>
    """, unsafe_allow_html=True) # unsafe needed for the div class

    # Section 2: Data Flow Diagram Placeholder (Simplified Markdown Table in HTML)
    st.markdown("""
    <div class="landing-container" style="margin-top: 2rem;">
        <h2>Data Flow Overview (Conceptual)</h2>
        <p>
            The accuracy of the KPIs in this dashboard depends on the availability and quality of data from various sources.
            The <b>Data Discovery</b> tab helps assess the readiness of these sources. Based on those findings,
            this conceptual diagram can be updated to reflect the actual data landscape:
        </p>
        <div class="diagram-placeholder">
             <table>
                <thead>
                    <tr>
                        <th>Source Systems (Examples)</th>
                        <th></th> <th>Data Elements (Examples)</th>
                        <th></th> <th>KPIs Calculated</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td> <ul>
                                <li>SAP / Finance System</li>
                                <li>Maximo / Asset Mgmt</li>
                                <li>Planning Tools (P6, etc.)</li>
                                <li>BCDE (Business Collaborator)</li>
                                <li>Resource Mgmt Tool</li>
                                <li>H&S Systems/Registers</li>
                                <li>Other Databases/Spreadsheets</li>
                            </ul>
                        </td>
                        <td class="arrow">&rarr;</td>
                        <td> <ul>
                                <li>Baseline/Forecast Dates</li>
                                <li>IBP Budgets</li>
                                <li>Forecast/Actual Outturns</li>
                                <li>Engineering Costs/Budgets</li>
                                <li>Document Status (BCDE)</li>
                                <li>Resource Allocations</li>
                                <li>H&S File/Checklist Status</li>
                            </ul>
                        </td>
                        <td class="arrow">&rarr;</td>
                        <td> <ul>
                                <li>KPI 1: Schedule Perf.</li>
                                <li>KPI 2: Budget Perf. (Fcst)</li>
                                <li>KPI 3: Budget Perf. (Act)</li>
                                <li>KPI 4: Solution Cost %</li>
                                <li>KPI 5: Doc Completion %</li>
                                <li>KPI 6: Resourcing Alloc.</li>
                                <li>KPI 7: H&S Completion %</li>
                            </ul>
                        </td>
                    </tr>
                </tbody>
            </table>
            <p class="update-note">
                <i>(Update this section based on findings from the Data Discovery tab to illustrate
                the actual systems, data points, and processes involved in generating the KPI data.)</i>
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Section 3: Note about Mock Data
    st.markdown("""
    <div class="landing-container" style="margin-top: 2rem; background-color: #e7f3f7; border-left: 5px solid #00A1D6;">
        <p>
            <i><b>Note:</b> This dashboard currently uses mock data. Uploading real data via the sidebar is required for accurate insights specific to your programmes.</i>
        </p>
    </div>
    """, unsafe_allow_html=True)


# =============================
# 🏠 Tab Summary: Executive Summary (KPI Focused)
# =============================
with tab_summary:
    st.header("🏠 Executive Summary")
    st.markdown("High-level overview of programme status based on key performance indicators.")

    if filtered_df is None or filtered_df.empty:
        st.warning("No data available for the selected filters.")
    else:
        # --- KPI Summary Cards ---
        st.subheader("KPI Snapshot")
        col1, col2, col3, col4 = st.columns(4)

        # Calculate overall KPI summaries
        avg_schedule_dev = pd.to_numeric(filtered_df.get('Schedule Deviation (Days)'), errors='coerce').mean()

        # Programme-level KPIs need care if filtered_df has multiple rows per programme
        # Get unique programme rows first
        prog_cols = ['Programme Name', 'Forecast Budget RAG', 'Actual Budget RAG', 'Resourcing RAG', 'Forecast Budget Variance (%)'] # Added Variance
        prog_cols_exist = [col for col in prog_cols if col in filtered_df.columns]
        unique_programmes_df = filtered_df[prog_cols_exist].drop_duplicates(subset=['Programme Name']) if 'Programme Name' in prog_cols_exist else pd.DataFrame(columns=prog_cols_exist)

        # Count RAG statuses for key programme KPIs
        forecast_rag_counts = unique_programmes_df['Forecast Budget RAG'].value_counts() if 'Forecast Budget RAG' in unique_programmes_df else pd.Series(dtype=int)
        resourcing_rag_counts = unique_programmes_df['Resourcing RAG'].value_counts() if 'Resourcing RAG' in unique_programmes_df else pd.Series(dtype=int)

        # Site-level KPI summaries
        doc_rag_counts = filtered_df['Documentation RAG'].value_counts() if 'Documentation RAG' in filtered_df else pd.Series(dtype=int)
        # sol_cost_rag_counts = filtered_df['Solution Cost RAG'].value_counts() if 'Solution Cost RAG' in filtered_df else pd.Series(dtype=int) # Can add this back if needed

        # H&S Summary
        hs_files_ok = filtered_df['HS File Signed'].sum() if 'HS File Signed' in filtered_df else 0
        cdm_ok = filtered_df['CDM Checklist Complete'].sum() if 'CDM Checklist Complete' in filtered_df else 0
        total_sites_hs = len(filtered_df)


        # Display KPI summaries
        with col1:
            st.markdown(create_kpi_card("Avg Schedule Perf.", avg_schedule_dev, unit="days", decimals=0), unsafe_allow_html=True)
            st.caption("Average deviation from baseline SG1 date across selected sites.")

        with col2:
            # Show budget forecast RAG distribution (Programme Level)
            rag_summary_html = create_rag_summary_html(forecast_rag_counts)
            overall_fc_rag = "Red" if forecast_rag_counts.get('Red', 0) > 0 else \
                             "Amber" if forecast_rag_counts.get('Amber', 0) > 0 else \
                             "Green" if forecast_rag_counts.get('Green', 0) > 0 else "Unknown"
            st.markdown(create_kpi_card("Prog Fcst Budget Perf.", value=f"{len(unique_programmes_df)} Progs", rag_status=overall_fc_rag, status_text=rag_summary_html), unsafe_allow_html=True)
            st.caption("Programme forecast budget vs. IBP RAG distribution.")


        with col3:
             # Show Resourcing RAG distribution (Programme Level)
             rag_summary_html = create_rag_summary_html(resourcing_rag_counts)
             overall_res_rag = "Red" if resourcing_rag_counts.get('Red', 0) > 0 else \
                              "Amber" if resourcing_rag_counts.get('Amber', 0) > 0 else \
                              "Green" if resourcing_rag_counts.get('Green', 0) > 0 else "Unknown"
             st.markdown(create_kpi_card("Prog Resourcing Status", value=f"{len(unique_programmes_df)} Progs", rag_status=overall_res_rag, status_text=rag_summary_html), unsafe_allow_html=True)
             st.caption("Programme resource allocation horizon RAG distribution.")

        with col4:
            # Show H&S Completion Summary (Site Level)
            hs_value = f"{hs_files_ok}/{total_sites_hs}"
            cdm_value = f"{cdm_ok}/{total_sites_hs}"
            hs_status = "⚠️ Issues" if hs_files_ok < total_sites_hs or cdm_ok < total_sites_hs else "✅ OK"
            hs_rag = "Red" if hs_files_ok < total_sites_hs or cdm_ok < total_sites_hs else "Green"
            st.markdown(create_kpi_card("Site H&S Compliance", value=hs_status, rag_status=hs_rag, status_text=f"File: {hs_value}<br>CDM: {cdm_value}"), unsafe_allow_html=True)
            st.caption("Sites with H&S File Signed & CDM Checklist Complete.")


        st.divider() # Visual separation

        # --- Commentary & Focus ---
        col_commentary, col_focus = st.columns([2,1])

        with col_commentary:
            st.subheader("📝 Automated Commentary")
            # Generate commentary focusing on the KPI summaries calculated above
            comment = generate_executive_commentary_kpi(filtered_df, unique_programmes_df)
            st.markdown(f"<div class='insight-box'>{comment}</div>", unsafe_allow_html=True)

            # --- Schedule Deviation Bar Chart ---
            st.subheader("📊 Avg Schedule Deviation by Programme")
            if 'Programme Name' in filtered_df.columns and 'Schedule Deviation (Days)' in filtered_df.columns:
                 schedule_dev_prog = filtered_df.groupby('Programme Name')['Schedule Deviation (Days)'].mean().reset_index().sort_values('Schedule Deviation (Days)')
                 if not schedule_dev_prog.empty:
                      fig_sched_bar = px.bar(schedule_dev_prog,
                                             x='Schedule Deviation (Days)',
                                             y='Programme Name',
                                             orientation='h',
                                             title="Average Schedule Deviation by Programme",
                                             labels={'Schedule Deviation (Days)': 'Avg Deviation (Days)', 'Programme Name': 'Programme'},
                                             text='Schedule Deviation (Days)')
                      fig_sched_bar.update_traces(texttemplate='%{text:.0f}d', textposition='outside')
                      fig_sched_bar.update_layout(yaxis={'categoryorder':'total ascending'}, uniformtext_minsize=8, uniformtext_mode='hide')
                      st.plotly_chart(fig_sched_bar, use_container_width=True)
                 else:
                      st.info("No schedule deviation data to display by programme.")
            else:
                 st.info("Required columns ('Programme Name', 'Schedule Deviation (Days)') not available for chart.")


        with col_focus:
             # --- Priority Focus Box (Enhanced) ---
             st.subheader("⚡ Priority Focus Areas")

             # 1. Programmes Forecast Over Budget
             st.markdown("**Programmes Forecast Over Budget (Red):**")
             if 'Forecast Budget RAG' in unique_programmes_df.columns:
                 red_budget_progs = unique_programmes_df[unique_programmes_df['Forecast Budget RAG'] == 'Red']
                 if not red_budget_progs.empty:
                     focus_cols = ['Programme Name', 'Forecast Budget Variance (%)']
                     focus_cols_exist = [col for col in focus_cols if col in red_budget_progs.columns]
                     if focus_cols_exist:
                          display_df = red_budget_progs[focus_cols_exist].sort_values('Forecast Budget Variance (%)', ascending=False)
                          st.dataframe(style_dataframe(display_df), hide_index=True, use_container_width=True, height=min(150, (len(display_df)+1)*35)) # Dynamic height
                     else:
                          st.write(red_budget_progs['Programme Name'].tolist())
                 else:
                     st.success("✅ None")
             else:
                 st.warning("Budget data unavailable.")

             # 2. Sites Significantly Behind Schedule
             st.markdown("**Sites >30 Days Behind Schedule:**")
             if 'Schedule Deviation (Days)' in filtered_df.columns:
                 schedule_numeric = pd.to_numeric(filtered_df['Schedule Deviation (Days)'], errors='coerce')
                 behind_schedule_sites = filtered_df[schedule_numeric > 30].sort_values('Schedule Deviation (Days)', ascending=False)
                 if not behind_schedule_sites.empty:
                     focus_cols = ['Site', 'Programme Name', 'Schedule Deviation (Days)']
                     focus_cols_exist = [col for col in focus_cols if col in behind_schedule_sites.columns]
                     if focus_cols_exist:
                          st.dataframe(style_dataframe(behind_schedule_sites[focus_cols_exist].head(5)), hide_index=True, use_container_width=True, height=min(200, (len(behind_schedule_sites.head(5))+1)*35))
                     else:
                          st.write(behind_schedule_sites['Site'].head(5).tolist())
                 else:
                     st.success("✅ None")
             else:
                 st.warning("Schedule data unavailable.")

             # 3. Sites Failing H&S Checks
             st.markdown("**Sites Failing H&S Checks (KPI 7):**")
             hs_cols = ['HS File Signed', 'CDM Checklist Complete']
             if all(col in filtered_df.columns for col in hs_cols):
                  failed_hs_sites = filtered_df[(filtered_df['HS File Signed'] == False) | (filtered_df['CDM Checklist Complete'] == False)]
                  if not failed_hs_sites.empty:
                      focus_cols = ['Site', 'Programme Name', 'HS File Signed', 'CDM Checklist Complete']
                      focus_cols_exist = [col for col in focus_cols if col in failed_hs_sites.columns]
                      if focus_cols_exist:
                           st.dataframe(style_dataframe(failed_hs_sites[focus_cols_exist].head(5)), hide_index=True, use_container_width=True, height=min(200, (len(failed_hs_sites.head(5))+1)*35))
                      else:
                           st.write(failed_hs_sites['Site'].head(5).tolist())
                  else:
                      st.success("✅ None")
             else:
                  st.warning("H&S data unavailable.")


# =============================
# 🏆 Tab KPI: KPI Dashboard
# =============================
with tab_kpi:
    st.header("🏆 Key Performance Indicator (KPI) Dashboard")
    st.markdown("Overview of performance against defined business KPIs.")

    if filtered_df is None or filtered_df.empty:
        st.warning("No data available for the selected filters.")
    else:
        # --- Calculate Programme-Level Aggregations for Display ---
        programme_summary = pd.DataFrame()
        # Check required columns exist before attempting aggregation
        required_agg_cols = ['Programme Name', 'Schedule Deviation (Days)', 'HS File Signed',
                           'CDM Checklist Complete', 'Forecast Budget Variance (%)', 'Forecast Budget RAG',
                           'Actual Budget Variance (%)', 'Actual Budget RAG', 'Resourcing RAG',
                           'Months Allocated', 'Solution Cost (%)', 'Solution Cost RAG',
                           'Documentation Completion (%)', 'Documentation RAG']
        # Check if all required columns are present in the filtered dataframe
        all_cols_present = all(col in filtered_df.columns for col in required_agg_cols)

        if all_cols_present:
            try:
                # Aggregate KPI 1 (Schedule Deviation) - Average for simplicity
                kpi1_agg = filtered_df.groupby('Programme Name')['Schedule Deviation (Days)'].mean()

                # Aggregate H&S (KPI 7)
                kpi7a_agg = filtered_df.groupby('Programme Name')['HS File Signed'].mean() * 100 # Percentage True
                kpi7b_agg = filtered_df.groupby('Programme Name')['CDM Checklist Complete'].mean() * 100 # Percentage True

                # Get unique programme-level KPIs (already calculated and merged)
                # Ensure the columns actually exist before trying to select them
                prog_kpi_cols_to_select = ['Programme Name'] + [col for col in [
                                        'Forecast Budget Variance (%)', 'Forecast Budget RAG',
                                        'Actual Budget Variance (%)', 'Actual Budget RAG',
                                        'Resourcing RAG', 'Months Allocated'] if col in filtered_df.columns]

                prog_kpis = filtered_df[prog_kpi_cols_to_select].drop_duplicates(subset=['Programme Name'])

                # Combine aggregated and unique programme KPIs
                programme_summary = prog_kpis.set_index('Programme Name')

                # Add aggregated values safely using .get() on the index
                programme_summary['Avg Schedule Deviation (Days)'] = programme_summary.index.map(kpi1_agg.get)
                programme_summary['HS File Signed (%)'] = programme_summary.index.map(kpi7a_agg.get)
                programme_summary['CDM Checklist Complete (%)'] = programme_summary.index.map(kpi7b_agg.get)


                # Aggregate Site-Level KPIs for Programme View (KPI 4, 5) - Average
                # Use .get() with default NaN to handle potentially missing columns after filtering
                if 'Solution Cost (%)' in filtered_df.columns:
                    programme_summary['Avg Solution Cost (%)'] = programme_summary.index.map(filtered_df.groupby('Programme Name')['Solution Cost (%)'].mean().get)
                else: programme_summary['Avg Solution Cost (%)'] = np.nan

                if 'Solution Cost RAG' in filtered_df.columns:
                    programme_summary['Solution Cost RAG (Mode)'] = programme_summary.index.map(filtered_df.groupby('Programme Name')['Solution Cost RAG'].agg(lambda x: x.mode()[0] if not x.mode().empty else 'Unknown').get)
                else: programme_summary['Solution Cost RAG (Mode)'] = 'Unknown'

                if 'Documentation Completion (%)' in filtered_df.columns:
                    programme_summary['Avg Documentation Completion (%)'] = programme_summary.index.map(filtered_df.groupby('Programme Name')['Documentation Completion (%)'].mean().get)
                else: programme_summary['Avg Documentation Completion (%)'] = np.nan

                if 'Documentation RAG' in filtered_df.columns:
                    programme_summary['Documentation RAG (Mode)'] = programme_summary.index.map(filtered_df.groupby('Programme Name')['Documentation RAG'].agg(lambda x: x.mode()[0] if not x.mode().empty else 'Unknown').get)
                else: programme_summary['Documentation RAG (Mode)'] = 'Unknown'


            except KeyError as ke:
                 st.error(f"KeyError during KPI aggregation: Column '{ke}' not found. This might happen if KPI calculations failed earlier.")
                 programme_summary = pd.DataFrame() # Reset summary on error
            except Exception as e:
                 st.error(f"An error occurred during KPI aggregation: {e}")
                 st.code(traceback.format_exc())
                 programme_summary = pd.DataFrame() # Reset summary on error

        elif 'Programme Name' not in filtered_df.columns:
             st.warning("Programme Name column missing. Cannot display Programme level KPIs.")
        else:
             # Identify which required columns are missing
             missing_cols = [col for col in required_agg_cols if col not in filtered_df.columns]
             st.warning(f"Cannot display Programme KPIs due to missing columns: {', '.join(missing_cols)}. Check data upload and KPI calculation steps.")


        # --- Display Programme KPIs ---
        if not programme_summary.empty:
            st.subheader("Programme Level KPIs")
            num_programmes = len(programme_summary)
            # Adjust columns based on number of programmes, max 4
            num_cols = min(num_programmes, 4) if num_programmes > 0 else 1
            cols = st.columns(num_cols)

            prog_idx = 0
            for programme_name, kpi_data in programme_summary.iterrows():
                # Ensure kpi_data is a Series (it should be)
                if isinstance(kpi_data, pd.Series):
                    with cols[prog_idx % num_cols]:
                        st.markdown(f"##### {programme_name}")
                        # Use .get() with default values for safety
                        # KPI 1
                        st.markdown(create_kpi_card("Schedule Performance", kpi_data.get('Avg Schedule Deviation (Days)', np.nan), unit="days", decimals=0), unsafe_allow_html=True)
                        # KPI 2
                        st.markdown(create_kpi_card("Budget Perf. (Forecast)", kpi_data.get('Forecast Budget Variance (%)', np.nan), unit="%", decimals=1, rag_status=kpi_data.get('Forecast Budget RAG', 'Error')), unsafe_allow_html=True)
                        # KPI 3
                        st.markdown(create_kpi_card("Budget Perf. (Actual)", kpi_data.get('Actual Budget Variance (%)', np.nan), unit="%", decimals=1, rag_status=kpi_data.get('Actual Budget RAG', 'Error')), unsafe_allow_html=True)
                        # KPI 4 (Aggregated)
                        st.markdown(create_kpi_card("Avg. Solution Cost", kpi_data.get('Avg Solution Cost (%)', np.nan), unit="%", decimals=1, rag_status=kpi_data.get('Solution Cost RAG (Mode)', 'Error'), status_text=f"Mode: {kpi_data.get('Solution Cost RAG (Mode)', 'Error')}"), unsafe_allow_html=True)
                        # KPI 5 (Aggregated)
                        st.markdown(create_kpi_card("Avg. Doc Completion", kpi_data.get('Avg Documentation Completion (%)', np.nan), unit="%", decimals=1, rag_status=kpi_data.get('Documentation RAG (Mode)', 'Error'), status_text=f"Mode: {kpi_data.get('Documentation RAG (Mode)', 'Error')}"), unsafe_allow_html=True)
                        # KPI 6
                        st.markdown(create_kpi_card("Resourcing Allocation", kpi_data.get('Months Allocated', np.nan), unit=" Months", decimals=1, rag_status=kpi_data.get('Resourcing RAG', 'Error')), unsafe_allow_html=True) # Show 1 decimal for months
                        # KPI 7a
                        st.markdown(create_kpi_card("H&S File Signed", kpi_data.get('HS File Signed (%)', np.nan), unit="%", decimals=1, status_text=f"{kpi_data.get('HS File Signed (%)', 0):.1f}% Complete"), unsafe_allow_html=True)
                        # KPI 7b
                        st.markdown(create_kpi_card("CDM Checklist Complete", kpi_data.get('CDM Checklist Complete (%)', np.nan), unit="%", decimals=1, status_text=f"{kpi_data.get('CDM Checklist Complete (%)', 0):.1f}% Complete"), unsafe_allow_html=True)
                    prog_idx += 1
                else:
                     st.warning(f"Could not process KPI data for programme: {programme_name}")

            st.divider()


        # --- Display Site-Level Detail Table ---
        st.subheader("Site Level KPI Details")
        st.markdown("Detailed view of KPIs for each selected site.")

        # Select and order columns for the site detail table
        kpi_site_cols = [
            'Site', 'Programme Name', 'Catchment',
            'Schedule Deviation (Days)', # KPI 1
            'Solution Cost (%)', 'Solution Cost RAG', # KPI 4
            'Documentation Completion (%)', 'Documentation RAG', # KPI 5
            'HS File Signed', 'CDM Checklist Complete', # KPI 7
            # Add other relevant context columns if needed
            'Risk Score', 'SG1 Progress (%)'
        ]
        # Filter to only columns that actually exist in the dataframe
        kpi_site_cols_exist = [col for col in kpi_site_cols if col in filtered_df.columns]

        if kpi_site_cols_exist:
            # Sort by schedule deviation (most delayed first) if column exists
            sort_col = 'Schedule Deviation (Days)' if 'Schedule Deviation (Days)' in filtered_df.columns else 'Site'
            ascending_sort = False if sort_col == 'Schedule Deviation (Days)' else True

            # Create a copy for display to avoid modifying filtered_df
            df_display = filtered_df[kpi_site_cols_exist].copy()

            # Apply sorting
            if sort_col in df_display.columns:
                 # Ensure sort column is numeric before sorting
                 df_display[sort_col] = pd.to_numeric(df_display[sort_col], errors='coerce')
                 df_display = df_display.sort_values(by=sort_col, ascending=ascending_sort, na_position='last')

            # Apply styling
            st.dataframe(style_dataframe(df_display), hide_index=True, use_container_width=True)
        else:
            st.info("No site-level KPI columns available to display.")


# ==================================
# 🔍 Tab Discovery: Data Discovery (ENHANCED)
# ==================================
with tab_discovery:
    st.header("🔍 Data Discovery & Readiness Assessment")
    st.markdown("Evaluate the availability, quality, and governance of data required for KPI reporting.")
    st.markdown("---")

    # Retrieve stored inputs or initialize if not present
    discovery_data = st.session_state.setdefault('discovery_inputs', {})

    # --- Instructions ---
    st.info("Please answer the following questions based on the current understanding of data sources and processes supporting the SG1 KPIs.")

    # --- Section 1: Schedule Performance Data (KPI 1) ---
    with st.expander("🗓️ Schedule Performance Data (KPI 1)", expanded=True):
        st.markdown("<div class='discovery-section'>", unsafe_allow_html=True)
        st.markdown("<p class='persona-note'><i>Focus for: Programme Manager, Asset Manager</i></p>", unsafe_allow_html=True)

        discovery_data['kpi1_baseline_source'] = st.text_input(
            "1.1 What is the source system/document for Baseline SG1 Dates?",
            value=discovery_data.get('kpi1_baseline_source', ''), key='kpi1_bsrc'
        )
        discovery_data['kpi1_baseline_update_freq'] = st.selectbox(
            "1.2 How often is the Baseline SG1 Date formally updated/reviewed?",
            ["Per Gate", "Quarterly", "Annually", "Ad-hoc", "Never", "Unknown"],
            index=["Per Gate", "Quarterly", "Annually", "Ad-hoc", "Never", "Unknown"].index(discovery_data.get('kpi1_baseline_update_freq', 'Unknown')),
            key='kpi1_bupd'
        )
        discovery_data['kpi1_forecast_source'] = st.text_input(
            "1.3 What is the source system/process for Forecast SG1 Dates?",
            value=discovery_data.get('kpi1_forecast_source', ''), key='kpi1_fsrc'
        )
        discovery_data['kpi1_forecast_update_freq'] = st.selectbox(
            "1.4 How often are Forecast SG1 Dates updated?",
            ["Weekly", "Monthly", "Quarterly", "Per Gate", "Ad-hoc", "Unknown"],
            index=["Weekly", "Monthly", "Quarterly", "Per Gate", "Ad-hoc", "Unknown"].index(discovery_data.get('kpi1_forecast_update_freq', 'Unknown')),
            key='kpi1_fupd'
        )
        discovery_data['kpi1_date_quality'] = st.text_area(
            "1.5 Describe any known issues with the quality, consistency, or accessibility of baseline/forecast dates.",
            value=discovery_data.get('kpi1_date_quality', ''), key='kpi1_qual'
        )
        # Add more questions as needed by editing here
        # Example:
        # discovery_data['kpi1_change_control'] = st.text_input(
        #     "1.6 Is there a formal change control process for baseline dates?",
        #     value=discovery_data.get('kpi1_change_control', ''), key='kpi1_cc'
        # )
        st.markdown("</div>", unsafe_allow_html=True)


    # --- Section 2: Budget Performance Data (KPI 2 & 3) ---
    with st.expander("💰 Budget Performance Data (KPI 2 & 3)"):
        st.markdown("<div class='discovery-section'>", unsafe_allow_html=True)
        st.markdown("<p class='persona-note'><i>Focus for: Programme Manager</i></p>", unsafe_allow_html=True)

        discovery_data['kpi2_ibp_source'] = st.text_input(
            "2.1 What is the source system/process for Programme IBP Budgets?",
            value=discovery_data.get('kpi2_ibp_source', ''), key='kpi2_ibpsrc'
        )
        discovery_data['kpi2_ibp_level'] = st.radio(
            "2.2 Is the IBP Budget defined at the Programme or Project level for KPI calculation?",
            ["Programme", "Project", "Other", "Unknown"], horizontal=True,
             index=["Programme", "Project", "Other", "Unknown"].index(discovery_data.get('kpi2_ibp_level', 'Unknown')),
             key='kpi2_iblevel'
        )
        discovery_data['kpi2_forecast_outturn_source'] = st.text_input(
            "2.3 What is the source system/process for Project Forecast Outturns?",
             value=discovery_data.get('kpi2_forecast_outturn_source', ''), key='kpi2_fosrc'
        )
        discovery_data['kpi3_actual_outturn_source'] = st.text_input(
            "2.4 What is the source system/process for Project Actual Outturns?",
             value=discovery_data.get('kpi3_actual_outturn_source', ''), key='kpi3_aosrc'
        )
        discovery_data['kpi2_aggregation'] = st.text_area(
            "2.5 Describe the process for aggregating project forecasts/actuals to the programme level for KPI reporting.",
             value=discovery_data.get('kpi2_aggregation', ''), key='kpi2_agg'
        )
        discovery_data['kpi2_budget_quality'] = st.text_area(
            "2.6 Describe any known issues with budget/outturn data quality, timeliness, or consistency.",
             value=discovery_data.get('kpi2_budget_quality', ''), key='kpi2_bqual'
        )
        # Add more questions as needed by editing here
        st.markdown("</div>", unsafe_allow_html=True)


    # --- Section 3: Engineering Cost Data (KPI 4) ---
    with st.expander("🛠️ Engineering Cost Data (KPI 4)"):
        st.markdown("<div class='discovery-section'>", unsafe_allow_html=True)
        st.markdown("<p class='persona-note'><i>Focus for: Programme Manager, Asset Manager</i></p>", unsafe_allow_html=True)

        discovery_data['kpi4_eng_budget_source'] = st.text_input(
            "3.1 What is the source system/process for Engineering Budgets (used for KPI 4)?",
             value=discovery_data.get('kpi4_eng_budget_source', ''), key='kpi4_ebsrc'
        )
        discovery_data['kpi4_eng_cost_source'] = st.text_input(
            "3.2 What is the source system/process for tracking Engineering Design Costs?",
             value=discovery_data.get('kpi4_eng_cost_source', ''), key='kpi4_ecsrc'
        )
        discovery_data['kpi4_cost_definition'] = st.text_area(
            "3.3 How is 'Engineering Design Cost' defined? What cost types are included/excluded?",
             value=discovery_data.get('kpi4_cost_definition', ''), key='kpi4_cdef'
        )
        discovery_data['kpi4_cost_quality'] = st.text_area(
            "3.4 Describe any known issues with the accuracy or completeness of engineering cost/budget data.",
             value=discovery_data.get('kpi4_cost_quality', ''), key='kpi4_cqual'
        )
        # Add more questions as needed by editing here
        st.markdown("</div>", unsafe_allow_html=True)


    # --- Section 4: Documentation Completion Data (KPI 5) ---
    with st.expander("📄 Documentation Completion Data (KPI 5)"):
        st.markdown("<div class='discovery-section'>", unsafe_allow_html=True)
        st.markdown("<p class='persona-note'><i>Focus for: Compliance Officer, Programme Manager</i></p>", unsafe_allow_html=True)

        discovery_data['kpi5_bcde_usage'] = st.radio(
            "4.1 Is BCDE consistently used for submitting mandatory gate documentation?",
            ["Yes", "Mostly", "Partially", "No", "Unknown"], horizontal=True,
            index=["Yes", "Mostly", "Partially", "No", "Unknown"].index(discovery_data.get('kpi5_bcde_usage', 'Unknown')),
            key='kpi5_bcde'
        )
        discovery_data['kpi5_bcassure_usage'] = st.radio(
            "4.2 Is BC Assure used by Project Managers to select mandatory documents for each gate?",
            ["Yes", "Planned", "No", "Unknown"], horizontal=True,
            index=["Yes", "Planned", "No", "Unknown"].index(discovery_data.get('kpi5_bcassure_usage', 'Unknown')),
            key='kpi5_bcassure'
        )
        discovery_data['kpi5_completion_tracking'] = st.text_input(
            "4.3 How is the 'completion' status of submitted documents tracked/verified?",
             value=discovery_data.get('kpi5_completion_tracking', ''), key='kpi5_track'
        )
        discovery_data['kpi5_reporting_process'] = st.text_area(
            "4.4 Describe the process for generating the Documentation Completion (%) KPI.",
             value=discovery_data.get('kpi5_reporting_process', ''), key='kpi5_report'
        )
        # Add more questions as needed by editing here
        st.markdown("</div>", unsafe_allow_html=True)


    # --- Section 5: Resourcing Data (KPI 6) ---
    with st.expander("👥 Resourcing Data (KPI 6)"):
        st.markdown("<div class='discovery-section'>", unsafe_allow_html=True)
        st.markdown("<p class='persona-note'><i>Focus for: Programme Manager</i></p>", unsafe_allow_html=True)

        discovery_data['kpi6_resource_tool'] = st.text_input(
            "5.1 What tool/system is used for resource planning and allocation to projects?",
             value=discovery_data.get('kpi6_resource_tool', ''), key='kpi6_tool'
        )
        discovery_data['kpi6_allocation_level'] = st.radio(
            "5.2 Are resources allocated to specific named individuals or roles/teams?",
            ["Named Individuals", "Roles/Teams", "Both", "Unknown"], horizontal=True,
            index=["Named Individuals", "Roles/Teams", "Both", "Unknown"].index(discovery_data.get('kpi6_allocation_level', 'Unknown')),
            key='kpi6_level'
        )
        discovery_data['kpi6_forecast_horizon'] = st.text_input(
            "5.3 How far ahead is resource allocation typically planned/visible in the system (e.g., 3 months, 12 months)?",
             value=discovery_data.get('kpi6_forecast_horizon', ''), key='kpi6_horizon'
        )
        discovery_data['kpi6_data_extraction'] = st.text_area(
            "5.4 Describe how the 'Resource Allocation End Date' or equivalent data is extracted for KPI reporting.",
             value=discovery_data.get('kpi6_data_extraction', ''), key='kpi6_extract'
        )
        # Add more questions as needed by editing here
        st.markdown("</div>", unsafe_allow_html=True)


    # --- Section 6: Health & Safety Data (KPI 7) ---
    with st.expander("⚕️ Health & Safety Data (KPI 7)"):
        st.markdown("<div class='discovery-section'>", unsafe_allow_html=True)
        st.markdown("<p class='persona-note'><i>Focus for: Compliance Officer, Programme Manager</i></p>", unsafe_allow_html=True)

        discovery_data['kpi7_hsfile_process'] = st.text_input(
            "6.1 What is the process/system for tracking the creation and sign-off of the H&S file at SG1?",
             value=discovery_data.get('kpi7_hsfile_process', ''), key='kpi7_hsfile'
        )
        discovery_data['kpi7_cdm_process'] = st.text_input(
            "6.2 What is the process/system for tracking the completion of the CDM responsibility checklist at SG1?",
             value=discovery_data.get('kpi7_cdm_process', ''), key='kpi7_cdm'
        )
        discovery_data['kpi7_data_quality'] = st.text_area(
            "6.3 Describe any known challenges in consistently tracking or reporting these H&S metrics.",
            value=discovery_data.get('kpi7_data_quality', ''), key='kpi7_qual'
        )
        # Add more questions as needed by editing here
        st.markdown("</div>", unsafe_allow_html=True)

    # --- Section 7: General Data Governance & Quality ---
    with st.expander("🌐 General Data Governance & Quality"):
         st.markdown("<div class='discovery-section'>", unsafe_allow_html=True)
         st.markdown("<p class='persona-note'><i>Focus for: All Personas, especially Data Analyst, Compliance Officer</i></p>", unsafe_allow_html=True)
         discovery_data['gen_data_ownership'] = st.text_area(
             "7.1 Who are the designated owners for the key data elements used in these KPIs (e.g., budget data, schedule data, H&S data)?",
             value=discovery_data.get('gen_data_ownership',''), key='gen_owner'
         )
         discovery_data['gen_data_definitions'] = st.radio(
             "7.2 Are there clear, documented definitions for key terms (e.g., 'Baseline Date', 'Forecast Outturn', 'Actual Outturn', 'Engineering Cost')?",
             ["Yes", "Partially", "No", "Unknown"], horizontal=True,
             index=["Yes", "Partially", "No", "Unknown"].index(discovery_data.get('gen_data_definitions', 'Unknown')),
             key='gen_def'
         )
         discovery_data['gen_data_quality_process'] = st.text_area(
             "7.3 Describe any existing processes for monitoring or validating the quality of data used for KPI reporting.",
             value=discovery_data.get('gen_data_quality_process',''), key='gen_qualproc'
         )
         discovery_data['gen_integration_issues'] = st.text_area(
             "7.4 Are there known issues related to integrating data from different source systems for reporting?",
             value=discovery_data.get('gen_integration_issues',''), key='gen_integ'
         )
         # Add more general questions here
         st.markdown("</div>", unsafe_allow_html=True)


    # --- Section 8: Other Notes ---
    st.markdown("---")
    st.subheader("📝 Other Notes & Considerations")
    discovery_data['other_notes'] = st.text_area(
        "Use this space to add any other relevant information, questions, or potential data sources discovered.",
        height=150, value=discovery_data.get('other_notes', ''), key='other_notes'
    )

    # Save the updated discovery data back to session state
    st.session_state['discovery_inputs'] = discovery_data

    # Note: To add more questions permanently, edit the Python code in the sections above.


# =========================
# 📊 Tab Gap: Gap Analysis (KPI Focused)
# =========================
with tab_gap:
    st.header("📊 KPI Performance Gap Analysis")
    st.markdown("Analyze variations in KPI performance across sites.")

    if filtered_df is None or filtered_df.empty:
        st.warning("No data available for the selected filters.")
    else:
        # --- KPI RAG Status Heatmap ---
        st.subheader("🔥 KPI RAG Status Heatmap")
        st.markdown("Overview of RAG status for KPIs across selected sites.")

        rag_kpi_cols = ['Solution Cost RAG', 'Documentation RAG', 'Resourcing RAG', 'Forecast Budget RAG', 'Actual Budget RAG']
        rag_cols_exist = [col for col in rag_kpi_cols if col in filtered_df.columns]

        if rag_cols_exist and 'Site' in filtered_df.columns:
            # Map RAG status to numeric values for heatmap coloring
            rag_map = {"Green": 1, "Amber": 2, "Red": 3, "N/A": 0, "Unknown": 0, "Error": -1}
            heatmap_data = filtered_df[['Site'] + rag_cols_exist].set_index('Site')
            heatmap_numeric = heatmap_data.copy()
            for col in rag_cols_exist:
                 heatmap_numeric[col] = heatmap_data[col].map(rag_map).fillna(0) # Map and fill NaN with 0

            if not heatmap_numeric.empty:
                # Define custom colorscale: Gray (-1: Error), LightGray (0: N/A/Unknown), Green (1), Amber (2), Red (3)
                color_scale = [
                    [0.0, THAMES_COLORS['Gray']], # Error = -1 -> map to 0.0
                    [0.25, THAMES_COLORS['Gray']],
                    [0.25, '#f0f0f0'], # N/A/Unknown = 0 -> map to 0.25
                    [0.5, '#f0f0f0'],
                    [0.5, THAMES_COLORS['Green']], # Green = 1 -> map to 0.5
                    [0.75, THAMES_COLORS['Green']],
                    [0.75, THAMES_COLORS['Amber']], # Amber = 2 -> map to 0.75
                    [0.9, THAMES_COLORS['Amber']],
                    [0.9, THAMES_COLORS['Red']], # Red = 3 -> map to 1.0
                    [1.0, THAMES_COLORS['Red']]
                ]

                fig_heat_kpi = px.imshow(heatmap_numeric.T, # Use numeric version
                                        labels=dict(x="Site", y="KPI", color="RAG Status"),
                                        title="Site-Level KPI RAG Status",
                                        color_continuous_scale=color_scale,
                                        zmin=-1, zmax=3, # Set scale limits based on mapped values
                                        aspect="auto")

                # Customize color bar ticks and text
                fig_heat_kpi.update_layout(coloraxis_colorbar=dict(
                    title="RAG Status",
                    tickvals=[-1, 0, 1, 2, 3],
                    ticktext=["Error", "N/A", "Green", "Amber", "Red"]
                ))
                fig_heat_kpi.update_xaxes(side="bottom", tickangle=45)
                st.plotly_chart(fig_heat_kpi, use_container_width=True)
            else:
                st.info("No data available for the KPI RAG heatmap.")
        else:
             st.warning("Required RAG columns or 'Site' column not available for heatmap.")


        st.divider()

        # --- KPI Comparison Scatter Plot ---
        st.subheader("🆚 KPI Comparison Scatter Plot")
        st.markdown("Explore relationships between different KPI metrics.")

        # Select potential KPI metrics for axes (numeric ones)
        scatter_kpi_options = ['Schedule Deviation (Days)', 'Solution Cost (%)', 'Documentation Completion (%)', 'Months Allocated', 'Forecast Budget Variance (%)', 'Actual Budget Variance (%)', 'Risk Score']
        scatter_cols_exist = [col for col in scatter_kpi_options if col in filtered_df.columns and pd.api.types.is_numeric_dtype(filtered_df[col]) and filtered_df[col].notna().any()]

        if len(scatter_cols_exist) >= 2:
            col_x, col_y = st.columns(2)
            with col_x:
                x_axis_kpi = st.selectbox("Select X-axis KPI:", scatter_cols_exist, index=0, key='gap_scat_x')
            with col_y:
                y_axis_kpi = st.selectbox("Select Y-axis KPI:", scatter_cols_exist, index=1, key='gap_scat_y')

            # Select color/size options (can include non-KPIs like Risk Score)
            bubble_options = ['Risk Score', 'Carbon Impact (tCO2e)', 'SG1 Progress (%)'] + scatter_cols_exist
            bubble_cols_exist = [col for col in bubble_options if col in filtered_df.columns and pd.api.types.is_numeric_dtype(filtered_df[col]) and filtered_df[col].notna().any()]
            bubble_col = st.selectbox("Select Bubble Size Metric (Optional):", ["None"] + bubble_cols_exist, index=0, key='gap_scat_size')
            size_param = bubble_col if bubble_col != "None" else None

            color_options = ['Programme Name', 'Catchment', 'Forecast Budget RAG', 'Resourcing RAG', 'Documentation RAG', 'Solution Cost RAG'] + bubble_cols_exist
            color_cols_exist = [col for col in color_options if col in filtered_df.columns and filtered_df[col].notna().any()]
            color_col = st.selectbox("Select Color Metric (Optional):", ["None"] + color_cols_exist, index=0, key='gap_scat_color')
            color_param = color_col if color_col != "None" else None

            # Prepare data for plotting - drop NaNs for selected axes
            plot_df_kpi = filtered_df.dropna(subset=[x_axis_kpi, y_axis_kpi, 'Site'])

            if not plot_df_kpi.empty:
                 fig_scatter_kpi = px.scatter(
                     plot_df_kpi,
                     x=x_axis_kpi,
                     y=y_axis_kpi,
                     size=size_param,
                     color=color_param,
                     hover_name='Site',
                     hover_data=['Programme Name', 'Catchment'] + scatter_cols_exist, # Show all KPI options in hover
                     title=f"{y_axis_kpi} vs. {x_axis_kpi}",
                     color_discrete_map={ # Add RAG colors if a RAG column is selected
                            "Green": THAMES_COLORS['Green'], "Amber": THAMES_COLORS['Amber'],
                            "Red": THAMES_COLORS['Red'], "N/A": THAMES_COLORS['Gray'], "Unknown": THAMES_COLORS['Gray'], "Error": '#FF00FF' # Magenta for Error
                        }
                 )
                 # Add reference lines if useful (e.g., zero deviation)
                 if x_axis_kpi == 'Schedule Deviation (Days)': fig_scatter_kpi.add_vline(x=0, line_dash="dot", line_color=THAMES_COLORS['Gray'])
                 if y_axis_kpi == 'Schedule Deviation (Days)': fig_scatter_kpi.add_hline(y=0, line_dash="dot", line_color=THAMES_COLORS['Gray'])
                 if 'Variance' in x_axis_kpi: fig_scatter_kpi.add_vline(x=0, line_dash="dot", line_color=THAMES_COLORS['Gray'])
                 if 'Variance' in y_axis_kpi: fig_scatter_kpi.add_hline(y=0, line_dash="dot", line_color=THAMES_COLORS['Gray'])

                 st.plotly_chart(fig_scatter_kpi, use_container_width=True)
            else:
                 st.info(f"No data available to plot for the selected axes ({x_axis_kpi} vs {y_axis_kpi}) after removing missing values.")

        else:
            st.warning("Not enough suitable numeric KPI columns found for scatter plot comparison.")


# ==================================
# 📈 Tab Prog: Programme Reporting (KPI Focused)
# ==================================
with tab_prog:
    st.header("📈 Programme Reporting")
    st.markdown("Track delivery progress, milestones, and schedule performance.")

    if filtered_df is None or filtered_df.empty:
        st.warning("No data available for the selected filters.")
    else:
        # Check for required columns for this tab
        required_prog_cols = ['Site', 'Baseline SG1 Date', 'Forecast SG1 Date', 'Schedule Deviation (Days)']
        missing_or_bad_cols = [col for col in required_prog_cols if col not in filtered_df.columns or filtered_df[col].isna().all()]
        if missing_or_bad_cols:
             st.error(f"Programme Reporting requires data in columns like: {', '.join(required_prog_cols)}. Missing or all-null: {', '.join(missing_or_bad_cols)}. Please check data.")
        else:
            # --- Gantt-style Milestone Timeline (Focus on KPI 1) ---
            st.subheader("🗓️ Milestone Timeline (KPI 1)")
            st.markdown("Visualizing Baseline vs. Forecast Stage Gate 1 Dates.")
            gantt_tasks = []
            today_dt = pd.to_datetime(date.today()) # Use datetime for comparison

            try:
                # Prepare data for Gantt: Need Task (Site), Start (Baseline), Finish (Forecast), Resource (Status)
                gantt_df = filtered_df[['Site', 'Baseline SG1 Date', 'Forecast SG1 Date']].copy()
                # Ensure dates are datetime objects
                gantt_df['Baseline SG1 Date'] = pd.to_datetime(gantt_df['Baseline SG1 Date'], errors='coerce')
                gantt_df['Forecast SG1 Date'] = pd.to_datetime(gantt_df['Forecast SG1 Date'], errors='coerce')
                gantt_df.dropna(subset=['Baseline SG1 Date', 'Forecast SG1 Date', 'Site'], inplace=True) # Need both dates and site name

                if not gantt_df.empty:
                    # Determine Status based on forecast date vs today and baseline
                    def get_gantt_status(row):
                        if not isinstance(row['Forecast SG1 Date'], pd.Timestamp) or not isinstance(row['Baseline SG1 Date'], pd.Timestamp): return "Date Missing"
                        if row['Forecast SG1 Date'] <= today_dt and row['Forecast SG1 Date'] <= row['Baseline SG1 Date']: return "Complete/On Time"
                        elif row['Forecast SG1 Date'] <= today_dt and row['Forecast SG1 Date'] > row['Baseline SG1 Date']: return "Complete/Delayed"
                        elif row['Forecast SG1 Date'] > today_dt and row['Forecast SG1 Date'] <= row['Baseline SG1 Date']: return "Forecast On Time"
                        elif row['Forecast SG1 Date'] > today_dt and row['Forecast SG1 Date'] > row['Baseline SG1 Date']: return "Forecast Delayed"
                        else: return "Unknown"

                    gantt_df['Status'] = gantt_df.apply(get_gantt_status, axis=1)

                    # Create task list for figure_factory
                    for index, row in gantt_df.iterrows():
                         start_date = min(row['Baseline SG1 Date'], row['Forecast SG1 Date'])
                         finish_date = max(row['Baseline SG1 Date'], row['Forecast SG1 Date'])
                         if start_date == finish_date: finish_date += timedelta(days=1)
                         gantt_tasks.append(dict(Task=row['Site'], Start=start_date, Finish=finish_date, Resource=row['Status']))

                    if gantt_tasks:
                        colors = { "Complete/On Time": THAMES_COLORS['Green'], "Complete/Delayed": THAMES_COLORS['Amber'],
                                   "Forecast On Time": THAMES_COLORS['Secondary Blue'], "Forecast Delayed": THAMES_COLORS['Red'],
                                   "Date Missing": THAMES_COLORS['Gray'], "Unknown": THAMES_COLORS['Gray']}
                        all_start_dates = [t['Start'] for t in gantt_tasks if isinstance(t['Start'], pd.Timestamp)]
                        all_finish_dates = [t['Finish'] for t in gantt_tasks if isinstance(t['Finish'], pd.Timestamp)]
                        valid_dates = all_start_dates + all_finish_dates
                        x_range = None
                        if valid_dates:
                             min_date, max_date = min(valid_dates), max(valid_dates)
                             x_range = [min_date - timedelta(days=30), max_date + timedelta(days=30)] # Add buffer

                        fig_gantt = ff.create_gantt(gantt_tasks, index_col='Resource', colors=colors, show_colorbar=True,
                                                    group_tasks=True, title="Site SG1 Baseline vs. Forecast Dates (Color = Status)",
                                                    showgrid_x=True, showgrid_y=True)
                        fig_gantt.update_layout(xaxis_title="Date", yaxis_title="Site", legend_title="Status", xaxis_range=x_range)
                        st.plotly_chart(fig_gantt, use_container_width=True)
                    else:
                        st.info("No valid tasks generated for Gantt chart (check Baseline/Forecast SG1 Dates).")
                else:
                     st.info("No sites with valid Baseline and Forecast SG1 Dates found.")
            except Exception as e:
                st.error(f"Could not generate Gantt chart. Check date formats and data: {e}")
                st.code(traceback.format_exc())

            st.divider()

            # --- Schedule Deviation Distribution (Focus on KPI 1) ---
            st.subheader("⏳ Schedule Deviation Distribution (KPI 1)")
            if 'Schedule Deviation (Days)' in filtered_df.columns and filtered_df['Schedule Deviation (Days)'].notna().any():
                 plot_data_hist = filtered_df.dropna(subset=['Schedule Deviation (Days)'])
                 if not plot_data_hist.empty:
                     fig_hist_dev = px.histogram(plot_data_hist,
                                                x='Schedule Deviation (Days)',
                                                title="Distribution of Schedule Deviation (Days)",
                                                labels={'Schedule Deviation (Days)': 'Days (Negative = Ahead, Positive = Behind)'},
                                                marginal="box") # Add box plot
                     fig_hist_dev.add_vline(x=0, line_dash="dash", line_color="black", annotation_text="On Time")
                     st.plotly_chart(fig_hist_dev, use_container_width=True)
                 else:
                      st.info("No valid 'Schedule Deviation (Days)' data points after dropping NaNs.")
            else:
                 st.info("No valid 'Schedule Deviation (Days)' data to display distribution.")

            # --- Budget Performance Summary (KPI 2 / 3) ---
            st.divider()
            st.subheader("💰 Programme Budget Performance Summary (KPI 2 & 3)")
            prog_budget_cols = ['Programme Name', 'Forecast Budget Variance (%)', 'Forecast Budget RAG', 'Actual Budget Variance (%)', 'Actual Budget RAG']
            prog_budget_cols_exist = [col for col in prog_budget_cols if col in filtered_df.columns]

            if 'Programme Name' in prog_budget_cols_exist and len(prog_budget_cols_exist) > 1:
                 prog_budget_summary = filtered_df[prog_budget_cols_exist].drop_duplicates(subset=['Programme Name'])
                 st.dataframe(style_dataframe(prog_budget_summary.set_index('Programme Name')), use_container_width=True)
            else:
                 st.info("Programme budget performance data not available for summary.")


# =================================
# 📉 Tab Adv: Advanced Analytics (KPI Focused)
# =================================
with tab_adv:
    st.header("📉 Advanced Analytics")
    st.markdown("Explore site performance using ranking, correlation, clustering, and forecasting based on KPIs.")

    if filtered_df is None or filtered_df.empty:
        st.warning("No data available for the selected filters.")
    else:
        try: # Wrap analytics in try-except for robustness
            analytics_df = filtered_df.copy() # Use filtered data

            # Define potential KPI-related numeric columns for analysis
            kpi_numeric_cols = [
                'Schedule Deviation (Days)', 'Solution Cost (%)', 'Documentation Completion (%)',
                'Months Allocated', 'Forecast Budget Variance (%)', 'Actual Budget Variance (%)'
            ]
            # Include other relevant metrics like Risk Score
            other_numeric_cols = ['Risk Score', 'SG1 Progress (%)', 'Carbon Impact (tCO2e)', 'Innovation Potential (%)']
            # Get columns that actually exist and have numeric data with variance
            available_numeric_cols = []
            for col in kpi_numeric_cols + other_numeric_cols:
                 if col in analytics_df.columns:
                     numeric_col = pd.to_numeric(analytics_df[col], errors='coerce').dropna()
                     if not numeric_col.empty and numeric_col.nunique() > 1:
                          available_numeric_cols.append(col)

            # --- Site Ranking Engine (KPI Focused Defaults) ---
            st.subheader("🏆 Site Ranking Engine")
            st.markdown("Rank sites based on weighted criteria. Adjust weights to explore priorities.")

            # Default to KPI columns if available
            default_rank_cols = [col for col in ['Schedule Deviation (Days)', 'Forecast Budget Variance (%)', 'Solution Cost (%)'] if col in available_numeric_cols]
            if len(default_rank_cols) < 2: # Fallback if defaults aren't available
                 default_rank_cols = available_numeric_cols[:3] # Take first few available

            rank_input_cols_exist = available_numeric_cols # Allow selection from all available

            if len(rank_input_cols_exist) < 2: # Need at least 2 for meaningful ranking
                 st.warning(f"Ranking requires at least two numeric columns with data. Found: {', '.join(rank_input_cols_exist)}. Please check data.")
            else:
                weights = {}
                cols_rank_w = st.columns(len(rank_input_cols_exist))
                for i, col in enumerate(rank_input_cols_exist):
                     # Determine default weight - focus on selected defaults if possible
                     default_weight = (1.0 / len(default_rank_cols)) if col in default_rank_cols and len(default_rank_cols)>0 else 0.1
                     # Determine help text and direction
                     lower_is_better = any(term in col for term in ['Cost', 'Carbon', 'Risk', 'Deviation', 'Variance'])
                     help_text = "Higher weight = Lower value prioritized." if lower_is_better else "Higher weight = Higher value prioritized."
                     label = col.replace(' (%)', '').replace(' (£K)', '').replace('(tCO2e)', '').replace('(Days)', '').replace('Project Forecast Outturn', 'Cost')
                     with cols_rank_w[i]:
                         weights[col] = st.slider(f"⚖️ Weight: {label}", 0.0, 1.0, default_weight, 0.05, key=f'w_{col}_adv', help=help_text)

                # Normalize weights
                total_weight = sum(weights.values())
                if total_weight == 0: total_weight = 1 # Avoid division by zero
                normalized_weights = {col: w / total_weight for col, w in weights.items()}
                st.caption(f"Normalized Weights Used: {', '.join([f'{col.split()[0]}={w:.2f}' for col, w in normalized_weights.items()])}")

                # Calculate Rank Score
                rank_scores = pd.Series(0.0, index=analytics_df.index)
                rank_data_clean = analytics_df[rank_input_cols_exist].apply(pd.to_numeric, errors='coerce') # Ensure numeric

                for col, weight in normalized_weights.items():
                     # Normalize column (0-1 scale) after handling NaNs
                     valid_col_data = rank_data_clean[col].dropna()
                     if not valid_col_data.empty:
                         min_val = valid_col_data.min()
                         max_val = valid_col_data.max()
                         range_val = max_val - min_val
                         if range_val == 0: norm_col = pd.Series(0.5, index=rank_data_clean.index)
                         else: norm_col = (rank_data_clean[col] - min_val) / range_val
                     else: norm_col = pd.Series(0.5, index=rank_data_clean.index)
                     # Invert score if lower is better
                     if any(term in col for term in ['Cost', 'Carbon', 'Risk', 'Deviation', 'Variance']): norm_col = 1 - norm_col
                     # Add weighted score, filling NaNs in the original column with neutral 0.5 *before* weighting
                     rank_scores += weight * norm_col.fillna(0.5)

                analytics_df['Rank Score'] = rank_scores * 100

                # Display Ranked Sites
                ranked_sites = analytics_df.sort_values('Rank Score', ascending=False)
                fig_rank = px.bar(ranked_sites, x='Site', y='Rank Score',
                                  title=f"Site Ranking based on Current Weights",
                                  color='Rank Score', color_continuous_scale='Blues',
                                  hover_data=['Catchment'] + rank_input_cols_exist)
                fig_rank.update_layout(xaxis={'categoryorder':'total descending'}, yaxis_title="Weighted Rank Score (0-100)", coloraxis_showscale=False)
                st.plotly_chart(fig_rank, use_container_width=True)

                with st.expander("View Ranked Data Table"):
                    display_cols_rank = ['Site', 'Catchment', 'Rank Score'] + rank_input_cols_exist
                    st.dataframe(style_dataframe(ranked_sites[display_cols_rank]), hide_index=True, use_container_width=True)

            st.divider()

            # --- Correlation Analysis (KPI Focused Defaults) ---
            st.subheader("🔗 Correlation Analysis")
            st.markdown("Explore linear relationships between key numeric variables.")
            corr_cols_options = available_numeric_cols # Use all available numeric cols with variance
            default_corr_cols = [col for col in kpi_numeric_cols + ['Risk Score'] if col in available_numeric_cols] # Default to KPIs + Risk

            selected_corr_cols = st.multiselect("Select variables for correlation matrix:", corr_cols_options, default=default_corr_cols, key='corr_cols_adv')

            if len(selected_corr_cols) > 1:
                numeric_df = analytics_df[selected_corr_cols].apply(pd.to_numeric, errors='coerce').dropna() # Ensure numeric and drop rows with any NaNs
                if len(numeric_df) > 1: # Need at least 2 rows for correlation
                    corr_matrix = numeric_df.corr()
                    fig_corr = px.imshow(corr_matrix, text_auto=".2f", aspect="auto",
                                        color_continuous_scale='RdBu_r', color_continuous_midpoint=0,
                                        title="Correlation Matrix of Selected Metrics")
                    fig_corr.update_layout(height=max(400, 50 * len(selected_corr_cols)), coloraxis_colorbar_x=1.02) # Adjust height
                    st.plotly_chart(fig_corr, use_container_width=True)
                    st.caption("Values close to 1 (blue) = strong positive linear correlation, close to -1 (red) = strong negative linear correlation, close to 0 = weak linear correlation.")
                else:
                     st.info("Not enough data rows remaining after dropping missing values in selected columns to calculate correlations.")
            else:
                st.info("Please select at least two variables for correlation analysis.")

            st.divider()

            # --- Cluster Analysis (KPI Focused Defaults) ---
            st.subheader("🧩 Site Segmentation (K-Means Clustering)")
            st.markdown("Group similar sites based on selected performance metrics.")
            cluster_cols_options = available_numeric_cols # Use all available numeric cols with variance
            default_cluster_cols = [col for col in kpi_numeric_cols if col in available_numeric_cols] # Default to KPIs
            if not default_cluster_cols: default_cluster_cols = available_numeric_cols[:2] # Fallback

            if not cluster_cols_options:
                st.warning("No suitable numeric columns available for clustering in the filtered data.")
            else:
                cluster_cols = st.multiselect("Select metrics for clustering:", cluster_cols_options, default=default_cluster_cols, key='clus_cols')

                if len(cluster_cols) >= 2:
                    # Prepare data for clustering (ensure numeric, drop NaNs)
                    cluster_data_pre = analytics_df[cluster_cols].apply(pd.to_numeric, errors='coerce').dropna()

                    if len(cluster_data_pre) < 2: # Need at least 2 points to cluster
                         st.warning(f"Not enough data points ({len(cluster_data_pre)}) for clustering after dropping missing values in selected columns.")
                    else:
                         max_k = min(8, len(cluster_data_pre)-1 if len(cluster_data_pre)>1 else 2)
                         if max_k < 2:
                            st.warning(f"Not enough data points ({len(cluster_data_pre)}) to form at least 2 clusters.")
                         else:
                            n_clusters = st.slider("Select number of clusters (K):", 2, max_k, min(3, max_k), key='k_clusters')

                            if len(cluster_data_pre) >= n_clusters:
                                scaler = StandardScaler()
                                scaled_data = scaler.fit_transform(cluster_data_pre)
                                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                                cluster_labels = kmeans.fit_predict(scaled_data)
                                cluster_col_name = 'Cluster'
                                # Assign cluster labels back using the index of cluster_data_pre
                                analytics_df.loc[cluster_data_pre.index, cluster_col_name] = [f"Cluster {i+1}" for i in cluster_labels]
                                # Convert to category type where assigned
                                if cluster_col_name in analytics_df.columns:
                                     analytics_df[cluster_col_name] = analytics_df[cluster_col_name].astype('category')

                                st.write(f"**{n_clusters} Clusters Identified:**")
                                # Select default axes for visualization
                                vis_x_default = cluster_cols[0]
                                vis_y_default = cluster_cols[1] if len(cluster_cols)>1 else cluster_cols[0]
                                vis_x = st.selectbox("X-axis for Cluster Plot:", cluster_cols, index=cluster_cols.index(vis_x_default))
                                vis_y = st.selectbox("Y-axis for Cluster Plot:", cluster_cols, index=cluster_cols.index(vis_y_default))
                                size_col = 'Risk Score' if 'Risk Score' in analytics_df.columns else None # Size by Risk maybe?

                                # Plot only rows where cluster was assigned
                                plot_cluster_df = analytics_df.dropna(subset=[cluster_col_name, vis_x, vis_y])
                                if not plot_cluster_df.empty:
                                    fig_cluster = px.scatter(plot_cluster_df,
                                                            x=vis_x, y=vis_y, color=cluster_col_name,
                                                            size=size_col if size_col in plot_cluster_df.columns else None,
                                                            hover_name='Site', hover_data=cluster_cols + ['Catchment'],
                                                            title=f"Site Clusters based on Selected Metrics (k={n_clusters})",
                                                            category_orders={cluster_col_name: sorted(plot_cluster_df[cluster_col_name].dropna().unique())})
                                    st.plotly_chart(fig_cluster, use_container_width=True)

                                    with st.expander("View Average Metrics per Cluster"):
                                        # Calculate summary on the data actually used for clustering
                                        cluster_summary_data = analytics_df.loc[cluster_data_pre.index].copy() # Get original data for clustered points
                                        cluster_summary_data[cluster_col_name] = analytics_df.loc[cluster_data_pre.index, cluster_col_name] # Add labels
                                        cluster_summary = cluster_summary_data.groupby(cluster_col_name)[cluster_cols].mean()
                                        st.dataframe(style_dataframe(cluster_summary), use_container_width=True)
                                else:
                                     st.info("No data points available for the selected cluster plot axes after removing missing values.")

                            else:
                                st.warning(f"Not enough data points ({len(cluster_data_pre)}) to form {n_clusters} clusters.")
                else:
                    st.info("Please select at least two metrics for clustering.")

            st.divider()

            # --- Forecasting (Keep original logic using SG1 Progress as example) ---
            st.subheader("🔮 SG1 Progress Forecast (Statistical)")
            st.markdown("ARIMA forecast based on the mocked historical trend of average SG1 Progress. (Can be adapted for other time-series KPIs).")
            try:
                if 'SG1 Progress (%)' in filtered_df.columns:
                    # Calculate mean only on valid numeric data
                    sg1_numeric = pd.to_numeric(filtered_df['SG1 Progress (%)'], errors='coerce').dropna()
                    if not sg1_numeric.empty:
                        current_avg_sg1_fc = sg1_numeric.mean()

                        # Generate consistent mock history using a fixed random seed locally
                        rng = np.random.RandomState(42) # Consistent seed
                        history_vals = [ max(0, current_avg_sg1_fc - rng.uniform(8,12)), max(0, current_avg_sg1_fc - rng.uniform(4,7)),
                                         max(0, current_avg_sg1_fc - rng.uniform(1,3)), current_avg_sg1_fc ]
                        end_date = pd.to_datetime(f'{datetime.now().year}-{datetime.now().month}-01') # Use current month start
                        history_dates = pd.date_range(end=end_date, periods=len(history_vals), freq='MS')
                        ts_data = pd.Series(history_vals, index=history_dates).dropna()

                        if len(ts_data) >= 4:
                            # Fit ARIMA model
                            model = ARIMA(ts_data, order=(1, 1, 0)) # Simple ARIMA order
                            model_fit = model.fit()
                            forecast_periods = 4
                            forecast_result = model_fit.get_forecast(steps=forecast_periods)
                            forecast_df_arima = forecast_result.summary_frame(alpha=0.10) # 90% CI
                            forecast_df_arima.index.name = 'Month'
                            forecast_df_arima.reset_index(inplace=True)
                            forecast_df_arima['MonthStr'] = forecast_df_arima['Month'].dt.strftime('%b %Y') # Format month string

                            # Prepare history data for plotting
                            history_df = ts_data.reset_index()
                            history_df.columns = ['Month', 'Actual Avg SG1 (%)']
                            history_df['MonthStr'] = history_df['Month'].dt.strftime('%b %Y')

                            # Create Plotly figure
                            fig_forecast_stat = go.Figure()
                            fig_forecast_stat.add_trace(go.Scatter(x=history_df['MonthStr'], y=history_df['Actual Avg SG1 (%)'], mode='lines+markers', name='History', line=dict(color=THAMES_COLORS['Gray'])))
                            fig_forecast_stat.add_trace(go.Scatter(x=forecast_df_arima['MonthStr'], y=forecast_df_arima['mean'], mode='lines+markers', name='Forecast', line=dict(color=THAMES_COLORS['Secondary Blue'], width=3)))
                            fig_forecast_stat.add_trace(go.Scatter(x=forecast_df_arima['MonthStr'], y=forecast_df_arima['mean_ci_upper'], mode='lines', line=dict(width=0), showlegend=False))
                            fig_forecast_stat.add_trace(go.Scatter(x=forecast_df_arima['MonthStr'], y=forecast_df_arima['mean_ci_lower'], mode='lines', line=dict(width=0), fillcolor='rgba(0, 161, 214, 0.2)', fill='tonexty', name='90% CI', showlegend=True))
                            max_y_val = max(100, forecast_df_arima['mean_ci_upper'].max()*1.1 if not forecast_df_arima['mean_ci_upper'].empty else 100)
                            fig_forecast_stat.update_layout(title="Statistical Forecast for Average SG1 Progress (ARIMA)", xaxis_title="Month", yaxis_title="Average SG1 Progress (%)", yaxis_range=[0, max_y_val], hovermode="x unified")
                            st.plotly_chart(fig_forecast_stat, use_container_width=True)
                        else:
                            st.info("Not enough historical data points generated/available to run forecast.")
                    else:
                        st.info("Cannot calculate average SG1 progress for forecasting (no valid data).")
                else:
                    st.warning("'SG1 Progress (%)' column not found for forecasting.")
            except ImportError:
                st.error("Statsmodels library not installed. Please run `pip install statsmodels`")
            except Exception as e:
                st.error(f"Could not generate forecast: {e}.")
                st.code(traceback.format_exc())

            # --- Driver Analysis (Simple Regression - KPI Focused Defaults) ---
            st.divider()
            st.subheader("📊 Driver Analysis (Exploratory Regression)")
            st.markdown("Explore potential drivers of KPI outcomes using linear regression. *Note: Shows association, not causation.*")
            # Allow selecting a KPI as the dependent variable
            dep_var_options = [col for col in kpi_numeric_cols + ['Risk Score'] if col in available_numeric_cols]
            if not dep_var_options:
                st.warning("No suitable numeric columns available to select as dependent variable for regression.")
            else:
                reg_dep_var = st.selectbox("Select Dependent Variable (Outcome):", dep_var_options, index=dep_var_options.index('Risk Score') if 'Risk Score' in dep_var_options else 0, key='reg_dep_var_adv')

                reg_indep_vars_options = [col for col in available_numeric_cols if col != reg_dep_var] # Exclude dependent var

                if reg_indep_vars_options:
                    default_indep_vars = [col for col in ['Data Completeness (%)', 'SG1 Progress (%)', 'Schedule Deviation (Days)'] if col in reg_indep_vars_options]
                    selected_indep_vars = st.multiselect("Select Independent Variables (Potential Drivers):", reg_indep_vars_options, default=default_indep_vars, key='reg_vars')

                    if selected_indep_vars:
                        reg_cols = [reg_dep_var] + selected_indep_vars
                        # Ensure all selected columns are numeric before dropping NaNs
                        reg_data_numeric = analytics_df[reg_cols].apply(pd.to_numeric, errors='coerce')
                        reg_data = reg_data_numeric.dropna()

                        if len(reg_data) > len(reg_cols): # Need more rows than columns
                            reg_data_clean = reg_data.copy()
                            # Clean column names for statsmodels formula more robustly
                            clean_cols = {col: f'var_{i}' for i, col in enumerate(reg_data_clean.columns)} # Assign generic names
                            orig_cols_map = {f'var_{i}': col for i, col in enumerate(reg_data_clean.columns)} # Map back to original
                            reg_data_clean = reg_data_clean.rename(columns=clean_cols)
                            dependent_var_clean = [k for k, v in orig_cols_map.items() if v == reg_dep_var][0]
                            independent_vars_clean = [k for k, v in orig_cols_map.items() if v in selected_indep_vars]
                            # FIX: Remove backticks from formula
                            formula = f"{dependent_var_clean} ~ {' + '.join(independent_vars_clean)}"

                            try:
                                model_ols = smf.ols(formula=formula, data=reg_data_clean).fit()
                                st.write(f"**Regression Summary (Dependent Variable: {reg_dep_var})**")
                                # Display summary table with original names
                                summary_df = pd.read_html(model_ols.summary().tables[1].as_html(), header=0, index_col=0)[0]
                                summary_df.index = summary_df.index.map(orig_cols_map).fillna(summary_df.index) # Map index back
                                summary_df.index = summary_df.index.str.replace('`', '') # Clean backticks if any remain
                                st.dataframe(summary_df) # Display as dataframe

                                st.caption(f"R-squared: {model_ols.rsquared:.3f}. Adj. R-squared: {model_ols.rsquared_adj:.3f}")
                                st.caption("Interpret with caution. P>|t| < 0.05 suggests potential statistical significance.")
                            except Exception as e:
                                st.error(f"Could not perform regression analysis: {e}")
                                st.code(traceback.format_exc())
                        else:
                            st.info("Not enough data points remaining after removing missing values for the selected variables.")
                    else:
                        st.info("Please select at least one independent variable for regression.")
                else:
                    st.warning(f"No suitable independent variables found to predict {reg_dep_var}.")


        # --- Final Error Handling for the Tab ---
        except ImportError as ie:
            st.error(f"ImportError: Required library missing for Advanced Analytics. Please install necessary packages: {ie}")
            st.code("pip install statsmodels scikit-learn") # Remind user
        except Exception as e:
            st.error(f"An error occurred in the Advanced Analytics tab: {e}")
            st.error("Traceback:")
            st.code(traceback.format_exc())

# End of Streamlit App Code
