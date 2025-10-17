import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import glob
import os
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ================= CONFIG =================
DATA_FOLDER = "processed_apartments2"

# ================= ENERGY COLUMN HELPER =================
def get_energy_columns(df: pd.DataFrame) -> list:
    """
    Heuristic: return columns that look like energy/power series and are not meta fields.
    Excludes any column containing these keys in its (lowercased) name.
    """
    exclude_keys = ['timestamp', 'apartment', 'phase', 'date', 'time']
    return [c for c in df.columns if all(k not in c.lower() for k in exclude_keys)]

# ================= LOAD DATA (with MOD/absolute normalization) =================
@st.cache_data
def load_all_data():
    """
    Loads all CSVs from DATA_FOLDER, parses Timestamp, coerces energy columns to numeric,
    and applies absolute value so any negative readings become positive.
    """
    all_files = glob.glob(os.path.join(DATA_FOLDER, "*.csv"))
    dfs = {}

    for f in all_files:
        try:
            df = pd.read_csv(f)

            # Name apartment from filename
            apt_name = os.path.splitext(os.path.basename(f))[0]
            df["Apartment"] = apt_name

            # Parse timestamp if present
            if "Timestamp" in df.columns:
                df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

            # Identify likely energy columns
            energy_cols = get_energy_columns(df)

            if energy_cols:
                # Coerce to numeric (handle strings like "1,234.5" too)
                df[energy_cols] = (
                    df[energy_cols]
                    .apply(lambda s: pd.to_numeric(s.astype(str).str.replace(",", ""), errors="coerce"))
                    .abs()  # MOD → make all values non-negative
                )
                # Optional: fill NaNs if desired
                # df[energy_cols] = df[energy_cols].fillna(0)

            dfs[apt_name] = df

        except Exception as e:
            st.warning(f"Could not load {f}: {str(e)}")

    return dfs

# ================= IMPROVED UTILITY FUNCTIONS =================
def calculate_operation_duration_improved(df, appliance_col, threshold=0.01):
    """Improved function to calculate operation duration for appliances"""
    if appliance_col not in df.columns:
        return pd.DataFrame()
    
    # Create a copy and sort by timestamp
    data = df[['Timestamp', appliance_col]].copy()
    data = data.sort_values('Timestamp').dropna()
    
    if data.empty:
        return pd.DataFrame()
    
    # Reset index for proper iteration
    data = data.reset_index(drop=True)
    
    # Detect ON periods (values above threshold)
    data['is_on'] = data[appliance_col] > threshold
    data['state_change'] = data['is_on'].ne(data['is_on'].shift())
    data['group_id'] = data['state_change'].cumsum()
    
    # Calculate duration for each ON period
    events = []
    for group_id, group in data.groupby('group_id'):
        if group['is_on'].iloc[0]:  # This is an ON period
            start_time = group['Timestamp'].iloc[0]
            end_time = group['Timestamp'].iloc[-1]
            duration_hours = (end_time - start_time).total_seconds() / 3600
            
            # Only include events with reasonable duration (more than 5 minutes)
            if duration_hours >= 0.083:  # 5 minutes in hours
                events.append({
                    'appliance': appliance_col,
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration_hours': duration_hours,
                    'avg_power': group[appliance_col].mean(),
                    'max_power': group[appliance_col].max(),
                    'day_of_week': start_time.strftime('%A')
                })
    
    return pd.DataFrame(events)

def detect_on_off_events(df, appliance_col, threshold=0.01):
    """Detect ON/OFF events for an appliance"""
    if appliance_col not in df.columns:
        return pd.DataFrame()
    
    # Create a copy and sort by timestamp
    data = df[['Timestamp', appliance_col]].copy().sort_values('Timestamp')
    data = data.dropna()
    
    if data.empty:
        return pd.DataFrame()
    
    # Calculate difference to detect state changes
    data['power_diff'] = data[appliance_col].diff()
    data['state'] = data[appliance_col] > threshold
    data['state_change'] = data['state'].ne(data['state'].shift())
    data['event_id'] = data['state_change'].cumsum()
    
    # Filter only ON events (transition from OFF to ON)
    on_events = data[data['state_change'] & data['state']].copy()
    
    return on_events

def detect_anomalies_operation(df, appliance_col, expected_hours=None):
    """Detect anomalies based on operation schedules"""
    anomalies = []
    
    if appliance_col not in df.columns:
        return anomalies
    
    # Get operation durations
    operation_df = calculate_operation_duration_improved(df, appliance_col)
    
    if operation_df.empty:
        return anomalies
    
    # Define expected operation hours based on appliance type
    if expected_hours is None:
        # Default expected hours based on appliance name
        appliance_lower = appliance_col.lower()
        if any(word in appliance_lower for word in ['ac', 'air conditioner']):
            expected_hours = {'min': 1, 'max': 8}  # AC typically runs 1-8 hours
        elif any(word in appliance_lower for word in ['geyser', 'water heater']):
            expected_hours = {'min': 0.5, 'max': 3}  # Geyser typically 0.5-3 hours
        elif any(word in appliance_lower for word in ['washing', 'laundry']):
            expected_hours = {'min': 0.5, 'max': 2}  # Washing machine 0.5-2 hours
        elif any(word in appliance_lower for word in ['oven', 'cook']):
            expected_hours = {'min': 0.25, 'max': 4}  # Oven 0.25-4 hours
        else:
            expected_hours = {'min': 0.1, 'max': 24}  # Default
    
    # Detect duration anomalies
    long_operations = operation_df[operation_df['duration_hours'] > expected_hours['max']]
    short_operations = operation_df[operation_df['duration_hours'] < expected_hours['min']]
    
    for _, op in long_operations.iterrows():
        anomalies.append({
            'appliance': appliance_col,
            'type': 'Long Operation',
            'description': f'Operation duration {op["duration_hours"]:.2f}h exceeds expected maximum {expected_hours["max"]}h',
            'start_time': op['start_time'],
            'severity': 'Medium'
        })
    
    for _, op in short_operations.iterrows():
        anomalies.append({
            'appliance': appliance_col,
            'type': 'Short Operation',
            'description': f'Operation duration {op["duration_hours"]:.2f}h below expected minimum {expected_hours["min"]}h',
            'start_time': op['start_time'],
            'severity': 'Low'
        })
    
    return anomalies

# ================= NEW GRAPHS FOR ALL APARTMENTS =================
def plot_hourly_consumption_all_apartments(all_dfs):
    """Plot average consumption by hour of the day for all apartments (Scatter Plot)"""
    hourly_data = []
    
    for apt_name, df in all_dfs.items():
        if 'Timestamp' not in df.columns:
            continue
            
        # Get appliance columns
        appliance_cols = [col for col in df.columns 
                         if all(phase not in col.lower() for phase in ['b phase', 'r phase', 'y phase', 'timestamp', 'apartment'])]
        
        if not appliance_cols:
            continue
            
        # Add hour column
        df_temp = df.copy()
        df_temp['hour'] = df_temp['Timestamp'].dt.hour
        
        # Calculate total consumption per hour
        hourly_avg = df_temp.groupby('hour')[appliance_cols].sum().sum(axis=1)
        
        for hour, consumption in hourly_avg.items():
            hourly_data.append({
                'Apartment': apt_name,
                'Hour': hour,
                'Average Consumption (kWh)': consumption
            })
    
    if not hourly_data:
        st.warning("No data available for hourly consumption analysis")
        return
        
    hourly_df = pd.DataFrame(hourly_data)
    
    # Create scatter plot
    fig = px.scatter(hourly_df, x='Hour', y='Average Consumption (kWh)', 
                     color='Apartment', title='Average Consumption by Hour of Day - All Apartments',
                     labels={'Hour': 'Hour of Day', 'Average Consumption (kWh)': 'Average Consumption (kWh)'})
    
    fig.update_traces(marker=dict(size=8, opacity=0.7))
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

def plot_weekday_weekend_comparison_all_apartments(all_dfs):
    """Plot kWh/Day for Weekdays vs Weekends for all apartments (Scatter Plot)"""
    comparison_data = []
    
    for apt_name, df in all_dfs.items():
        if 'Timestamp' not in df.columns:
            continue
            
        # Get appliance columns
        appliance_cols = [col for col in df.columns 
                         if all(phase not in col.lower() for phase in ['b phase', 'r phase', 'y phase', 'timestamp', 'apartment'])]
        
        if not appliance_cols:
            continue
            
        # Add day type and date columns
        df_temp = df.copy()
        df_temp['date'] = df_temp['Timestamp'].dt.date
        df_temp['day_type'] = df_temp['Timestamp'].dt.dayofweek.apply(lambda x: 'Weekend' if x >= 5 else 'Weekday')
        
        # Calculate daily consumption
        daily_consumption = df_temp.groupby('date')[appliance_cols].sum().sum(axis=1)
        daily_consumption = daily_consumption.reset_index()
        daily_consumption.columns = ['date', 'daily_kwh']
        
        # Merge back day_type
        day_types = df_temp[['date', 'day_type']].drop_duplicates()
        daily_consumption = daily_consumption.merge(day_types, on='date')
        
        # Calculate average by day type
        avg_weekday = daily_consumption[daily_consumption['day_type'] == 'Weekday']['daily_kwh'].mean()
        avg_weekend = daily_consumption[daily_consumption['day_type'] == 'Weekend']['daily_kwh'].mean()
        
        if not np.isnan(avg_weekday) and not np.isnan(avg_weekend):
            comparison_data.append({
                'Apartment': apt_name,
                'Day Type': 'Weekday',
                'Average Daily Consumption (kWh)': avg_weekday
            })
            comparison_data.append({
                'Apartment': apt_name,
                'Day Type': 'Weekend', 
                'Average Daily Consumption (kWh)': avg_weekend
            })
    
    if not comparison_data:
        st.warning("No data available for weekday/weekend comparison")
        return
        
    comp_df = pd.DataFrame(comparison_data)
    
    # Create scatter plot
    fig = px.scatter(comp_df, x='Day Type', y='Average Daily Consumption (kWh)', 
                     color='Apartment', title='Weekday vs Weekend Average Daily Consumption - All Apartments',
                     labels={'Average Daily Consumption (kWh)': 'Average Daily Consumption (kWh)'})
    
    fig.update_traces(marker=dict(size=12, opacity=0.7))
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

def plot_running_hours_box_plot(all_dfs):
    """Plot running hours for AC, Geyser, Lights, Plugs, Points (Box and Whiskers)"""
    running_hours_data = []
    
    # Define appliance categories with more comprehensive keywords
    appliance_categories = {
        'AC': ['ac', 'air conditioner', 'air conditioning', 'cooling'],
        'Geyser': ['geyser', 'water heater', 'heater', 'hot water'],
        'Lights': ['light', 'lights', 'lamp', 'bulb', 'lighting', 'lumin'],
        'Plugs': ['plug', 'socket', 'outlet', 'point', 'receptacle'],
        'Points': ['point', 'connection', 'terminal', 'junction']
    }
    
    # Debug information
    total_appliances_checked = 0
    appliances_with_data = 0
    
    for apt_name, df in all_dfs.items():
        if 'Timestamp' not in df.columns:
            continue
            
        # Get all appliance columns (exclude phase columns and metadata)
        appliance_cols = [col for col in df.columns 
                         if all(phase not in col.lower() for phase in ['phase', 'timestamp', 'apartment', 'date', 'time'])]
        
        for col in appliance_cols:
            total_appliances_checked += 1
            col_lower = col.lower()
            category = None
            
            # Categorize appliance
            for cat, keywords in appliance_categories.items():
                if any(keyword in col_lower for keyword in keywords):
                    category = cat
                    break
            
            if category is None:
                # If no category found, skip
                continue
                
            # Debug: Check if appliance has any non-zero data
            non_zero_count = (df[col] > 0.01).sum()
            total_records = len(df[col].dropna())
            
            if non_zero_count == 0:
                continue
                
            appliances_with_data += 1
            
            # Calculate operation durations using improved function
            operation_df = calculate_operation_duration_improved(df, col)
            
            if not operation_df.empty:
                for _, operation in operation_df.iterrows():
                    running_hours_data.append({
                        'Apartment': apt_name,
                        'Appliance Category': category,
                        'Appliance': col,
                        'Running Hours': operation['duration_hours'],
                        'Start Time': operation['start_time'],
                        'Average Power': operation['avg_power']
                    })
    
    # Debug information in sidebar
    with st.sidebar.expander("Running Hours Debug Info"):
        st.write(f"**Total appliances checked**: {total_appliances_checked}")
        st.write(f"**Appliances with data**: {appliances_with_data}")
        st.write(f"**Total ON events found**: {len(running_hours_data)}")
    
    if not running_hours_data:
        st.warning("""
        No running hours data available for analysis. This could be due to:
        
        1. **Threshold too high**: The power threshold (0.01 kWh) might be too high for your data
        2. **Data format**: Your appliance columns might not follow expected naming patterns
        3. **No ON events**: Appliances might not have significant ON periods above threshold
        
        **Troubleshooting steps**:
        - Check if your appliance columns contain keywords like: AC, Geyser, Light, Plug, Point
        - Verify that appliances have power consumption above 0.01 kWh
        """)
        
        # Show sample of available columns for debugging
        if all_dfs:
            sample_df = next(iter(all_dfs.values()))
            available_cols = [col for col in sample_df.columns 
                            if all(phase not in col.lower() for phase in ['phase', 'timestamp', 'apartment', 'date', 'time'])]
            st.info(f"Available appliance columns (first 10): {available_cols[:10]}")
            
        return
        
    running_df = pd.DataFrame(running_hours_data)
    
    # Remove outliers for better visualization (top 1%)
    if len(running_df) > 0:
        Q1 = running_df['Running Hours'].quantile(0.01)
        Q3 = running_df['Running Hours'].quantile(0.99)
        running_df = running_df[(running_df['Running Hours'] >= Q1) & (running_df['Running Hours'] <= Q3)]
    
    # Create box plot
    fig = px.box(running_df, x='Appliance Category', y='Running Hours', 
                 color='Apartment', 
                 title='Running Hours Distribution by Appliance Category - All Apartments',
                 labels={'Running Hours': 'Running Hours', 'Appliance Category': 'Appliance Category'})
    
    fig.update_layout(
        height=600, 
        xaxis_tickangle=-45,
        showlegend=True
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Show statistics
    st.subheader("Running Hours Statistics")
    stats_df = running_df.groupby(['Appliance Category', 'Apartment'])['Running Hours'].agg([
        'count', 'mean', 'median', 'min', 'max', 'std'
    ]).round(3).reset_index()
    
    st.dataframe(stats_df, use_container_width=True)

# ================= EXISTING GRAPH FUNCTIONS =================
def plot_energy_consumption_over_time(df, apartment_name, tab_name):
    """Plot energy consumption over time for all appliances"""
    # Get non-phase columns (appliances)
    appliance_cols = [col for col in df.columns 
                     if all(phase not in col.lower() for phase in ['b phase', 'r phase', 'y phase', 'timestamp', 'apartment'])]
    
    if not appliance_cols:
        st.warning("No appliance data found for this apartment")
        return
    
    # Melt data for plotting
    melt_df = df.melt(id_vars=['Timestamp'], value_vars=appliance_cols, 
                     var_name='Appliance', value_name='Energy Consumption')
    melt_df = melt_df.dropna()
    
    if melt_df.empty:
        st.warning("No valid energy consumption data found")
        return
    
    fig = px.line(melt_df, x='Timestamp', y='Energy Consumption', color='Appliance',
                  title=f'{apartment_name} - Energy Consumption Over Time',
                  labels={'Energy Consumption': 'Energy (kWh)', 'Timestamp': 'Time'})
    
    fig.update_layout(height=500, showlegend=True)
    st.plotly_chart(fig, use_container_width=True, key=f"{apartment_name}_energy_time_{tab_name}")

def plot_appliance_wise_energy(df, apartment_name, tab_name):
    """Plot appliance-wise total energy usage"""
    appliance_cols = [col for col in df.columns 
                     if all(phase not in col.lower() for phase in ['b phase', 'r phase', 'y phase', 'timestamp', 'apartment'])]
    
    if not appliance_cols:
        return
    
    # Calculate total energy per appliance
    energy_totals = []
    for col in appliance_cols:
        total_energy = df[col].sum()
        energy_totals.append({'Appliance': col, 'Total Energy (kWh)': total_energy})
    
    energy_df = pd.DataFrame(energy_totals).sort_values('Total Energy (kWh)', ascending=False)
    
    fig = px.bar(energy_df, x='Appliance', y='Total Energy (kWh)',
                 title=f'{apartment_name} - Appliance-wise Total Energy Usage',
                 color='Total Energy (kWh)')
    
    fig.update_layout(xaxis_tickangle=-45, height=500)
    st.plotly_chart(fig, use_container_width=True, key=f"{apartment_name}_appliance_wise_{tab_name}")

def plot_weekday_weekend_comparison(df, apartment_name, tab_name):
    """Compare energy consumption on weekdays vs weekends"""
    appliance_cols = [col for col in df.columns 
                     if all(phase not in col.lower() for phase in ['b phase', 'r phase', 'y phase', 'timestamp', 'apartment'])]
    
    if not appliance_cols or 'Timestamp' not in df.columns:
        return
    
    # Add day type column
    data = df.copy()
    data['day_type'] = data['Timestamp'].dt.dayofweek.apply(lambda x: 'Weekend' if x >= 5 else 'Weekday')
    
    # Calculate average consumption by appliance and day type
    comparison_data = []
    for col in appliance_cols:
        weekday_avg = data[data['day_type'] == 'Weekday'][col].mean()
        weekend_avg = data[data['day_type'] == 'Weekend'][col].mean()
        
        if not np.isnan(weekday_avg) and not np.isnan(weekend_avg):
            comparison_data.extend([
                {'Appliance': col, 'Day Type': 'Weekday', 'Average Energy (kWh)': weekday_avg},
                {'Appliance': col, 'Day Type': 'Weekend', 'Average Energy (kWh)': weekend_avg}
            ])
    
    comp_df = pd.DataFrame(comparison_data)
    comp_df = comp_df.dropna()
    
    if comp_df.empty:
        st.warning("No data available for weekday/weekend comparison")
        return
    
    fig = px.bar(comp_df, x='Appliance', y='Average Energy (kWh)', color='Day Type',
                 barmode='group', title=f'{apartment_name} - Weekday vs Weekend Energy Consumption')
    
    fig.update_layout(xaxis_tickangle=-45, height=500)
    st.plotly_chart(fig, use_container_width=True, key=f"{apartment_name}_weekday_weekend_{tab_name}")

def plot_ac_energy_time_bins(df, apartment_name, tab_name):
    """Plot AC energy consumption in bins vs time of day"""
    # Find AC columns
    ac_cols = [col for col in df.columns if any(word in col.lower() for word in ['ac', 'air conditioner', 'air conditioning'])]
    
    if not ac_cols:
        st.info("No AC data found for this apartment")
        return
    
    for i, ac_col in enumerate(ac_cols):
        # Filter only when AC is ON (energy > threshold)
        ac_data = df[['Timestamp', ac_col]].copy()
        ac_data = ac_data[ac_data[ac_col] > 0.01]  # Threshold for ON state
        
        if ac_data.empty:
            continue
        
        # Extract hour of day
        ac_data['hour'] = ac_data['Timestamp'].dt.hour
        ac_data['energy_bin'] = pd.cut(ac_data[ac_col], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
        
        # Create pivot table for heatmap
        pivot_data = ac_data.groupby(['hour', 'energy_bin']).size().unstack(fill_value=0)
        
        fig = px.imshow(pivot_data.T, 
                       title=f'{apartment_name} - {ac_col} Energy Consumption by Time of Day',
                       labels=dict(x="Hour of Day", y="Energy Level", color="Count"),
                       aspect="auto")
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True, key=f"{apartment_name}_ac_bins_{i}_{tab_name}")

def plot_on_off_occurrences(df, apartment_name, timeframe='daily', tab_name=""):
    """Plot number of ON/OFF occurrences for appliances"""
    appliance_cols = [col for col in df.columns 
                     if all(phase not in col.lower() for phase in ['b phase', 'r phase', 'y phase', 'timestamp', 'apartment'])]
    
    if not appliance_cols:
        return
    
    # Analyze ON/OFF events for each appliance
    occurrences_data = []
    
    for appliance in appliance_cols:
        on_events = detect_on_off_events(df, appliance)
        
        if on_events.empty:
            continue
        
        if timeframe == 'daily':
            # Group by date
            on_events['date'] = on_events['Timestamp'].dt.date
            daily_counts = on_events.groupby('date').size().reset_index(name='count')
            daily_counts['appliance'] = appliance
            
            for _, row in daily_counts.iterrows():
                occurrences_data.append({
                    'Date': row['date'],
                    'Appliance': appliance,
                    'ON_Events': row['count']
                })
        
        elif timeframe == 'weekly':
            # Group by week
            on_events['week'] = on_events['Timestamp'].dt.isocalendar().week
            on_events['year'] = on_events['Timestamp'].dt.year
            weekly_counts = on_events.groupby(['year', 'week']).size().reset_index(name='count')
            weekly_counts['appliance'] = appliance
            
            for _, row in weekly_counts.iterrows():
                occurrences_data.append({
                    'Week': f"{row['year']}-W{row['week']}",
                    'Appliance': appliance,
                    'ON_Events': row['count']
                })
    
    if not occurrences_data:
        st.info("No ON/OFF events detected for the selected timeframe")
        return
    
    occ_df = pd.DataFrame(occurrences_data)
    
    if timeframe == 'daily':
        fig = px.line(occ_df, x='Date', y='ON_Events', color='Appliance',
                     title=f'{apartment_name} - Daily ON Events for Appliances')
    else:
        fig = px.bar(occ_df, x='Week', y='ON_Events', color='Appliance',
                    title=f'{apartment_name} - Weekly ON Events for Appliances',
                    barmode='group')
    
    fig.update_layout(height=500, showlegend=True)
    st.plotly_chart(fig, use_container_width=True, key=f"{apartment_name}_on_off_{timeframe}_{tab_name}")

def plot_operation_duration(df, apartment_name, tab_name):
    """Plot average duration of operation for appliances per week"""
    appliance_cols = [col for col in df.columns 
                     if all(phase not in col.lower() for phase in ['b phase', 'r phase', 'y phase', 'timestamp', 'apartment'])]
    
    if not appliance_cols:
        return
    
    duration_data = []
    
    for appliance in appliance_cols:
        operation_df = calculate_operation_duration_improved(df, appliance)
        
        if operation_df.empty:
            continue
        
        # Group by week and calculate average duration
        operation_df['week'] = operation_df['start_time'].dt.isocalendar().week
        operation_df['year'] = operation_df['start_time'].dt.year
        
        weekly_avg = operation_df.groupby(['year', 'week'])['duration_hours'].mean().reset_index()
        weekly_avg['appliance'] = appliance
        
        for _, row in weekly_avg.iterrows():
            duration_data.append({
                'Week': f"{row['year']}-W{row['week']}",
                'Appliance': appliance,
                'Avg_Duration_Hours': row['duration_hours']
            })
    
    if not duration_data:
        st.info("No operation duration data available")
        return
    
    duration_df = pd.DataFrame(duration_data)
    
    fig = px.line(duration_df, x='Week', y='Avg_Duration_Hours', color='Appliance',
                 title=f'{apartment_name} - Average Operation Duration per Week',
                 labels={'Avg_Duration_Hours': 'Average Duration (Hours)'})
    
    fig.update_layout(height=500, showlegend=True)
    st.plotly_chart(fig, use_container_width=True, key=f"{apartment_name}_operation_duration_{tab_name}")

def plot_appliance_operation_heatmap(df, apartment_name, tab_name):
    """Create heatmap of appliance operation by hour and day of week"""
    appliance_cols = [col for col in df.columns 
                     if all(phase not in col.lower() for phase in ['b phase', 'r phase', 'y phase', 'timestamp', 'apartment'])]
    
    if not appliance_cols or 'Timestamp' not in df.columns:
        return
    
    # Select top 5 appliances by total energy for clarity
    energy_totals = {col: df[col].sum() for col in appliance_cols}
    top_appliances = sorted(energy_totals, key=energy_totals.get, reverse=True)[:5]
    
    for i, appliance in enumerate(top_appliances):
        # Filter ON periods
        appliance_data = df[['Timestamp', appliance]].copy()
        appliance_data = appliance_data[appliance_data[appliance] > 0.01]
        
        if appliance_data.empty:
            continue
        
        # Extract time features
        appliance_data['hour'] = appliance_data['Timestamp'].dt.hour
        appliance_data['day_of_week'] = appliance_data['Timestamp'].dt.day_name()
        
        # Create pivot table
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        pivot_table = appliance_data.groupby(['day_of_week', 'hour']).size().unstack(fill_value=0)
        pivot_table = pivot_table.reindex(days_order)
        
        fig = px.imshow(pivot_table, 
                       title=f'{apartment_name} - {appliance} Operation Pattern',
                       labels=dict(x="Hour of Day", y="Day of Week", color="Operation Count"),
                       aspect="auto")
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True, key=f"{apartment_name}_heatmap_{i}_{tab_name}")

# ================= DATA DIAGNOSIS FUNCTION =================
def diagnose_data_issues(all_dfs):
    """Diagnose why running hours aren't being calculated"""
    st.header("🔍 Data Diagnosis")
    
    for apt_name, df in all_dfs.items():
        with st.expander(f"Diagnosis for {apt_name}"):
            # Show basic info
            st.write(f"Total records: {len(df)}")
            if 'Timestamp' in df.columns:
                st.write(f"Date range: {df['Timestamp'].min()} to {df['Timestamp'].max()}")
            
            # Show available columns
            appliance_cols = [col for col in df.columns 
                             if all(phase not in col.lower() for phase in ['phase', 'timestamp', 'apartment', 'date', 'time'])]
            st.write(f"Potential appliance columns: {appliance_cols}")
            
            # Show sample data for first few appliance columns
            if appliance_cols:
                sample_cols = appliance_cols[:5]  # First 5 columns
                for col in sample_cols:
                    st.write(f"**{col}**:")
                    col_data = df[col].dropna()
                    non_zero = (col_data > 0.01).sum()
                    st.write(f"  - Non-zero values (>0.01): {non_zero}/{len(col_data)} ({non_zero/len(col_data)*100:.1f}%)")
                    st.write(f"  - Mean: {col_data.mean():.4f}")
                    st.write(f"  - Max: {col_data.max():.4f}")
                    st.write(f"  - Min: {col_data.min():.4f}")
                    
                    # Test the calculation function
                    test_result = calculate_operation_duration_improved(df, col)
                    st.write(f"  - ON events detected: {len(test_result)}")

# ================= MAIN APPLICATION =================
def main():
    st.set_page_config(page_title="Apartment Energy Analytics", layout="wide")
    st.title("🏢 Apartment Energy Consumption Dashboard")
    
    # Load data
    with st.spinner("Loading apartment data..."):
        all_dfs = load_all_data()
    
    if not all_dfs:
        st.error("No data files found. Please check the DATA_FOLDER path.")
        st.stop()
    
    st.success(f"✅ Loaded data for {len(all_dfs)} apartments")
    
    # Sidebar for apartment selection
    st.sidebar.header("Apartment Selection")
    apartment_names = sorted(list(all_dfs.keys()))
    
    # Add "All Apartments" option
    selected_option = st.sidebar.selectbox("Select Option", ["All Apartments"] + apartment_names)
    
    # Show data diagnosis in sidebar
    if st.sidebar.checkbox("Show Data Diagnosis", False):
        diagnose_data_issues(all_dfs)
    
    # Show All Apartments analysis if selected
    if selected_option == "All Apartments":
        st.header("🏘️ All Apartments - Comparative Analysis")
        
        # Show the three new graphs
        plot_hourly_consumption_all_apartments(all_dfs)
        plot_weekday_weekend_comparison_all_apartments(all_dfs)
        plot_running_hours_box_plot(all_dfs)
        
        return  # Exit early since we're showing only comparative analysis
    
    # Otherwise, continue with individual apartment analysis
    selected_apartment = selected_option
    
    if selected_apartment not in all_dfs:
        st.error("Selected apartment data not found")
        return
    
    # Get selected apartment data
    df = all_dfs[selected_apartment].copy()
    
    # Basic info
    st.sidebar.subheader("Apartment Info")
    st.sidebar.write(f"**Apartment:** {selected_apartment}")
    st.sidebar.write(f"**Total Records:** {len(df)}")
    
    if 'Timestamp' in df.columns:
        st.sidebar.write(f"**Date Range:** {df['Timestamp'].min().strftime('%Y-%m-%d')} to {df['Timestamp'].max().strftime('%Y-%m-%d')}")
    
    # Main dashboard
    st.header(f"Energy Analytics for {selected_apartment}")
    
    # Tabs for different analyses
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📈 Overview", 
        "🔌 Appliance Usage", 
        "📅 Time Analysis", 
        "❄️ AC Analysis", 
        "🔄 ON/OFF Events", 
        "⏱️ Operation Duration",
        "🚨 Anomalies"
    ])
    
    with tab1:
        st.subheader("Energy Consumption Overview")
        plot_energy_consumption_over_time(df, selected_apartment, "overview")
        
        col1, col2 = st.columns(2)
        with col1:
            plot_appliance_wise_energy(df, selected_apartment, "overview")
        with col2:
            plot_weekday_weekend_comparison(df, selected_apartment, "overview")
    
    with tab2:
        st.subheader("Detailed Appliance Analysis")
        plot_appliance_operation_heatmap(df, selected_apartment, "appliance_usage")
        
        # Individual appliance selection for detailed view
        appliance_cols = [col for col in df.columns 
                         if all(phase not in col.lower() for phase in ['b phase', 'r phase', 'y phase', 'timestamp', 'apartment'])]
        
        if appliance_cols:
            selected_appliance = st.selectbox("Select Appliance for Detailed View", appliance_cols, key="appliance_select")
            
            if selected_appliance in df.columns:
                # Time series for selected appliance
                fig = px.line(df, x='Timestamp' if 'Timestamp' in df.columns else df.index, 
                             y=selected_appliance,
                             title=f'{selected_appliance} - Energy Consumption Over Time')
                st.plotly_chart(fig, use_container_width=True, key=f"{selected_apartment}_appliance_detail")
                
                # Statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Energy", f"{df[selected_appliance].sum():.2f} kWh")
                with col2:
                    st.metric("Average Power", f"{df[selected_appliance].mean():.2f} kWh")
                with col3:
                    st.metric("Max Power", f"{df[selected_appliance].max():.2f} kWh")
                with col4:
                    on_time = (df[selected_appliance] > 0.01).sum()
                    st.metric("ON Time Records", f"{on_time}")
    
    with tab3:
        st.subheader("Time-based Analysis")
        plot_weekday_weekend_comparison(df, selected_apartment, "time_analysis")
        
        # Hourly analysis
        if 'Timestamp' in df.columns:
            df['hour'] = df['Timestamp'].dt.hour
            appliance_cols = [col for col in df.columns 
                            if all(phase not in col.lower() for phase in ['b phase', 'r phase', 'y phase', 'timestamp', 'apartment', 'hour'])]
            
            if appliance_cols:
                hourly_avg = df.groupby('hour')[appliance_cols].mean().mean(axis=1)
                
                fig = px.line(x=hourly_avg.index, y=hourly_avg.values,
                            title='Average Energy Consumption by Hour of Day',
                            labels={'x': 'Hour of Day', 'y': 'Average Energy (kWh)'})
                st.plotly_chart(fig, use_container_width=True, key=f"{selected_apartment}_hourly_avg")
    
    with tab4:
        st.subheader("AC Energy Analysis")
        plot_ac_energy_time_bins(df, selected_apartment, "ac_analysis")
        
        # Additional AC analysis
        ac_cols = [col for col in df.columns if any(word in col.lower() for word in ['ac', 'air conditioner', 'air conditioning'])]
        
        if ac_cols:
            for i, ac_col in enumerate(ac_cols):
                ac_data = df[['Timestamp', ac_col]].copy()
                ac_data = ac_data[ac_data[ac_col] > 0.01]
                
                if not ac_data.empty:
                    ac_data['hour'] = ac_data['Timestamp'].dt.hour
                    hourly_ac = ac_data.groupby('hour')[ac_col].mean()
                    
                    fig = px.bar(x=hourly_ac.index, y=hourly_ac.values,
                                title=f'{ac_col} - Average Consumption by Hour',
                                labels={'x': 'Hour of Day', 'y': 'Average Energy (kWh)'})
                    st.plotly_chart(fig, use_container_width=True, key=f"{selected_apartment}_ac_hourly_{i}")
    
    with tab5:
        st.subheader("ON/OFF Events Analysis")
        
        timeframe = st.radio("Select Timeframe", ["daily", "weekly"], horizontal=True, key="timeframe_select")
        plot_on_off_occurrences(df, selected_apartment, timeframe, "on_off_events")
    
    with tab6:
        st.subheader("Operation Duration Analysis")
        plot_operation_duration(df, selected_apartment, "operation_duration")
        
        # Duration statistics by appliance
        duration_stats = []
        appliance_cols = [col for col in df.columns 
                         if all(phase not in col.lower() for phase in ['b phase', 'r phase', 'y phase', 'timestamp', 'apartment'])]
        
        for appliance in appliance_cols:
            operation_df = calculate_operation_duration_improved(df, appliance)
            if not operation_df.empty:
                avg_duration = operation_df['duration_hours'].mean()
                max_duration = operation_df['duration_hours'].max()
                duration_stats.append({
                    'Appliance': appliance,
                    'Avg Duration (h)': avg_duration,
                    'Max Duration (h)': max_duration,
                    'Operation Count': len(operation_df)
                })
        
        if duration_stats:
            stats_df = pd.DataFrame(duration_stats)
            st.dataframe(stats_df.round(2), use_container_width=True)
    
    with tab7:
        st.subheader("Operation Anomaly Detection")
        
        # Detect anomalies for all appliances
        all_anomalies = []
        appliance_cols = [col for col in df.columns 
                         if all(phase not in col.lower() for phase in ['b phase', 'r phase', 'y phase', 'timestamp', 'apartment'])]
        
        with st.spinner("Detecting anomalies..."):
            for appliance in appliance_cols:
                anomalies = detect_anomalies_operation(df, appliance)
                all_anomalies.extend(anomalies)
        
        if all_anomalies:
            anomalies_df = pd.DataFrame(all_anomalies)
            
            # Severity counts
            severity_counts = anomalies_df['severity'].value_counts()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Anomalies", len(all_anomalies))
            with col2:
                st.metric("High Severity", severity_counts.get('High', 0))
            with col3:
                st.metric("Medium Severity", severity_counts.get('Medium', 0))
            
            # Display anomalies
            st.subheader("Detected Anomalies")
            for severity in ['High', 'Medium', 'Low']:
                severity_anomalies = anomalies_df[anomalies_df['severity'] == severity]
                if not severity_anomalies.empty:
                    with st.expander(f"{severity} Severity Anomalies ({len(severity_anomalies)})"):
                        st.dataframe(severity_anomalies[['appliance', 'type', 'description', 'start_time']], use_container_width=True)
        else:
            st.success("✅ No operation anomalies detected!")

if __name__ == "__main__":
    main()