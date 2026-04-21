# Run with: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(layout="wide", page_title="BizAnalytics", page_icon="📊")

# Custom CSS for styling
st.markdown("""
<style>
.metric-card {
    background-color: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 8px;
    padding: 15px;
    margin: 10px 0;
}
.stMetric {
    background-color: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 8px;
    padding: 15px;
}
.sidebar-title {
    font-size: 24px;
    font-weight: bold;
    color: #1f77b4;
}
.sidebar-subtitle {
    font-size: 14px;
    color: #666;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# Sidebar title and subtitle
st.sidebar.markdown('<div class="sidebar-title">📊 BizAnalytics</div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="sidebar-subtitle">Smart Sales System v1.0</div>', unsafe_allow_html=True)

# Data Generation Function
@st.cache_data
def generate_data():
    np.random.seed(42)
    
    # Generate date range
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2022, 12, 31)
    dates = pd.date_range(start_date, end_date, freq='D')
    
    # Create base data
    data = []
    categories = ['Electronics', 'Clothing', 'Food', 'Home & Garden', 'Sports']
    regions = ['North', 'South', 'East', 'West']
    
    for date in dates:
        for _ in range(np.random.randint(1, 4)):  # 1-3 transactions per day
            # Generate random values
            category = np.random.choice(categories)
            region = np.random.choice(regions)
            units_sold = np.random.randint(10, 200)
            unit_price = np.random.uniform(10, 500)
            discount = np.random.uniform(0, 0.30)
            marketing_spend = np.random.uniform(500, 5000)
            temperature = np.random.uniform(5, 40)
            is_weekend = 1 if pd.Timestamp(date).weekday() >= 5 else 0
            is_holiday = np.random.choice([0, 1], p=[0.9, 0.1])  # 10% chance of holiday
            
            # Calculate revenue with noise
            base_revenue = units_sold * unit_price * (1 - discount)
            marketing_contribution = marketing_spend * 0.15
            weekend_bonus = is_weekend * 200
            holiday_bonus = is_holiday * 500
            noise = np.random.normal(0, 100)
            
            revenue = base_revenue + marketing_contribution + weekend_bonus + holiday_bonus + noise
            
            data.append({
                'date': date,
                'category': category,
                'region': region,
                'units_sold': units_sold,
                'unit_price': unit_price,
                'discount': discount,
                'marketing_spend': marketing_spend,
                'temperature': temperature,
                'is_weekend': is_weekend,
                'is_holiday': is_holiday,
                'revenue': revenue
            })
    
    df = pd.DataFrame(data)
    
    # Add time-based features
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    df['year'] = df['date'].dt.year
    df['day_of_week'] = df['date'].dt.dayofweek
    
    # Ensure we have exactly 1000 rows
    if len(df) > 1000:
        df = df.sample(n=1000, random_state=42)
    elif len(df) < 1000:
        # Add more rows if needed
        additional_rows = 1000 - len(df)
        extra_data = []
        for _ in range(additional_rows):
            date = np.random.choice(dates)
            category = np.random.choice(categories)
            region = np.random.choice(regions)
            units_sold = np.random.randint(10, 200)
            unit_price = np.random.uniform(10, 500)
            discount = np.random.uniform(0, 0.30)
            marketing_spend = np.random.uniform(500, 5000)
            temperature = np.random.uniform(5, 40)
            is_weekend = 1 if pd.Timestamp(date).weekday() >= 5 else 0
            is_holiday = np.random.choice([0, 1], p=[0.9, 0.1])
            
            base_revenue = units_sold * unit_price * (1 - discount)
            marketing_contribution = marketing_spend * 0.15
            weekend_bonus = is_weekend * 200
            holiday_bonus = is_holiday * 500
            noise = np.random.normal(0, 100)
            revenue = base_revenue + marketing_contribution + weekend_bonus + holiday_bonus + noise
            
            extra_data.append({
                'date': date,
                'category': category,
                'region': region,
                'units_sold': units_sold,
                'unit_price': unit_price,
                'discount': discount,
                'marketing_spend': marketing_spend,
                'temperature': temperature,
                'is_weekend': is_weekend,
                'is_holiday': is_holiday,
                'revenue': revenue,
                'month': pd.Timestamp(date).month,
                'quarter': pd.Timestamp(date).quarter,
                'year': pd.Timestamp(date).year,
                'day_of_week': pd.Timestamp(date).dayofweek
            })
        
        df = pd.concat([df, pd.DataFrame(extra_data)], ignore_index=True)
        df = df.sample(n=1000, random_state=42).reset_index(drop=True)
    
    return df

# ML Models Training Function
@st.cache_resource
def train_models(df):
    with st.spinner("Training models..."):
        # Preprocessing
        df_ml = df.copy()
        
        # Label encoding
        le_category = LabelEncoder()
        le_region = LabelEncoder()
        df_ml['category_encoded'] = le_category.fit_transform(df_ml['category'])
        df_ml['region_encoded'] = le_region.fit_transform(df_ml['region'])
        
        # Feature selection
        features = ['units_sold', 'unit_price', 'discount', 'marketing_spend', 
                   'temperature', 'is_weekend', 'is_holiday', 'month', 'quarter', 
                   'day_of_week', 'category_encoded', 'region_encoded']
        
        X = df_ml[features]
        y = df_ml['revenue']
        
        # Standardization
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        # Train models
        models = {}
        
        # Linear Regression
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        models['Linear Regression'] = lr
        
        # Random Forest
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        models['Random Forest'] = rf
        
        # XGBoost
        xgb_model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)
        xgb_model.fit(X_train, y_train)
        models['XGBoost'] = xgb_model
        
        # Moving Average Baseline
        weekly_revenue = df_ml.set_index('date')['revenue'].resample('W').mean()
        ma_baseline = weekly_revenue.rolling(window=4).mean().iloc[-1]
        
        # Evaluate models
        results = []
        for name, model in models.items():
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            results.append({
                'Model': name,
                'MAE': mae,
                'RMSE': rmse,
                'R²': r2
            })
        
        # Add Moving Average baseline
        y_ma = np.full_like(y_test, ma_baseline)
        mae_ma = mean_absolute_error(y_test, y_ma)
        rmse_ma = np.sqrt(mean_squared_error(y_test, y_ma))
        r2_ma = r2_score(y_test, y_ma)
        results.append({
            'Model': 'Moving Average',
            'MAE': mae_ma,
            'RMSE': rmse_ma,
            'R²': r2_ma
        })
        
        results_df = pd.DataFrame(results)
        
        return {
            'models': models,
            'results': results_df,
            'scaler': scaler,
            'le_category': le_category,
            'le_region': le_region,
            'features': features,
            'X_test': X_test,
            'y_test': y_test,
            'ma_baseline': ma_baseline
        }

# Generate data
df = generate_data()

# Train models
ml_results = train_models(df)

# Sidebar navigation
page = st.sidebar.radio("Navigate", [
    "Overview Dashboard",
    "Exploratory Data Analysis", 
    "ML Models & Comparison",
    "Sales Predictor",
    "Data Explorer"
])

# Page 1: Overview Dashboard
if page == "Overview Dashboard":
    st.title("📊 Overview Dashboard")
    
    # Sidebar filters
    st.sidebar.header("Filters")
    selected_categories = st.sidebar.multiselect("Select Categories", df['category'].unique(), default=df['category'].unique())
    selected_regions = st.sidebar.multiselect("Select Regions", df['region'].unique(), default=df['region'].unique())
    
    # Filter data
    filtered_df = df[
        (df['category'].isin(selected_categories)) & 
        (df['region'].isin(selected_regions))
    ]
    
    # KPI Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_revenue = filtered_df['revenue'].sum()
        st.metric("Total Revenue", f"${total_revenue:,.0f}")
    
    with col2:
        total_units = filtered_df['units_sold'].sum()
        st.metric("Total Units Sold", f"{total_units:,}")
    
    with col3:
        avg_daily_revenue = filtered_df.groupby('date')['revenue'].sum().mean()
        st.metric("Avg Daily Revenue", f"${avg_daily_revenue:,.0f}")
    
    with col4:
        top_category = filtered_df.groupby('category')['revenue'].sum().idxmax()
        st.metric("Top Category", top_category)
    
    # Monthly revenue trend
    monthly_revenue = filtered_df.groupby(['year', 'month'])['revenue'].sum().reset_index()
    monthly_revenue['date'] = pd.to_datetime(monthly_revenue['year'].astype(str) + '-' + monthly_revenue['month'].astype(str) + '-01')
    
    fig_trend = px.line(monthly_revenue, x='date', y='revenue', 
                       title="Monthly Revenue Trend",
                       template="plotly_dark")
    fig_trend.update_layout(showlegend=False)
    st.plotly_chart(fig_trend, use_container_width=True)
    
    # Revenue by category and region
    col1, col2 = st.columns(2)
    
    with col1:
        category_revenue = filtered_df.groupby('category')['revenue'].sum().reset_index()
        fig_category = px.bar(category_revenue, x='category', y='revenue',
                             title="Revenue by Category",
                             template="plotly_dark")
        st.plotly_chart(fig_category, use_container_width=True)
    
    with col2:
        region_revenue = filtered_df.groupby('region')['revenue'].sum().reset_index()
        fig_region = px.pie(region_revenue, values='revenue', names='region',
                            title="Revenue by Region",
                            template="plotly_dark")
        st.plotly_chart(fig_region, use_container_width=True)
    
    # Quarterly revenue comparison
    quarterly_revenue = filtered_df.groupby(['year', 'quarter'])['revenue'].sum().reset_index()
    quarterly_revenue['period'] = quarterly_revenue['year'].astype(str) + '-Q' + quarterly_revenue['quarter'].astype(str)
    
    fig_quarterly = px.bar(quarterly_revenue, x='quarter', y='revenue', color='year',
                          title="Quarterly Revenue Comparison",
                          template="plotly_dark",
                          barmode='group')
    st.plotly_chart(fig_quarterly, use_container_width=True)

# Page 2: Exploratory Data Analysis
elif page == "Exploratory Data Analysis":
    st.title("🔍 Exploratory Data Analysis")
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Distribution", "Relationships", "Time Patterns"])
    
    with tab1:
        # Revenue distribution
        fig_hist = px.histogram(df, x='revenue', nbins=50,
                               title="Revenue Distribution",
                               template="plotly_dark")
        st.plotly_chart(fig_hist, use_container_width=True)
        
        # Revenue by category box plot
        fig_box = px.box(df, x='category', y='revenue',
                        title="Revenue Distribution by Category",
                        template="plotly_dark")
        st.plotly_chart(fig_box, use_container_width=True)
    
    with tab2:
        # Units sold vs revenue scatter plot
        fig_scatter = px.scatter(df, x='units_sold', y='revenue', color='category',
                                title="Units Sold vs Revenue by Category",
                                template="plotly_dark")
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Correlation heatmap
        numeric_cols = ['revenue', 'units_sold', 'unit_price', 'discount', 
                       'marketing_spend', 'temperature', 'is_weekend', 'is_holiday']
        corr_matrix = df[numeric_cols].corr()
        
        fig_heatmap = px.imshow(corr_matrix, 
                               title="Correlation Matrix",
                               template="plotly_dark",
                               color_continuous_scale='RdBu')
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    with tab3:
        # Average revenue by day of week
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        avg_revenue_by_day = df.groupby('day_of_week')['revenue'].mean().reset_index()
        avg_revenue_by_day['day_name'] = avg_revenue_by_day['day_of_week'].map(lambda x: day_names[x])
        
        fig_day = px.bar(avg_revenue_by_day, x='day_name', y='revenue',
                        title="Average Revenue by Day of Week",
                        template="plotly_dark")
        st.plotly_chart(fig_day, use_container_width=True)
    
    # Raw statistics expander
    with st.expander("Raw Statistics"):
        st.write(df.describe())

# Page 3: ML Models & Comparison
elif page == "ML Models & Comparison":
    st.title("🤖 ML Models & Comparison")
    
    results_df = ml_results['results']
    
    # Find best model
    best_model_idx = results_df['R²'].idxmax()
    best_model = results_df.loc[best_model_idx, 'Model']
    
    # Highlight best row in green
    def highlight_best_row(row):
        if row['Model'] == best_model:
            return ['background-color: green'] * len(row)
        else:
            return [''] * len(row)
    
    # Display comparison table
    st.subheader("Model Performance Comparison")
    styled_results = results_df.style.apply(highlight_best_row, axis=1)
    st.dataframe(styled_results)
    
    # R² scores comparison
    fig_r2 = px.bar(results_df, x='Model', y='R²',
                   title="R² Scores Comparison",
                   template="plotly_dark")
    fig_r2.update_layout(yaxis_title="R² Score")
    st.plotly_chart(fig_r2, use_container_width=True)
    
    # XGBoost feature importances
    if 'XGBoost' in ml_results['models']:
        xgb_model = ml_results['models']['XGBoost']
        feature_importance = pd.DataFrame({
            'feature': ml_results['features'],
            'importance': xgb_model.feature_importances_
        }).sort_values('importance', ascending=True)
        
        fig_importance = px.bar(feature_importance, x='importance', y='feature',
                               orientation='h',
                               title="XGBoost Feature Importances",
                               template="plotly_dark")
        st.plotly_chart(fig_importance, use_container_width=True)
        
        # Actual vs Predicted scatter plot
        y_pred = xgb_model.predict(ml_results['X_test'])
        y_actual = ml_results['y_test']
        
        fig_actual_pred = go.Figure()
        fig_actual_pred.add_trace(go.Scatter(
            x=y_actual, y=y_pred,
            mode='markers',
            name='Predictions',
            marker=dict(color='blue', opacity=0.6)
        ))
        
        # Add perfect fit line
        min_val = min(y_actual.min(), y_pred.min())
        max_val = max(y_actual.max(), y_pred.max())
        fig_actual_pred.add_trace(go.Scatter(
            x=[min_val, max_val], y=[min_val, max_val],
            mode='lines',
            name='Perfect Fit',
            line=dict(color='red', dash='dash')
        ))
        
        fig_actual_pred.update_layout(
            title="Actual vs Predicted Revenue (XGBoost)",
            xaxis_title="Actual Revenue",
            yaxis_title="Predicted Revenue",
            template="plotly_dark"
        )
        st.plotly_chart(fig_actual_pred, use_container_width=True)
        
        # Residuals histogram
        residuals = y_actual - y_pred
        fig_residuals = px.histogram(x=residuals, nbins=50,
                                   title="Residuals Distribution (XGBoost)",
                                   template="plotly_dark")
        fig_residuals.update_layout(xaxis_title="Residuals", yaxis_title="Frequency")
        st.plotly_chart(fig_residuals, use_container_width=True)

# Page 4: Sales Predictor
elif page == "Sales Predictor":
    st.title("🔮 Sales Predictor")
    
    # Input widgets in two columns
    col1, col2 = st.columns(2)
    
    with col1:
        units_sold_input = st.number_input("Units Sold", min_value=10, max_value=200, value=100)
        unit_price_input = st.number_input("Unit Price ($)", min_value=10.0, max_value=500.0, value=250.0)
        discount_input = st.number_input("Discount (%)", min_value=0.0, max_value=30.0, value=10.0) / 100.0
        marketing_input = st.number_input("Marketing Spend ($)", min_value=500, max_value=5000, value=2500)
    
    with col2:
        category_input = st.selectbox("Category", df['category'].unique())
        region_input = st.selectbox("Region", df['region'].unique())
        weekend_input = st.radio("Day Type", ["Weekday", "Weekend"])
        is_weekend_input = 1 if weekend_input == "Weekend" else 0
    
    # Predict button
    if st.button("Predict Revenue"):
        # Prepare input data
        input_data = pd.DataFrame({
            'units_sold': [units_sold_input],
            'unit_price': [unit_price_input],
            'discount': [discount_input],
            'marketing_spend': [marketing_input],
            'temperature': [df['temperature'].mean()],  # Use average temperature
            'is_weekend': [is_weekend_input],
            'is_holiday': [0],  # Default to non-holiday
            'month': [6],  # Default to June
            'quarter': [2],  # Default to Q2
            'day_of_week': [1],  # Default to Tuesday
            'category_encoded': [ml_results['le_category'].transform([category_input])[0]],
            'region_encoded': [ml_results['le_region'].transform([region_input])[0]]
        })
        
        # Scale input
        input_scaled = ml_results['scaler'].transform(input_data[ml_results['features']])
        
        # Make prediction
        xgb_model = ml_results['models']['XGBoost']
        prediction = xgb_model.predict(input_scaled)[0]
        
        # Show prediction
        st.success(f"🎯 Predicted Revenue: ${prediction:,.2f}")
        
        # Sensitivity analysis
        st.subheader("Sensitivity Analysis")
        
        sensitivity_data = []
        base_features = input_data[ml_results['features']].iloc[0].copy()
        
        for feature in ['units_sold', 'unit_price', 'discount', 'marketing_spend']:
            # -20% change
            temp_features = base_features.copy()
            if feature != 'discount':
                temp_features[feature] *= 0.8
            else:
                temp_features[feature] = max(0, temp_features[feature] * 0.8)
            
            temp_scaled = ml_results['scaler'].transform([temp_features])
            pred_neg = xgb_model.predict(temp_scaled)[0]
            
            # +20% change
            temp_features = base_features.copy()
            if feature != 'discount':
                temp_features[feature] *= 1.2
            else:
                temp_features[feature] = min(0.3, temp_features[feature] * 1.2)
            
            temp_scaled = ml_results['scaler'].transform([temp_features])
            pred_pos = xgb_model.predict(temp_scaled)[0]
            
            sensitivity_data.append({
                'Feature': feature.replace('_', ' ').title(),
                '-20%': pred_neg,
                'Base': prediction,
                '+20%': pred_pos
            })
        
        sensitivity_df = pd.DataFrame(sensitivity_data)
        
        fig_sensitivity = go.Figure()
        
        for i, feature in enumerate(sensitivity_df['Feature']):
            fig_sensitivity.add_trace(go.Bar(
                name=feature,
                x=['-20%', 'Base', '+20%'],
                y=[sensitivity_df.iloc[i]['-20%'], 
                   sensitivity_df.iloc[i]['Base'], 
                   sensitivity_df.iloc[i]['+20%']],
                offsetgroup=i
            ))
        
        fig_sensitivity.update_layout(
            title="Revenue Sensitivity to Input Changes",
            xaxis_title="Change Percentage",
            yaxis_title="Predicted Revenue ($)",
            template="plotly_dark",
            barmode='group'
        )
        st.plotly_chart(fig_sensitivity, use_container_width=True)
        
        # Revenue breakdown
        st.subheader("Revenue Breakdown")
        base_revenue = units_sold_input * unit_price_input * (1 - discount_input)
        marketing_contribution = marketing_input * 0.15
        weekend_bonus = is_weekend_input * 200
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Base Revenue", f"${base_revenue:,.2f}")
        with col2:
            st.metric("Marketing Contribution", f"${marketing_contribution:,.2f}")
        with col3:
            st.metric("Weekend Bonus", f"${weekend_bonus:,.2f}")

# Page 5: Data Explorer
elif page == "Data Explorer":
    st.title("📋 Data Explorer")
    
    # Sidebar filters
    st.sidebar.header("Filters")
    explore_categories = st.sidebar.multiselect("Categories", df['category'].unique(), default=df['category'].unique())
    explore_regions = st.sidebar.multiselect("Regions", df['region'].unique(), default=df['region'].unique())
    
    # Date range slider
    min_date = df['date'].min().date()
    max_date = df['date'].max().date()
    start_date, end_date = st.sidebar.date_input("Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)
    
    # Filter data
    explore_df = df[
        (df['category'].isin(explore_categories)) &
        (df['region'].isin(explore_regions)) &
        (df['date'].dt.date >= start_date) &
        (df['date'].dt.date <= end_date)
    ]
    
    # Show metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Filtered Rows", f"{len(explore_df):,}")
    with col2:
        st.metric("Filtered Total Revenue", f"${explore_df['revenue'].sum():,.0f}")
    
    # Download button
    csv = explore_df.to_csv(index=False)
    st.download_button(
        label="Download Filtered Data as CSV",
        data=csv,
        file_name='business_analytics_data.csv',
        mime='text/csv'
    )
    
    # Data table
    st.subheader("Dataset")
    st.dataframe(explore_df, use_container_width=True)
    
    # Scatter matrix
    st.subheader("Scatter Matrix of Key Variables")
    key_cols = ['revenue', 'units_sold', 'unit_price', 'marketing_spend']
    
    # Create scatter matrix using plotly
    fig_scatter_matrix = px.scatter_matrix(
        explore_df[key_cols],
        title="Scatter Matrix of Key Variables",
        template="plotly_dark"
    )
    st.plotly_chart(fig_scatter_matrix, use_container_width=True)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown(" 2024 BizAnalytics - Smart Sales System")