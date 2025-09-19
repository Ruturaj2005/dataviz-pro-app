import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
import numpy as np
from datetime import datetime

# Imports for the AI feature
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import os          
from pathlib import Path 
def setup_secrets():
    # Check if we are in a deployed environment by looking for the GOOGLE_API_KEY env var
    if "GOOGLE_API_KEY" in os.environ:
        # Create the .streamlit directory if it doesn't exist
        streamlit_dir = Path(".streamlit")
        streamlit_dir.mkdir(exist_ok=True)
        
        # Create the secrets.toml file
        secrets_file_path = streamlit_dir / "secrets.toml"
        
        # Write the secret from the environment variable into the file
        with open(secrets_file_path, "w") as f:
            f.write(f'GOOGLE_API_KEY = "{os.environ["GOOGLE_API_KEY"]}"\n')
setup_secrets()
# Page configuration
st.set_page_config(
    page_title="DataViz Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header { text-align: center; padding: 20px; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px; margin-bottom: 30px; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1); }
    .metric-card { background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); padding: 20px; border-radius: 15px; text-align: center; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1); border-left: 4px solid #667eea; margin: 10px 0; }
    .insight-box { background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%); padding: 20px; border-radius: 10px; margin: 15px 0; border-left: 4px solid #2196f3; }
    .stButton > button { border-radius: 8px; border: none; background: linear-gradient(45deg, #667eea, #764ba2); color: white; font-weight: bold; padding: 0.5rem 1rem; transition: all 0.3s; }
    .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4); }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown("""
<div class="main-header">
    <h1>🚀 DataViz Pro - Advanced Analytics Platform</h1>
    <p style="font-size: 1.1rem; margin: 0;">Transform your data into beautiful insights with zero coding required!</p>
</div>
""", unsafe_allow_html=True)


# Helper Function for AI Chat Interface
def display_chat_interface(df, page_key):
    st.markdown("---")
    st.subheader("💬 Ask a Follow-up Question")
    st.markdown("Use the AI to dig deeper into the data you see above.")

    if "GOOGLE_API_KEY" not in st.secrets:
        st.error("🚨 Google API key not found! Please add it to your secrets file.")
        st.info("Create a file at `.streamlit/secrets.toml` and add your key: `GOOGLE_API_KEY = 'Your-Key-Here'`")
        return

    try:
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=st.secrets["GOOGLE_API_KEY"])
        agent = create_pandas_dataframe_agent(
            llm, df, agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True, handle_parsing_errors=True, allow_dangerous_code=True
        )
        prompt = st.text_area("e.g., How many rows are there?", key=f"chat_prompt_{page_key}")
        if st.button("🚀 Get Answer", key=f"chat_button_{page_key}"):
            if prompt:
                with st.spinner("💎 Gemini is thinking..."):
                    try:
                        response = agent.invoke(prompt)
                        st.markdown("##### 💡 Answer")
                        st.write(response['output'])
                    except Exception as e:
                        st.error(f"An error occurred: {e}")
            else:
                st.warning("Please enter a question.")
    except Exception as e:
        st.error(f"Failed to initialize AI model: {e}")


# Sidebar navigation
with st.sidebar:
    st.markdown("### 🎯 Navigation")
    menu_options = {
        "🏠 Dashboard": "📂 Data Preview",
        "🔍 Smart Insights": "📑 Auto Insights",
        "🎨 Custom Charts": "📈 Custom Visualization",
        "📄 Export Hub": "📤 Export Report"
    }
    choice_display = st.selectbox("Choose your analysis:", list(menu_options.keys()))
    choice = menu_options[choice_display]
    st.markdown("---")
    st.markdown("### 📁 Upload Your Data")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded_file:
        st.success(f"✅ File loaded: {uploaded_file.name}")

# Main content area
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    for col in df.columns:
        if any(keyword in col.lower() for keyword in ["date", "time", "year", "month"]):
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            except Exception:
                pass

    # ================== DASHBOARD ==================
    if choice == "📂 Data Preview":
        st.markdown("## 🏠 Data Dashboard")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""<div class="metric-card"><h3>📊 Total Rows</h3><h2 style="color: #667eea; margin: 10px 0;">{len(df):,}</h2></div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""<div class="metric-card"><h3>📈 Columns</h3><h2 style="color: #667eea; margin: 10px 0;">{len(df.columns)}</h2></div>""", unsafe_allow_html=True)
        with col3:
            numeric_cols = len(df.select_dtypes(include=np.number).columns)
            st.markdown(f"""<div class="metric-card"><h3>🔢 Numeric</h3><h2 style="color: #667eea; margin: 10px 0;">{numeric_cols}</h2></div>""", unsafe_allow_html=True)
        with col4:
            missing_percent = (df.isnull().sum().sum() / df.size * 100)
            st.markdown(f"""<div class="metric-card"><h3>❌ Missing %</h3><h2 style="color: #667eea; margin: 10px 0;">{missing_percent:.1f}%</h2></div>""", unsafe_allow_html=True)
        st.markdown("---")
        st.subheader("🔍 Data Preview")
        st.dataframe(df.head(20), use_container_width=True)
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("📋 Column Information")
            info_df = pd.DataFrame({"Data Type": df.dtypes.astype(str), "Non-Null Count": df.count(), "Missing %": df.isnull().mean() * 100})
            st.dataframe(info_df, use_container_width=True)
        with c2:
            st.subheader("📊 Statistical Summary")
            st.dataframe(df.describe(include='all'), use_container_width=True)

    # ================== SMART INSIGHTS (EXPANDED) ==================
    elif choice == "📑 Auto Insights":
        st.markdown("## 🧠 AI-Powered Data Insights")
        
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        date_cols = df.select_dtypes(include=['datetime']).columns.tolist()

        # Correlation Analysis
        st.markdown("""<div class="insight-box"><h3>🔗 Correlation Analysis</h3></div>""", unsafe_allow_html=True)
        if len(numeric_cols) > 1:
            corr = df[numeric_cols].corr()
            fig = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale='RdBu_r', title="Correlation Heatmap")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("ℹ️ Not enough numeric columns to generate a correlation heatmap.")
        
        # Distribution Analysis
        st.markdown("""<div class="insight-box"><h3>📊 Numeric Data Distribution</h3></div>""", unsafe_allow_html=True)
        if numeric_cols:
            for col in numeric_cols[:4]: # Show for first 4 numeric columns
                c1, c2 = st.columns(2)
                with c1:
                    fig_hist = px.histogram(df, x=col, title=f"Distribution of {col}")
                    st.plotly_chart(fig_hist, use_container_width=True)
                with c2:
                    fig_box = px.box(df, y=col, title=f"Box Plot of {col}")
                    st.plotly_chart(fig_box, use_container_width=True)
        else:
            st.info("ℹ️ No numeric columns found for distribution analysis.")

        # Categorical Analysis
        st.markdown("""<div class="insight-box"><h3>🏷️ Categorical Data Summary</h3></div>""", unsafe_allow_html=True)
        if categorical_cols:
            for col in categorical_cols[:2]: # Show for first 2 categorical columns
                top_categories = df[col].value_counts().nlargest(10)
                fig = px.bar(top_categories, x=top_categories.index, y=top_categories.values, title=f"Top 10 Categories in {col}", labels={'x': col, 'y': 'Count'})
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("ℹ️ No categorical columns found for summary.")
        
        # Time Series Analysis
        st.markdown("""<div class="insight-box"><h3>⏰ Time Series Analysis</h3></div>""", unsafe_allow_html=True)
        if date_cols:
            date_col = date_cols[0]
            trend_data = df.set_index(date_col).resample('D').size().rename('count')
            fig = px.line(trend_data, title=f"Daily Trend (based on {date_col})")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("ℹ️ No date/time columns found for trend analysis.")

        display_chat_interface(df, page_key="insights")

    # ================== CUSTOM VISUALIZATION (EXPANDED) ==================
    elif choice == "📈 Custom Visualization":
        st.markdown("## 🎨 Custom Chart Builder")
        
        all_cols = df.columns.tolist()
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            chart_type = st.selectbox("📊 Chart Type", [
                "Bar Chart", "Line Chart", "Scatter Plot", "Pie Chart", 
                "Histogram", "Box Plot", "Violin Plot", "Area Chart", "Density Heatmap"
            ])
        with col2:
            x_axis = st.selectbox("X-Axis", all_cols, key="custom_x")
        with col3:
            y_axis = st.selectbox("Y-Axis (optional)", [None] + all_cols, key="custom_y")

        with st.expander("🎨 Advanced Options"):
            c1, c2 = st.columns(2)
            color_col = c1.selectbox("Color by (optional)", [None] + all_cols)
            size_col = c2.selectbox("Size by (optional, for scatter)", [None] + numeric_cols)

        if st.button("🚀 Generate Chart", type="primary"):
            try:
                fig = None
                if chart_type == "Bar Chart":
                    fig = px.bar(df, x=x_axis, y=y_axis, title=f"Bar Chart: {x_axis} vs {y_axis}", color=color_col)
                elif chart_type == "Line Chart":
                     fig = px.line(df, x=x_axis, y=y_axis, title=f"Line Chart: {x_axis} vs {y_axis}", color=color_col)
                elif chart_type == "Scatter Plot":
                    fig = px.scatter(df, x=x_axis, y=y_axis, title=f"Scatter Plot: {x_axis} vs {y_axis}", color=color_col, size=size_col)
                elif chart_type == "Histogram":
                    fig = px.histogram(df, x=x_axis, title=f"Histogram of {x_axis}", color=color_col)
                elif chart_type == "Pie Chart":
                    fig = px.pie(df, names=x_axis, values=y_axis, title=f"Pie Chart of {x_axis}")
                elif chart_type == "Box Plot":
                    fig = px.box(df, x=x_axis, y=y_axis, title=f"Box Plot: {x_axis} vs {y_axis}", color=color_col)
                elif chart_type == "Violin Plot":
                    fig = px.violin(df, x=x_axis, y=y_axis, title=f"Violin Plot: {x_axis} vs {y_axis}", color=color_col)
                elif chart_type == "Area Chart":
                    fig = px.area(df, x=x_axis, y=y_axis, title=f"Area Chart: {x_axis} vs {y_axis}", color=color_col)
                elif chart_type == "Density Heatmap":
                    fig = px.density_heatmap(df, x=x_axis, y=y_axis, title=f"Density Heatmap: {x_axis} vs {y_axis}")

                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    st.success("✅ Chart generated successfully!")
                    display_chat_interface(df, page_key="custom")
                else:
                    st.warning("⚠️ Please select appropriate columns for this chart type.")
            except Exception as e:
                st.error(f"❌ Error generating chart: {e}")
    
    # ================== EXPORT REPORT ==================
    elif choice == "📤 Export Report":
        # (This section is unchanged and complete)
        st.markdown("## 📄 Export Your Analysis")
        st.subheader("🎯 Customize Report")
        
        col1, col2 = st.columns(2)
        with col1:
            include_summary = st.checkbox("📊 Data Summary", value=True)
            include_correlations = st.checkbox("🔗 Correlations", value=True)
        with col2:
            include_missing = st.checkbox("❌ Missing Values", value=True)
            include_distributions = st.checkbox("📈 Distributions", value=True)

        st.markdown("---")
        def create_pdf_report(df, include_summary, include_correlations, include_missing, include_distributions):
            buffer = BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=inch*0.8, leftMargin=inch*0.8, topMargin=inch*0.8, bottomMargin=inch*0.8)
            styles = getSampleStyleSheet()
            story = []
            story.append(Paragraph("📊 DataViz Pro - Analysis Report", styles["Title"]))
            story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]))
            story.append(Spacer(1, 20))
            story.append(Paragraph("📋 Dataset Overview", styles["Heading2"]))
            overview_data = [['Metric', 'Value'], ['Total Rows', f'{len(df):,}'], ['Total Columns', str(len(df.columns))]]
            overview_table = Table(overview_data, colWidths=[2*inch, 4*inch])
            overview_table.setStyle([('BACKGROUND', (0, 0), (-1, 0), '#667eea'), ('TEXTCOLOR', (0, 0), (-1, 0), 'white'), ('ALIGN', (0, 0), (-1, -1), 'LEFT'), ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'), ('FONTSIZE', (0, 0), (-1, 0), 12), ('BOTTOMPADDING', (0, 0), (-1, 0), 12), ('BACKGROUND', (0, 1), (-1, -1), '#f8f9fa'), ('GRID', (0, 0), (-1, -1), 1, '#cccccc')])
            story.append(overview_table)
            story.append(Spacer(1, 20))
            if include_correlations:
                story.append(Paragraph("🔗 Correlation Analysis", styles["Heading2"]))
                numeric_df = df.select_dtypes(include=[np.number])
                if len(numeric_df.columns) > 1:
                    try:
                        plt.figure(figsize=(8, 6))
                        sns.heatmap(numeric_df.corr(), annot=True, cmap='viridis', fmt='.2f')
                        plt.title('Correlation Heatmap', pad=20)
                        plt.tight_layout()
                        img_buffer = BytesIO()
                        plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
                        plt.close()
                        img_buffer.seek(0)
                        story.append(Image(img_buffer, width=6*inch, height=4.5*inch))
                    except Exception as e:
                        story.append(Paragraph(f"Could not generate heatmap: {e}", styles["Normal"]))
                story.append(Spacer(1, 20))
            doc.build(story)
            return buffer.getvalue()
        if st.button("📥 Generate & Download PDF Report", type="primary"):
            with st.spinner("Brewing your PDF report... ☕"):
                try:
                    pdf_bytes = create_pdf_report(df, include_summary, include_correlations, include_missing, include_distributions)
                    st.download_button(label="✅ Click to Download PDF", data=pdf_bytes, file_name="DataViz_Pro_Report.pdf", mime="application/pdf")
                except Exception as e:
                    st.error(f"❌ Failed to generate PDF: {e}")

else:
    st.markdown("""
    <div style="text-align: center; padding: 60px 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 15px; margin: 50px 0;">
        <div style="font-size: 4rem; margin-bottom: 20px;">📊</div>
        <h2>Welcome to DataViz Pro!</h2>
        <p style="font-size: 1.2rem; margin: 20px 0;">Upload your CSV file to unlock powerful data insights</p>
        <p>👈 Use the sidebar to get started!</p>
    </div>
    """, unsafe_allow_html=True)