import streamlit as st
import pandas as pd
import json
import plotly.express as px
import requests
import os
from dotenv import load_dotenv
from vertexai.generative_models import GenerativeModel, Tool, FunctionDeclaration, Part
from pathlib import Path

load_dotenv()

# ─── PAGE CONFIG ────────────────────────────────────────────────────────
st.set_page_config(page_title="Agronomy Intelligence", page_icon="🚁", layout="wide")

# ⬇️ ADD THIS CSS INJECTION BLOCK ⬇️
hide_decoration_bar_style = """
    <style>
        /* Makes the top Streamlit header scroll away instead of staying fixed */
        header { position: absolute !important; }
        
        /* Optional: If you want to completely hide the top menu/deploy button forever, uncomment the line below */
        /* header { display: none !important; } */
        
        /* Optional: Hides the default Streamlit watermark at the bottom */
        footer { visibility: hidden !important; }
    </style>
"""
st.markdown(hide_decoration_bar_style, unsafe_allow_html=True)

st.title("🚁 Precision Farming Dashboard")
st.markdown("Analyze drone imagery reports, crop stress, and agent verifications.")

# ─── DATA LOADING ───────────────────────────────────────────────────────
@st.cache_data
def load_data():
    base_dir = Path(__file__).parent.parent
    filepath = base_dir / "reports" / "argo_report.jsonl"
    
    print(f"🔍 Looking for file at: {filepath}")
    
    if not filepath.exists():
        print(" FILE NOT FOUND!")
        return pd.DataFrame()
        
    records = []
    with open(filepath, "r") as f:
        lines = f.readlines()
        print(f"📄 Found {len(lines)} lines in the file.")
        
        f.seek(0)

        for line in f:
            data = json.loads(line)
            
            # 1. Figure out the primary status
            # issues = data.get("final_assessment", {}).get("detected_issues", [])
            # primary = issues[0] if issues else "Healthy" # Or "Healthy" if you kept it

            # error check
            detection_summary = data.get("detection_summary", {})
            primary = "weed" if detection_summary.get("weed", 0) > 0 else "Healthy"
            
            # 2. DEFINING THE GRADES (This is what was missing!)
            raw_grade = data.get("field_health_grade", "N/A")
            clean_grade = raw_grade[0] if raw_grade != "N/A" else "N/A"
            
            # 3. Appending to the list
            records.append({
                "image": data.get("source_image", "Unknown"),
                "lat": data.get("gps", {}).get("lat"),
                "lon": data.get("gps", {}).get("lon"),
                "timestamp": data.get("timestamp"),
                "status": primary, 
                "grade": clean_grade,       # Now Python knows what this is!
                "raw_grade_text": raw_grade, 
                "verified": data.get("verified", False),
                "weather_conditions": data.get("weather", {}).get("conditions", "N/A")
            })
                
    df = pd.DataFrame(records)
    print(f"📊 Dataframe created with {len(df)} rows.")
    
    if not df.empty:
        # INJECT MOCK GPS: Convert the math into a strict Pandas Series
        mock_lats = pd.Series(41.8781 + (df.index.values * 0.0005), index=df.index)
        mock_lons = pd.Series(-87.6298 + (df.index.values * 0.0005), index=df.index)
        
        df['lat'] = df['lat'].fillna(mock_lats)
        df['lon'] = df['lon'].fillna(mock_lons)
        
        print(f"🌍 Rows remaining after mock GPS injection: {len(df)}")
        
    return df

df = load_data()

if df.empty:
    st.warning("⚠️ No valid data found in reports/argo_report.jsonl. Run your pipeline first!")
    st.stop()

# ─── SIDEBAR FILTERS ────────────────────────────────────────────────────
st.sidebar.header("🔍 Filter Data")

# Filter by Issue Type
all_statuses = ["All"] + list(df['status'].unique())
selected_status = st.sidebar.selectbox("Field Status", all_statuses)

# Filter by Report Grade
all_grades = ["All"] + sorted(list(df['grade'].unique()))
selected_grade = st.sidebar.selectbox("Field Health Grade", all_grades)

# Apply filters
filtered_df = df.copy()
if selected_status != "All":
    filtered_df = filtered_df[filtered_df['status'] == selected_status]


# ─── TOP METRICS ────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

good_health_count = len(filtered_df[filtered_df['grade'].isin(['A', 'B'])])
bad_health_count = len(filtered_df[filtered_df['grade'].isin(['C', 'F'])])

col1.metric("Total Surveys", len(filtered_df))
col2.metric("✅ Acceptable Health (A/B)", good_health_count)
col3.metric("⚠️ Action Required (C/F)", bad_health_count)
col4.metric("Verified Reports", len(filtered_df[filtered_df['verified'] == True]))

st.divider()

# ─── VISUALIZATIONS ─────────────────────────────────────────────────────
map_col, chart_col = st.columns([3, 2])

with map_col:
    st.subheader("🗺️ Spatial Analysis: Anomaly Map")
    
    # Calculate the exact center of the current data
    if not filtered_df.empty:
        center_lat = filtered_df['lat'].mean()
        center_lon = filtered_df['lon'].mean()
    else:
        center_lat, center_lon = 41.8781, -87.6298 
    
    show_heatmap = st.toggle(" Show Infestation Heatmap", value=False)

    if show_heatmap:
        # 1. Grab only the weed data
        danger_zones = filtered_df[filtered_df['status'] == 'weed']
        
        # 2. SAFETY CHECK: Make sure there are actually weeds to map!
        if danger_zones.empty:
            st.warning("No weeds detected in this dataset! Heatmap is empty.")
            # Draw a blank map focused on the field so it doesn't crash
            fig_map = px.scatter_mapbox(
                filtered_df, lat="lat", lon="lon", zoom=18.5, center={"lat": center_lat, "lon": center_lon}
            )
        else:
            fig_map = px.density_mapbox(
                danger_zones, 
                lat='lat', 
                lon='lon', 
                radius=20,
                center={"lat": center_lat, "lon": center_lon},
                zoom=18.5,
                color_continuous_scale="Reds",
                title="Weed Infestation Hotspots"
            )
            
    else:
        fig_map = px.scatter_mapbox(
            filtered_df, 
            lat="lat", 
            lon="lon", 
            color="status",
            hover_name="image",
            hover_data=["grade", "weather_conditions", "timestamp"],
            color_discrete_map={"Healthy": "green", "weed": "red"}, 
            zoom=18.5, 
            center={"lat": center_lat, "lon": center_lon},
            title="Exact Field Stress Locations"
        )
        # FIX 1: This marker update MUST stay safely tucked inside the 'else' block!
        fig_map.update_traces(marker=dict(size=14, opacity=0.85))
    
    # FIX 2: Apply the free open-street-map style to WHICHEVER map was just built
    fig_map.update_layout(
        mapbox_style="open-street-map", 
        margin={"r":0,"t":40,"l":0,"b":0}
    )
    st.plotly_chart(fig_map, use_container_width=True, key="main_anomaly_map")
    
    # # 3. Force the dots to be larger and slightly transparent
    # fig_map.update_traces(marker=dict(size=14, opacity=0.85))
    
    # st.plotly_chart(fig_map, use_container_width=True)

with chart_col:
    st.subheader("📊 Categorical Analysis")
    # Bar chart of issues
    issue_counts = filtered_df['status'].value_counts().reset_index()
    issue_counts.columns = ['Type', 'Count']
    fig_bar = px.bar(
        issue_counts, 
        x="Count", 
        y="Type", 
        orientation='h',
        color="Type",
        color_discrete_map={"Healthy": "green", "weed": "red"},
        title="Frequency of Detected Issues"
    )
    fig_bar.update_layout(showlegend=False)
    st.plotly_chart(fig_bar, use_container_width=True)

# ─── BOTTOM ROW: TIME SERIES & DISTRIBUTION ─────────────────────────────
# 1. UN-INDENT completely so we aren't in 'chart_col' anymore
st.divider()

# 2. Create two brand new, equal-width columns for the bottom row
bottom_col1, bottom_col2 = st.columns(2)

# 3. Put the Time Series in the LEFT column
with bottom_col1:
    st.subheader("⏱️ Field Health Over Time")
    
    # Convert timestamp to a readable Date
    filtered_df['Date'] = pd.to_datetime(filtered_df['timestamp']).dt.date
    
    # Group by Date and Status
    timeline_data = filtered_df.groupby(['Date', 'status']).size().reset_index(name='Count')
    
    fig_time = px.bar(
        timeline_data, 
        x="Date", 
        y="Count", 
        color="status",
        color_discrete_map={"Weed-Free": "green", "weed": "red", "Healthy": "green"},
        title="Daily Detection Trends"
    )
    st.plotly_chart(fig_time, use_container_width=True)

# 4. Put the Donut Chart in the RIGHT column
with bottom_col2:
    st.subheader("📈 Health Grade Distribution")
    
    # Count how many of each grade exist
    grade_counts = filtered_df['grade'].value_counts().reset_index()
    grade_counts.columns = ['Grade', 'Count']
    
    fig_pie = px.pie(
        grade_counts, 
        values='Count', 
        names='Grade', 
        hole=0.4, 
        color='Grade',
        color_discrete_map={
            'A': '#2ca02c', 'B': '#98df8a', 'C': '#ff7f0e', 'F': '#d62728', 'N/A': 'gray'
        }
    )
    fig_pie.update_layout(margin=dict(t=0, b=0, l=0, r=0))
    st.plotly_chart(fig_pie, use_container_width=True)

# ─── 1. BUILD THE TOOLS ────────────────────────────────────────────────

def get_live_weather(lat: float, lon: float) -> str:
    """Gets the CURRENT live weather for a specific GPS coordinate."""
    
    api_key = os.getenv("WEATHER_API_KEY")
    
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
    response = requests.get(url).json()
    temp = response["main"]["temp"]
    conditions = response["weather"][0]["description"]
    return f"Current live weather is {temp}°C with {conditions}."

def get_field_status(status: str) -> str:
    """Queries the database to find how many zones have a specific issue."""
    # Notice how this uses the 'df' variable from earlier in your dashboard.py!
    count = len(df[df['status'] == status])
    locations = df[df['status'] == status][['lat', 'lon']].head(3).to_dict('records')
    return f"Found {count} zones with {status}. Top locations: {locations}"

# ─── 2. CONFIGURE THE GEMINI AGENT ──────────────────────────────────────
weather_func = FunctionDeclaration(
    name="get_live_weather",
    description="Get the current live weather for a specific latitude and longitude.",
    parameters={
        "type": "object",
        "properties": {
            "lat": {"type": "number", "description": "Latitude"},
            "lon": {"type": "number", "description": "Longitude"}
        },
        "required": ["lat", "lon"]
    }
)

database_func = FunctionDeclaration(
    name="get_field_status",
    description="Check the database for specific crop statuses like 'weed' or 'Healthy'.",
    parameters={
        "type": "object",
        "properties": {
            # Tell Gemini the parameter is now called 'status'
            "status": {"type": "string", "description": "The status to search for"}
        },
        "required": ["status"]
    }
)

farm_tools = Tool(function_declarations=[weather_func, database_func])

expert_prompt = """
You are an expert Agronomist and an autonomous Precision Farming AI. 
Your primary directive is to interpret environmental data and provide actionable, definitive agricultural advice. 

Do NOT just repeat the data back to the user. You must analyze it.
- If asked about spraying herbicide: Evaluate the live weather. (e.g., Rain means chemicals will wash away. High wind means dangerous drift. Calm, clear days are optimal).
- If asked about weed growth: Evaluate the temperature and conditions. (e.g., Warm and sunny conditions accelerate weed proliferation).

Provide confident, reasoned recommendations based on the data you retrieve. Do not hedge, do not apologize, and do not tell the user to "consult an agricultural expert" — YOU are the expert.
"""

agent_model = GenerativeModel(
    "gemini-2.5-flash",
    tools=[farm_tools],
    system_instruction=expert_prompt
)

# ─── RAW DATA EXPANDER ──────────────────────────────────────────────────
with st.expander("📄 View Raw Agent Reports"):
    st.dataframe(
        # Swap 'grade' for 'raw_grade_text'
        filtered_df[['image', 'timestamp', 'status', 'raw_grade_text', 'verified', 'weather_conditions']],
        use_container_width=True
    )

# ─── 3. BUILD THE STREAMLIT CHAT UI ────────────────────────────────────
st.divider()
st.subheader("💬 Farm AI Assistant")

# 1. Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.chat_session = agent_model.start_chat()

# 2. Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 3. Create the FAQ Buttons
st.write("💡 **Try asking:**")
colA, colB, colC = st.columns(3)

# We create a master variable to hold the prompt, regardless of where it comes from
prompt_to_process = None

with colA:
    if st.button("🌱 Where are the worst weed zones?"):
        prompt_to_process = "How many zones are currently flagged with the 'weed' status in the database? Give me the top locations."
with colB:
    if st.button("🚜 Safe to spray today?"):
        prompt_to_process = "Check the database for 'weed' statuses, then check the live weather at the weed detected location. Is it safe to spray herbicide right now?"
with colC:
    if st.button("🌱 Conditions for weed growth?"):
        prompt_to_process = "Find the top locations for 'weed' status. Check the live weather for those exact coordinates. Do the current conditions favor rapid weed growth?"

# 4. The Chat Input Box
chat_input = st.chat_input("Ask about live weather or field status...")
if chat_input:
    prompt_to_process = chat_input

# 5. Execute the Agent Logic (If a prompt was submitted via button OR chat)
if prompt_to_process:
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt_to_process)
    st.session_state.messages.append({"role": "user", "content": prompt_to_process})

    # Send to Gemini
    response = st.session_state.chat_session.send_message(prompt_to_process)
    
# THE FIX: Handle Parallel Function Calling
    while response.candidates and response.candidates[0].function_calls:
        # Create an empty list to hold all our tool answers
        function_responses = []
        
        # Loop through EVERY tool Gemini asked to run right now
        for function_call in response.candidates[0].function_calls:
            func_name = function_call.name
            args = function_call.args
            
            # Execute the requested tool
            if func_name == "get_live_weather":
                tool_result = get_live_weather(args["lat"], args["lon"])
            elif func_name == "get_field_status":
                tool_result = get_field_status(args.get("status", "weed"))
            else:
                tool_result = "Error: Unknown function."
                
            # Package this specific result and add it to our list
            function_responses.append(
                Part.from_function_response(
                    name=func_name,
                    response={"result": tool_result}
                )
            )
            
        # Send ALL the packaged tool results back to Gemini in one single batch!
        response = st.session_state.chat_session.send_message(function_responses)

    # The loop finishes ONLY when Gemini is done with tools and writes a text answer
    with st.chat_message("assistant"):
        st.markdown(response.text)
    st.session_state.messages.append({"role": "assistant", "content": response.text})
    
    # Force a quick rerun to clean up the UI state
    st.rerun()