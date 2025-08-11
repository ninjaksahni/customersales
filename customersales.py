import streamlit as st
import pandas as pd
import plotly.express as px
import time

st.set_page_config(page_title="Amazon Shipment Sales Explorer", layout="wide")

REQUIRED_COLS = ['Shipment To City', 'Shipment To State', 'FC', 'Merchant SKU']

@st.cache_data
def load_csv(uploaded_file) -> pd.DataFrame:
    return pd.read_csv(uploaded_file)

# A local lookup for common Indian cities (lat, lon). Keys are lowercase city names.
COMMON_CITY_COORDS = {
    'mumbai': (19.0760, 72.8777),
    'bombay': (19.0760, 72.8777),
    'delhi': (28.6139, 77.2090),
    'new delhi': (28.6139, 77.2090),
    'bangalore': (12.9716, 77.5946),
    'bengaluru': (12.9716, 77.5946),
    'kolkata': (22.5726, 88.3639),
    'calcutta': (22.5726, 88.3639),
    'chennai': (13.0827, 80.2707),
    'madras': (13.0827, 80.2707),
    'hyderabad': (17.3850, 78.4867),
    'pune': (18.5204, 73.8567),
    'ahmedabad': (23.0225, 72.5714),
    'surat': (21.1702, 72.8311),
    'jaipur': (26.9124, 75.7873),
    'lucknow': (26.8467, 80.9462),
    'kanpur': (26.4499, 80.3319),
    'nagpur': (21.1458, 79.0882),
    'indore': (22.7196, 75.8577),
    'bhopal': (23.2599, 77.4126),
    'patna': (25.5941, 85.1376),
    'vadodara': (22.3072, 73.1812),
    'ludhiana': (30.9000, 75.8573),
    'agra': (27.1767, 78.0081),
    'nashik': (19.9975, 73.7898),
    'faridabad': (28.4089, 77.3178),
    'meerut': (28.9845, 77.7064),
    'rajkot': (22.3039, 70.8022),
    'kalyan': (19.2403, 73.1305),
    'varanasi': (25.3176, 82.9739),
    'srinagar': (34.0837, 74.7973),
    'amritsar': (31.6340, 74.8723),
    'prayagraj': (25.4358, 81.8463),
    'allahabad': (25.4358, 81.8463),
    'jodhpur': (26.2389, 73.0243),
    'madurai': (9.9252, 78.1198),
    'jalandhar': (31.3260, 75.5762),
    'howrah': (22.5958, 88.2636),
    'dhanbad': (23.7957, 86.4304),
    'ranchi': (23.3441, 85.3096),
    'thane': (19.2183, 72.9781),
    'aligarh': (27.8926, 78.0880),
    'gwalior': (26.2183, 78.1828),
    'bhubaneswar': (20.2961, 85.8245),
    'moradabad': (28.8386, 78.7734),
    'jamshedpur': (22.8046, 86.2029),
    'noida': (28.5355, 77.3910),
    'ghaziabad': (28.6692, 77.4538),
    'udaipur': (24.5854, 73.7125),
    'trichy': (10.7905, 78.7047),
    'tiruchirappalli': (10.7905, 78.7047),
    'kochi': (9.9312, 76.2673),
    'cochin': (9.9312, 76.2673),
    'calicut': (11.2588, 75.7804),
    'mangalore': (12.9141, 74.8560),
    'visakhapatnam': (17.6868, 83.2185),
    'visakhapatnam/visakhapatnam': (17.6868, 83.2185)
}

@st.cache_data
def geocode_with_nominatim(query: str):
    """Try to geocode using geopy.Nominatim. Returns (lat, lon) or (None, error_string).
    If geopy is not installed or internet is not available, returns (None, message).
    """
    try:
        from geopy.geocoders import Nominatim
    except Exception as e:
        return None, "geopy not installed"

    try:
        geolocator = Nominatim(user_agent="amazon_shipment_mapper")
        location = geolocator.geocode(query, timeout=10)
        if location:
            return (location.latitude, location.longitude), None
        else:
            return None, "not found"
    except Exception as e:
        return None, str(e)


@st.cache_data
def get_city_coordinates(city: str, state: str = None):
    """Return (lat, lon, source) where source is 'local' or 'nominatim' or 'none'."""
    if not city or city.strip() == '':
        return None, None, 'none'
    key = city.strip().lower()

    # Fast local lookup
    if key in COMMON_CITY_COORDS:
        lat, lon = COMMON_CITY_COORDS[key]
        return lat, lon, 'local'

    # Try variants (without punctuation)
    alt = key.replace('.', '').replace("'", '').strip()
    if alt in COMMON_CITY_COORDS:
        lat, lon = COMMON_CITY_COORDS[alt]
        return lat, lon, 'local'

    # Try combined city + state key
    if state:
        combined = f"{key}, {state.strip().lower()}"
        if combined in COMMON_CITY_COORDS:
            lat, lon = COMMON_CITY_COORDS[combined]
            return lat, lon, 'local'

    # Fallback to Nominatim (OpenStreetMap) geocoding
    # Build sensible queries.
    queries = []
    if state and str(state).strip().lower() not in ("nan", "none", ""):
        queries.append(f"{city}, {state}, India")
    queries.append(f"{city}, India")

    for q in queries:
        coords, err = geocode_with_nominatim(q)
        # Small delay to be polite to the geocoding service
        time.sleep(1)
        if coords:
            lat, lon = coords
            return lat, lon, 'nominatim'

    return None, None, 'none'


# ----------------------------
# App UI
# ----------------------------

st.title("Amazon — Customer Shipment Sales Explorer")

st.markdown(
    """
**Important:** Go to your Seller Central → Reports → Shipment Sales. 
Download the *Customer Shipment Sales* CSV for the **last 30 days** and then upload it here.

[Open the Amazon SellerCentral report page (SHIPMENT_SALES).](https://sellercentral.amazon.in/reportcentral/SHIPMENT_SALES/1)
""",
    unsafe_allow_html=True,
)

uploaded = st.file_uploader("Upload your Customer Shipment Sales CSV (30-day)", type=["csv"])
if uploaded is None:
    st.info("Waiting for a CSV upload. Please download the 30-day report from the link above and upload it here.")
    st.stop()

# Load and validate
try:
    df = load_csv(uploaded)
except Exception as e:
    st.error(f"Failed to read CSV: {e}")
    st.stop()

st.write("### Raw file preview")
st.dataframe(df.head())

missing = [c for c in REQUIRED_COLS if c not in df.columns]
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

# Clean
for c in REQUIRED_COLS:
    df[c] = df[c].astype(str).str.strip()

# each row = one shipment (assumption)
df['_count'] = 1

# Sidebar Controls
st.sidebar.header("Filters & Map options")
all_skus = sorted(df['Merchant SKU'].dropna().unique())
all_fcs = sorted(df['FC'].dropna().unique())

selected_sku = st.sidebar.selectbox("Filter by SKU (optional)", options=["All"] + all_skus)
selected_fc = st.sidebar.selectbox("Filter by FC (optional)", options=["All"] + all_fcs)

top_n_map = st.sidebar.slider("Top N cities to show on map", min_value=5, max_value=50, value=10, step=1)

filtered_df = df.copy()
if selected_sku != "All":
    filtered_df = filtered_df[filtered_df['Merchant SKU'] == selected_sku]
if selected_fc != "All":
    filtered_df = filtered_df[filtered_df['FC'] == selected_fc]

# ----------------------------
# Tables
# ----------------------------
st.subheader("Aggregated Tables")

if selected_sku != "All":
    st.markdown(f"**Analysis for SKU:** {selected_sku}")
    col1, col2, col3 = st.columns(3)
    with col1:
        t = filtered_df.groupby('Shipment To City')['_count'].sum().reset_index().rename(columns={'_count': 'Count'}).sort_values('Count', ascending=False)
        st.write("SKU → By City (top)")
        st.dataframe(t)
        st.download_button("Download SKU→City CSV", t.to_csv(index=False).encode('utf-8'), file_name=f"{selected_sku}_by_city.csv")
    with col2:
        s = filtered_df.groupby('Shipment To State')['_count'].sum().reset_index().rename(columns={'_count': 'Count'}).sort_values('Count', ascending=False)
        st.write("SKU → By State (top)")
        st.dataframe(s)
        st.download_button("Download SKU→State CSV", s.to_csv(index=False).encode('utf-8'), file_name=f"{selected_sku}_by_state.csv")
    with col3:
        f = filtered_df.groupby('FC')['_count'].sum().reset_index().rename(columns={'_count': 'Count'}).sort_values('Count', ascending=False)
        st.write("SKU → By FC (top)")
        st.dataframe(f)
        st.download_button("Download SKU→FC CSV", f.to_csv(index=False).encode('utf-8'), file_name=f"{selected_sku}_by_fc.csv")

elif selected_fc != "All":
    st.markdown(f"**Analysis for FC:** {selected_fc}")
    col1, col2 = st.columns(2)
    with col1:
        t = filtered_df.groupby('Shipment To City')['_count'].sum().reset_index().rename(columns={'_count': 'Count'}).sort_values('Count', ascending=False)
        st.write("FC → By City (top)")
        st.dataframe(t)
        st.download_button("Download FC→City CSV", t.to_csv(index=False).encode('utf-8'), file_name=f"{selected_fc}_by_city.csv")
    with col2:
        s = filtered_df.groupby('Merchant SKU')['_count'].sum().reset_index().rename(columns={'_count': 'Count'}).sort_values('Count', ascending=False)
        st.write("FC → By SKU (top)")
        st.dataframe(s)
        st.download_button("Download FC→SKU CSV", s.to_csv(index=False).encode('utf-8'), file_name=f"{selected_fc}_by_sku.csv")

else:
    col1, col2, col3 = st.columns(3)
    with col1:
        t = df.groupby(['Merchant SKU', 'Shipment To City'])['_count'].sum().reset_index().rename(columns={'_count': 'Count'}).sort_values('Count', ascending=False)
        st.write("Overall SKU → By City (top)")
        st.dataframe(t)
        st.download_button("Download Overall SKU→City CSV", t.to_csv(index=False).encode('utf-8'), file_name="overall_sku_by_city.csv")
    with col2:
        s = df.groupby(['Merchant SKU', 'Shipment To State'])['_count'].sum().reset_index().rename(columns={'_count': 'Count'}).sort_values('Count', ascending=False)
        st.write("Overall SKU → By State (top)")
        st.dataframe(s)
        st.download_button("Download Overall SKU→State CSV", s.to_csv(index=False).encode('utf-8'), file_name="overall_sku_by_state.csv")
    with col3:
        f = df.groupby(['FC', 'Shipment To City'])['_count'].sum().reset_index().rename(columns={'_count': 'Count'}).sort_values('Count', ascending=False)
        st.write("Overall FC → By City (top)")
        st.dataframe(f)
        st.download_button("Download Overall FC→City CSV", f.to_csv(index=False).encode('utf-8'), file_name="overall_fc_by_city.csv")

# ----------------------------
# Map: Top N cities in India
# ----------------------------
st.subheader("Top Cities in India by Sales — Map")

# Determine the most likely state for each city (helps geocoding ambiguity)
city_state_agg = filtered_df.groupby(['Shipment To City', 'Shipment To State'])['_count'].sum().reset_index()
city_totals = city_state_agg.groupby('Shipment To City')['_count'].sum().reset_index().rename(columns={'_count': 'Count'})
city_totals = city_totals.sort_values('Count', ascending=False).head(top_n_map)

# For each top city, find the state with the max shipments (disambiguation)
rows = []
progress = st.progress(0)
n = len(city_totals)
for i, (_, r) in enumerate(city_totals.iterrows()):
    city = r['Shipment To City']
    count = int(r['Count'])
    state_rows = city_state_agg[city_state_agg['Shipment To City'] == city]
    # pick the state where shipments are highest for that city
    best_state = state_rows.sort_values('_count', ascending=False).iloc[0]['Shipment To State']

    lat, lon, source = get_city_coordinates(city, best_state)
    rows.append({'Shipment To City': city, 'Shipment To State': best_state, 'Count': count, 'lat': lat, 'lon': lon, 'source': source})
    progress.progress((i + 1) / max(n, 1))

progress.empty()
map_df = pd.DataFrame(rows)

# If any lat/lon missing, show a warning and list them
missing_coords = map_df[map_df['lat'].isnull() | map_df['lon'].isnull()]
if not missing_coords.empty:
    st.warning("Could not determine coordinates for these cities (please check spelling or install geopy for online geocoding):")
    st.dataframe(missing_coords[['Shipment To City', 'Shipment To State', 'Count']])

# Drop rows without coords for plotting
plot_df = map_df.dropna(subset=['lat', 'lon']).copy()
if not plot_df.empty:
    fig = px.scatter_mapbox(
        plot_df,
        lat='lat',
        lon='lon',
        size='Count',
        hover_name='Shipment To City',
        hover_data=['Shipment To State', 'Count', 'source'],
        zoom=4,
        center={'lat': 22.0, 'lon': 78.0},
        size_max=40,
        mapbox_style='open-street-map',
        title=f"Top {len(plot_df)} cities by shipment count"
    )
    st.plotly_chart(fig, use_container_width=True)
    st.download_button("Download top-cities-with-coords.csv", plot_df.to_csv(index=False).encode('utf-8'), file_name='top_cities_coords.csv')
else:
    st.error("No city coordinates available to display on the map. Consider installing geopy (`pip install geopy`) and retrying.")


