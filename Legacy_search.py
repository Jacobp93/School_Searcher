import streamlit as st
import pandas as pd
from geopy.geocoders import OpenCage
from sklearn.neighbors import NearestNeighbors
import numpy as np

# OpenCage API Key (from Streamlit secrets)
OPENCAGE_API_KEY = st.secrets["OPENCAGE_API_KEY"]

# Initialize OpenCage Geocoder
geolocator = OpenCage(api_key=OPENCAGE_API_KEY)

# Function to get latitude and longitude using OpenCage
def get_lat_lon(postcode):
    try:
        location = geolocator.geocode(postcode)
        if location:
            return location.latitude, location.longitude
        else:
            return None, None
    except Exception as e:
        st.error(f"Error with geocoding: {str(e)}")
        return None, None

# Load datasets
@st.cache_data
def load_data():
    primary_legacy = pd.read_csv("Primary_Legacy_with_Closest_SaaS.csv")
    re_legacy = pd.read_csv("RE_Legacy_with_Closest_SaaS.csv")

    # Replace 'Blank' strings in the latitude and longitude with NaN and drop them
    primary_legacy = primary_legacy.replace('Blank', np.nan).dropna(subset=['latitude', 'longitude'])
    re_legacy = re_legacy.replace('Blank', np.nan).dropna(subset=['latitude', 'longitude'])

    return primary_legacy, re_legacy

primary_legacy, re_legacy = load_data()

# Function to check if a row contains SaaS in the Customer Type columns
def is_saas_customer(row):
    # Ensure the customer is classified as SaaS in either Primary or RE
    if row['Customer type - Primary'] == 'Primary SaaS' or row['Customer type - RE'] == 'Jigsaw RE SaaS':
        return True
    return False

# Helper function to find the nearest SaaS schools
def find_nearest(postcode, data, n_neighbors=5, radius=50):
    # Geocoding the input postcode to get latitude and longitude
    lat_lon = get_lat_lon(postcode)
    if lat_lon is None:
        return None, None
    
    coords = np.array([lat_lon])  # Convert to array format

    # Filter out only SaaS customers based on 'Customer type - Primary' and 'Customer type - RE'
    saas_data = data[data.apply(is_saas_customer, axis=1)]
    
    if saas_data.empty:
        return None, None
    
    # Use NearestNeighbors to find the closest schools based on lat/long
    nbrs = NearestNeighbors(n_neighbors=n_neighbors * 2, algorithm='ball_tree', metric='haversine')  # Increase neighbors to reduce chance of missing distinct results
    school_coords = saas_data[['latitude', 'longitude']].values.astype(float)  # Ensure valid coordinates
    nbrs.fit(np.radians(school_coords))

    distances, indices = nbrs.kneighbors(np.radians(coords))
    distances_in_miles = distances.flatten() * 6371 * 0.621371  # Convert to miles

    # Filter schools within the specified radius
    within_radius = distances_in_miles <= radius

    nearest_schools = saas_data.iloc[indices.flatten()[within_radius]]
    distances_in_miles = distances_in_miles[within_radius]

    # Limit the number of schools to 5
    nearest_schools = nearest_schools.head(n_neighbors)
    distances_in_miles = distances_in_miles[:len(nearest_schools)]  # Adjust distances after limiting results

    return nearest_schools, distances_in_miles

# Streamlit app layout
st.title("School Nearest Neighbor Finder")
st.write("Search for the top 5 closest SaaS schools by entering a postcode.")

# Postcode input
postcode = st.text_input("Enter a postcode:", "")

# Set a search radius
radius = st.slider("Set Search Radius (in miles)", min_value=1, max_value=50, value=10)  # Set default to 10 miles

# Find the nearest SaaS schools based on user input
if st.button("Search"):
    if postcode:
        # Combine the data from Primary Legacy and RE Legacy
        combined_data = pd.concat([primary_legacy, re_legacy])

        if not combined_data.empty:
            nearest_schools, distances = find_nearest(postcode, combined_data, n_neighbors=5, radius=radius)

            # Check if the nearest schools and distances are found
            if nearest_schools is not None and distances is not None:
                st.write(f"Top 5 closest SaaS schools within {radius} miles:")
                
                # Prepare data for table display
                table_data = []
                for i, (index, row) in enumerate(nearest_schools.iterrows()):
                    school_name = row['Company name']  # Assuming 'Company name' is the column with the school name
                    # Get all unique types for this school
                    school_types = ', '.join([row[f'School {j} Type'] for j in range(1, 6) if pd.notna(row[f'School {j} Type'])])
                    
                    table_data.append({
                        "School Name": school_name,
                        "Distance (miles)": f"{distances[i]:.2f}",
                        "School Type": school_types
                    })
                
                # Convert the list of dictionaries to a DataFrame for display
                table_df = pd.DataFrame(table_data)
                
                # Display the results as a table
                st.table(table_df)
                
            else:
                st.error("No SaaS schools found for the provided postcode within the specified radius.")
        else:
            st.error("No schools found for the selected criteria.")
