import streamlit as st
import pandas as pd
from geopy.geocoders import OpenCage
from sklearn.neighbors import NearestNeighbors
import numpy as np

# OpenCage API Key
OPENCAGE_API_KEY = "YOUR_OPENCAGE_API_KEY"  # Replace with your actual OpenCage API key


OPENCAGE_API_KEY = st.secrets["OPENCAGE_API_KEY"]

geolocator = OpenCage(api_key=OPENCAGE_API_KEY)



# Function to get latitude and longitude
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
    primary_legacy = pd.read_csv(r"C:\Users\Jacob\OneDrive - Jigsaw PSHE Ltd\Documents\Python\Neighbour_Analysis\Primary_Legacy_with_Closest_SaaS.csv")
    re_legacy = pd.read_csv(r"C:\Users\Jacob\OneDrive - Jigsaw PSHE Ltd\Documents\Python\Neighbour_Analysis\RE_Legacy_with_Closest_SaaS.csv")

    # Replace 'Blank' strings in the latitude and longitude with NaN and drop them
    primary_legacy = primary_legacy.replace('Blank', np.nan).dropna(subset=['latitude', 'longitude'])
    re_legacy = re_legacy.replace('Blank', np.nan).dropna(subset=['latitude', 'longitude'])

    return primary_legacy, re_legacy

primary_legacy, re_legacy = load_data()

# Helper function to find the nearest neighbors
def find_nearest(postcode, data, n_neighbors=5, radius=10):
    # Geocoding the input postcode to get latitude and longitude
    lat_lon = get_lat_lon(postcode)
    if lat_lon is None:
        return None, None
    
    coords = np.array([lat_lon])  # Convert to array format

    # Use NearestNeighbors to find the closest schools based on lat/long
    nbrs = NearestNeighbors(n_neighbors=n_neighbors * 2, algorithm='ball_tree', metric='haversine')  # Increase neighbors to reduce chance of missing distinct results
    school_coords = data[['latitude', 'longitude']].values.astype(float)  # Ensure valid coordinates
    nbrs.fit(np.radians(school_coords))

    distances, indices = nbrs.kneighbors(np.radians(coords))
    distances_in_miles = distances.flatten() * 6371 * 0.621371  # Convert to miles

    # Filter schools within the specified radius
    within_radius = distances_in_miles <= radius

    nearest_schools = data.iloc[indices.flatten()[within_radius]]
    distances_in_miles = distances_in_miles[within_radius]

    # Limit the number of schools to 5 to match School 1 to School 5 Type columns
    nearest_schools = nearest_schools.head(n_neighbors)
    distances_in_miles = distances_in_miles[:len(nearest_schools)]  # Adjust distances after limiting results

    return nearest_schools, distances_in_miles

# Streamlit app layout
st.title("School Nearest Neighbor Finder")
st.write("Search for the top 5 closest schools by entering a postcode.")

# Postcode input
postcode = st.text_input("Enter a postcode:", "")

# Checkbox for selecting Primary Legacy or Jigsaw RE
search_primary_legacy = st.checkbox("Search Primary Legacy", value=True)
search_jigsaw_re = st.checkbox("Search Jigsaw RE", value=False)

# Set a search radius
radius = st.slider("Set Search Radius (in miles)", min_value=1, max_value=50, value=10)  # Set default to 10 miles

# Find the nearest neighbors based on user selection
if st.button("Search"):
    if postcode:
        combined_data = pd.DataFrame()  # Initialize an empty DataFrame to combine datasets

        # Combine the datasets based on user selection
        if search_primary_legacy:
            st.subheader("Searching in Primary Legacy dataset...")
            combined_data = pd.concat([combined_data, primary_legacy])

        if search_jigsaw_re:
            st.subheader("Searching in Jigsaw RE dataset...")
            combined_data = pd.concat([combined_data, re_legacy])

        if not combined_data.empty:
            nearest_schools, distances = find_nearest(postcode, combined_data, n_neighbors=5, radius=radius)  # Ensure max 5 schools
            
            # Check if the nearest schools and distances are found
            if nearest_schools is not None and distances is not None:
                st.write(f"Top 5 closest schools within {radius} miles:")
                
                # Prepare data for table display
                table_data = []
                for i, (index, row) in enumerate(nearest_schools.iterrows()):
                    school_name = row[f'School {i+1}']  # Dynamic school name column
                    # Get all unique types for this school
                    school_types = get_school_types(row)
                    
                    table_data.append({
                        "School Name": school_name,
                        "Distance (miles)": f"{distances[i]:.2f}",
                        "School Type": school_types  # Combined unique types
                    })
                
                # Convert the list of dictionaries to a DataFrame for display
                table_df = pd.DataFrame(table_data)
                
                # Display the results as a table
                st.table(table_df)
                
            else:
                st.error("Postcode not found, please try again.")
        else:
            st.error("No schools found for the selected criteria.")
