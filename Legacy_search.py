import streamlit as st
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import numpy as np
from geopy.geocoders import OpenCage  # Use OpenCage instead of Nominatim

# OpenCage API Key from Streamlit Secrets
OPENCAGE_API_KEY = st.secrets["OPENCAGE_API_KEY"]  # Make sure your API key is added in Streamlit secrets

# Initialize OpenCage Geocoder
geolocator = OpenCage(api_key=OPENCAGE_API_KEY)

# Load datasets from the specified directory
@st.cache_data
def load_data():
    primary_legacy = pd.read_csv("Primary_Legacy_with_Closest_SaaS.csv")
    re_legacy = pd.read_csv("RE_Legacy_with_Closest_SaaS.csv")

    
    # Replace 'Blank' strings in the latitude and longitude with NaN and drop them
    primary_legacy = primary_legacy.replace('Blank', np.nan).dropna(subset=['latitude', 'longitude'])
    re_legacy = re_legacy.replace('Blank', np.nan).dropna(subset=['latitude', 'longitude'])
    
    return primary_legacy, re_legacy

primary_legacy, re_legacy = load_data()

# Helper function to find the nearest neighbors using OpenCage
def find_nearest(postcode, data, n_neighbors=5, radius=10):
    # Geocoding the input postcode to get latitude and longitude using OpenCage
    location = geolocator.geocode(postcode)
    if location is None:
        return None, None
    
    coords = np.array([[location.latitude, location.longitude]])
    
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

# Helper function to get school types from the 'School 1 Type', 'School 2 Type', etc. columns
def get_school_types(row):
    school_types = set()  # Use a set to avoid duplicates
    
    # Loop through School 1 Type to School 5 Type to collect types
    for j in range(1, 6):
        school_type_column = f'School {j} Type'
        if school_type_column in row and not pd.isna(row[school_type_column]):
            school_types.add(row[school_type_column])  # Add to set to avoid duplicates
    
    return ', '.join(school_types) if school_types else 'Unknown'  # Return types as a comma-separated string

# Helper function to group schools by name and merge customer types
def group_schools_by_name(nearest_schools, distances):
    grouped_data = {}
    
    for i, (index, row) in enumerate(nearest_schools.iterrows()):
        school_name = row[f'School {i+1}']  # Dynamic school name column
        
        # If the school already exists in the grouped data, update its information
        if school_name in grouped_data:
            # Update the customer types (merge them)
            grouped_data[school_name]['School Type'] = ', '.join(set(grouped_data[school_name]['School Type'].split(', ') + get_school_types(row).split(', ')))
            # Keep the shortest distance
            grouped_data[school_name]['Distance (miles)'] = min(grouped_data[school_name]['Distance (miles)'], distances[i])
        else:
            # If the school doesn't exist, add it
            grouped_data[school_name] = {
                "School Name": school_name,
                "Distance (miles)": distances[i],
                "School Type": get_school_types(row)
            }
    
    # Convert grouped data back into a DataFrame
    return pd.DataFrame(grouped_data.values())

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
            
            # Group schools by name and merge customer types
            grouped_schools = group_schools_by_name(nearest_schools, distances)
            
            # Check if grouped schools and distances are found
            if not grouped_schools.empty:
                st.write(f"Top 5 closest schools within {radius} miles:")
                st.table(grouped_schools)  # Display the results as a table
            else:
                st.error("No schools found within the specified radius.")
        else:
            st.error("No schools found for the selected criteria.")
