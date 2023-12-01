import folium
import location
import numpy as np


def plot_cities_with_folium(city_df, adj_graph, lat_min, lat_max, lon_min, lon_max, add_city_labels=True, add_dist_labels=True):
    # Calculate the center of the map
    center_lat = (lat_min + lat_max) / 2
    center_lon = (lon_min + lon_max) / 2

    # Create a map centered around the calculated center
    m = folium.Map(location=[center_lat, center_lon], zoom_start=13)

    # Filter cities within the specified lat-long window
    filtered_cities = city_df[(city_df['lat'] >= lat_min) & (city_df['lat'] <= lat_max) &
                              (city_df['long'] >= lon_min) & (city_df['long'] <= lon_max)]

    # Extract indices of filtered cities
    indices = filtered_cities['city_idx'].values

    # Extract relevant part of the adjacency matrix
    sub_adj_graph = adj_graph[indices, :][:, indices]

    

    # Add bolded text for each city
    if add_city_labels:
        for _, row in filtered_cities.iterrows():
            folium.map.Marker(
                [row['lat'], row['long']],
                icon=folium.DivIcon(
                    icon_size=(150,36),
                    icon_anchor=(0,0),
                    html=f'<div style="font-size: 12pt; font-weight: bold">{row["city-state"].split(",")[0].split("(")[0]}</div>',
                    )
            ).add_to(m)

    # Draw lines between adjacent cities
    cx, cy = sub_adj_graph.nonzero()
    for i, j in zip(cx, cy):
        if i < j:  # To avoid double drawing
            start_coords = (filtered_cities.loc[indices[i], 'lat'], filtered_cities.loc[indices[i], 'long'])
            end_coords = (filtered_cities.loc[indices[j], 'lat'], filtered_cities.loc[indices[j], 'long'])
            folium.PolyLine([start_coords, end_coords], color="black", weight=2.5, opacity=0.5).add_to(m)

            # Annotation with distance in km using haversine
            lat1, lon1 = filtered_cities.loc[indices[i], ['lat', 'long']]
            lat2, lon2 = filtered_cities.loc[indices[j], ['lat', 'long']]
            dist = location.haversine_distance(lat1, lon1, lat2, lon2)

            # place text at midpoint of edge
            # and rotate text to be parallel to edge
            text_rotate_angle = np.degrees(np.arctan2(lat2 - lat1, lon2 - lon1))

            # Sometimes the text appears upside down; to addres this, we
            # check to see if the text is upside down and if so, we rotate
            # it by 180 degrees
            if text_rotate_angle > 90:
                text_rotate_angle -= 180
            elif text_rotate_angle < -90:
                text_rotate_angle += 180
            if add_dist_labels:
                folium.map.Marker(
                    [(lat1 + lat2) / 2, (lon1 + lon2) / 2],
                    icon=folium.DivIcon(
                        icon_size=(150,36),
                        icon_anchor=(0,0),
                        html=f'<div style="font-size: 8pt; ">{dist:.0f} km</div>',
                        )
                        
                ).add_to(m)

    # Add markers for each city
    for _, row in filtered_cities.iterrows():
        folium.CircleMarker(
            location=[row['lat'], row['long']],
            radius=5,
            popup=row['city-state'].split(',')[0],
            color='blue',
            fill=True,
            fill_color='blue'
        ).add_to(m)


    
            
    # Set zoom level to see all cities
    m.fit_bounds(m.get_bounds())
    
    return m