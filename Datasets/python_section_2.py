import pandas as pd

# Question 9:
def calculate_distance_matrix(df)->pd.DataFrame():
    """
    Calculate a distance matrix based on the dataframe, df.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Distance matrix
    """
    # Write your logic here
    # Load the dataset
    df = pd.read_csv("C:\Users\balar\OneDrive\Desktop\Mapup\MapUp-DA-Assessment-2024\datasets\dataset-2.csv")
    
    # Create a unique list of IDs
    ids = pd.unique(df[['id_start', 'id_end']].values.ravel('K'))
    
    # Initialize the distance matrix with zeros
    distance_matrix = pd.DataFrame(0, index=ids, columns=ids)
    
    # Fill in the known distances
    for _, row in df.iterrows():
        distance_matrix.loc[row['id_start'], row['id_end']] = row['distance']
        distance_matrix.loc[row['id_end'], row['id_start']] = row['distance']  # Make it symmetric
    
    # Calculate cumulative distances using the Floyd-Warshall algorithm
    for k in ids:
        for i in ids:
            for j in ids:
                if distance_matrix.loc[i, k] > 0 and distance_matrix.loc[k, j] > 0:
                    distance_matrix.loc[i, j] = max(distance_matrix.loc[i, j], distance_matrix.loc[i, k] + distance_matrix.loc[k, j])
    
    return distance_matrix

if __name__ == "__main__":
    # Define the file path
    file_path = r"C:\Users\balar\OneDrive\Desktop\Mapup\MapUp-DA-Assessment-2024\datasets\dataset-2.csv"
    
    # Calculate the distance matrix
    distance_df = calculate_distance_matrix(file_path)

    # Print the resulting distance matrix
    print(distance_df)
   




# Question 10:
def unroll_distance_matrix(df)->pd.DataFrame():
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """
    # Write your logic here
    # Create a list to hold the rows of the new DataFrame
    unrolled_data = []
    
    # Iterate over the index and columns of the distance matrix
    for id_start in df.index:
        for id_end in df.columns:
            # Skip the self-distance entries (where id_start == id_end)
            if id_start != id_end:
                distance = df.loc[id_start, id_end]
                unrolled_data.append({'id_start': id_start, 'id_end': id_end, 'distance': distance})
    
    # Convert the list of dictionaries into a DataFrame
    unrolled_df = pd.DataFrame(unrolled_data)
    
    return unrolled_df

if __name__ == "__main__":
    # Define the file path
    file_path = r"C:\Users\balar\OneDrive\Desktop\Mapup\MapUp-DA-Assessment-2024\datasets\dataset-2.csv"
    
    # Calculate the distance matrix
    distance_df = calculate_distance_matrix(file_path)

    # Unroll the distance matrix
    unrolled_df = unroll_distance_matrix(distance_df)

    # Print the resulting unrolled DataFrame
    print(unrolled_df)





# Question 11:
import numpy as np
def find_ids_within_ten_percentage_threshold(df, reference_id)->pd.DataFrame():
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame)
        reference_id (int)

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """
    # Write your logic here
    # Filter the DataFrame for rows where id_start matches the reference_id
    reference_distances = df[df['id_start'] == reference_id]['distance']
    
    # Calculate the average distance for the reference_id
    if reference_distances.empty:
        return []  # Return empty list if reference_id has no distances
    
    average_distance = reference_distances.mean()
    
    # Calculate the 10% threshold
    lower_bound = average_distance * 0.9
    upper_bound = average_distance * 1.1
    
    # Get unique id_start values that fall within the threshold
    ids_within_threshold = df[
        (df['distance'] >= lower_bound) & (df['distance'] <= upper_bound)
    ]['id_start'].unique()
    
    # Convert to list and sort
    sorted_ids = sorted(ids_within_threshold)
    
    return sorted_ids

if __name__ == "__main__":
    # Define the file path
    file_path = r"C:\Users\balar\OneDrive\Desktop\Mapup\MapUp-DA-Assessment-2024\datasets\dataset-2.csv"
    
    # Calculate the distance matrix
    distance_df = calculate_distance_matrix(file_path)

    # Unroll the distance matrix
    unrolled_df = unroll_distance_matrix(distance_df)

    # Example usage of find_ids_within_ten_percentage_threshold
    reference_id = 1  # Replace with your desired reference ID
    result_ids = find_ids_within_ten_percentage_threshold(unrolled_df, reference_id)

    # Print the resulting sorted list of IDs
    print(result_ids)






# Question 12:
def calculate_toll_rate(df)->pd.DataFrame():
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Wrie your logic here
    # Define the rate coefficients for different vehicle types
    rate_coefficients = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }
    
    # Iterate through each vehicle type and calculate the toll rate
    for vehicle, rate in rate_coefficients.items():
        df[vehicle] = df['distance'] * rate
    
    return df

if __name__ == "__main__":
    # Example usage, assuming the unrolled DataFrame from Question 10 is available
    file_path = r"C:\Users\balar\OneDrive\Desktop\Mapup\MapUp-DA-Assessment-2024\datasets\dataset-2.csv"
    
    # Generate the distance matrix from Question 10
    distance_df = calculate_distance_matrix(file_path)

    # Unroll the distance matrix from Question 10
    unrolled_df = unroll_distance_matrix(distance_df)

    # Calculate toll rates based on the unrolled DataFrame
    toll_rate_df = calculate_toll_rate(unrolled_df)

    # Print the resulting DataFrame with toll rates
    print("Toll Rate DataFrame:")
    print(toll_rate_df)






# Question 13:
from datetime import time
def calculate_time_based_toll_rates(df)->pd.DataFrame():
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Write your logic here
    # Define discount factors for different time intervals
    weekday_discount_factors = {
        'morning': 0.8,  # 00:00:00 to 10:00:00
        'day': 1.2,      # 10:00:00 to 18:00:00
        'evening': 0.8   # 18:00:00 to 23:59:59
    }
    weekend_discount_factor = 0.7

    # Define time intervals
    morning_time = (time(0, 0, 0), time(10, 0, 0))
    day_time = (time(10, 0, 0), time(18, 0, 0))
    evening_time = (time(18, 0, 0), time(23, 59, 59))

    # Define the days of the week
    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    weekends = ['Saturday', 'Sunday']

    # Add new columns for start_day, end_day, start_time, and end_time
    full_week = weekdays + weekends
    time_periods = [(morning_time, 'morning'), (day_time, 'day'), (evening_time, 'evening')]

    expanded_rows = []
    for index, row in df.iterrows():
        for day in full_week:
            for time_range, period in time_periods:
                # Apply discount based on the day of the week and time
                if day in weekdays:
                    discount = weekday_discount_factors[period]
                else:  # weekend
                    discount = weekend_discount_factor

                # Adjust the toll rates by the discount factor for each vehicle type
                adjusted_row = row.copy()
                adjusted_row['start_day'] = day
                adjusted_row['end_day'] = day
                adjusted_row['start_time'] = time_range[0]
                adjusted_row['end_time'] = time_range[1]
                
                # Adjust toll rates
                adjusted_row['moto'] *= discount
                adjusted_row['car'] *= discount
                adjusted_row['rv'] *= discount
                adjusted_row['bus'] *= discount
                adjusted_row['truck'] *= discount

                expanded_rows.append(adjusted_row)

    # Create a new DataFrame with the expanded rows
    expanded_df = pd.DataFrame(expanded_rows)

    return expanded_df

if __name__ == "__main__":
    # Example usage, assuming the toll_rate DataFrame from Question 12 is available
    file_path = r"C:\Users\balar\OneDrive\Desktop\Mapup\MapUp-DA-Assessment-2024\datasets\dataset-2.csv"
    
    # Generate the distance matrix from Question 10
    distance_df = calculate_distance_matrix(file_path)

    # Unroll the distance matrix from Question 10
    unrolled_df = unroll_distance_matrix(distance_df)

    # Calculate toll rates based on the unrolled DataFrame (from Question 12)
    toll_rate_df = calculate_toll_rate(unrolled_df)

    # Calculate time-based toll rates
    time_based_toll_rates_df = calculate_time_based_toll_rates(toll_rate_df)

    # Print the resulting DataFrame with time-based toll rates
    print("Time-based Toll Rate DataFrame:")
    print(time_based_toll_rates_df)

