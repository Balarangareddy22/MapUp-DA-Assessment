from typing import Dict, List

import pandas as pd





# Question 1:
def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    """
    Reverses the input list by groups of n elements.
    """
    # Your code goes here.
    result = []

    for i in range(0, len(lst), n):
        # Reverse the current group of 'n' elements and add to result
        result.extend(lst[i:i+n][::-1])

    return result

#Example
lst = [1, 2, 3, 4, 5, 6, 7, 8, 9]
n = 3
print(reverse_by_n_elements(lst, n))





# Question 2:
def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    """
    Groups the strings by their length and returns a dictionary.
    """
    # Your code here
    # Create an empty dictionary to store the groups
    length_dict = {}
    
    # Iterate over each string in the input list
    for string in lst:
        # Get the length of the current string
        length = len(string)
        
        # Add the string to the appropriate list in the dictionary
        if length in length_dict:
            length_dict[length].append(string)
        else:
            length_dict[length] = [string]
    
    # Return a dictionary sorted by the string lengths (keys)
    return dict(sorted(length_dict.items()))
    
#Example
input_list1 = ["apple", "bat", "car", "elephant", "dog", "bear"]
input_list2 = ["one", "two", "three", "four"]

print(group_by_length(input_list1))
print(group_by_length(input_list2))





# Question 3:
def flatten_dict(nested_dict: Dict, sep: str = '.') -> Dict:
    """
    Flattens a nested dictionary into a single-level dictionary with dot notation for keys.
    
    :param nested_dict: The dictionary object to flatten
    :param sep: The separator to use between parent and child keys (defaults to '.')
    :return: A flattened dictionary
    """
    # Your code here
        # Helper function to perform recursive flattening
    def _flatten(obj: Any, parent_key: str = '') -> Dict:
        items = {}
        
        # Iterate over the items in the current dictionary
        if isinstance(obj, dict):
            for k, v in obj.items():
                # Create new key based on parent key and separator
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                items.update(_flatten(v, new_key))
        
        # Handle list values by including the index in the key
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                new_key = f"{parent_key}[{i}]"
                items.update(_flatten(item, new_key))
        
        # For non-dict and non-list items, add them to the result
        else:
            items[parent_key] = obj
        
        return items
    
    # Call the helper function starting with the root dictionary
    return _flatten(nested_dict)
    
#Example 
nested_dict = {
    "road": {
        "name": "Highway 1",
        "length": 350,
        "sections": [
            {
                "id": 1,
                "condition": {
                    "pavement": "good",
                    "traffic": "moderate"
                }
            }
        ]
    }
}

flattened_dict = flatten_dict(nested_dict)
for key, value in flattened_dict.items():
    print(f"{key}: {value}")





# Question 4:
def unique_permutations(nums: List[int]) -> List[List[int]]:
    """
    Generate all unique permutations of a list that may contain duplicates.
    
    :param nums: List of integers (may contain duplicates)
    :return: List of unique permutations
    """
    # Your code here    
    # List to store all unique permutations
    result = []
    
    # Helper function for backtracking
    def backtrack(path, used):
        # When the path has all the elements, add it as a valid permutation
        if len(path) == len(nums):
            result.append(path[:])
            return
        
        # Iterate over the numbers to form permutations
        for i in range(len(nums)):
            # Skip used numbers or duplicate numbers (to ensure uniqueness)
            if used[i] or (i > 0 and nums[i] == nums[i - 1] and not used[i - 1]):
                continue
            
            # Mark the current number as used and add to the path
            used[i] = True
            path.append(nums[i])
            
            # Recurse with the updated path and used array
            backtrack(path, used)
            
            # Backtrack by removing the last number and unmarking it
            path.pop()
            used[i] = False
    
    # Start the backtracking with an empty path and 'used' flag array
    backtrack([], [False] * len(nums))
    
    return result

input_list = [1, 1, 2]
permutations = unique_permutations(input_list)
for perm in permutations:
    print(perm)





# Question 5:
def find_all_dates(text: str) -> List[str]:
    """
    This function takes a string as input and returns a list of valid dates
    in 'dd-mm-yyyy', 'mm/dd/yyyy', or 'yyyy.mm.dd' format found in the string.
    
    Parameters:
    text (str): A string containing the dates in various formats.

    Returns:
    List[str]: A list of valid dates in the formats specified.
    """
    import re
    # Define the regular expression pattern for the three date formats
    date_pattern = r'\b\d{2}-\d{2}-\d{4}\b|\b\d{2}/\d{2}/\d{4}\b|\b\d{4}\.\d{2}\.\d{2}\b'
    
    # Use re.findall to get all matches of the pattern in the input text
    text = "I was born on 23-08-1994, my friend on 08/23/1994, and another one on 1994.08.23."
    matches = re.findall(date_pattern, text)
    
    # Return the list of matched date strings
    return matches

dates = find_all_dates(text)
print(dates)





# Question 6:
from math import radians, sin, cos, sqrt, atan2

def haversine(lat1, lon1, lat2, lon2):
    # Radius of the Earth in meters
    R = 6371000

    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    # Differences in coordinates
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Haversine formula
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    # Distance in meters
    return R * c

def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    """
    Converts a polyline string into a DataFrame with latitude, longitude, and distance between consecutive points.
    
    Args:
        polyline_str (str): The encoded polyline string.

    Returns:
        pd.DataFrame: A DataFrame containing latitude, longitude, and distance in meters.
    """
    # Decode the polyline string into a list of (latitude, longitude) tuples
    coordinates = polyline.decode(polyline_str)
    # Create a DataFrame with latitude and longitude columns
    df = pd.DataFrame(coordinates, columns=['latitude', 'longitude'])
    
    # Initialize the distance column with 0 for the first row
    df['distance'] = 0.0
    
    # Calculate the distance between successive points using the Haversine formula
    for i in range(1, len(df)):
        lat1, lon1 = df.loc[i-1, ['latitude', 'longitude']]
        lat2, lon2 = df.loc[i, ['latitude', 'longitude']]
        
        # Calculate the distance and store it in the DataFrame
        df.loc[i, 'distance'] = haversine(lat1, lon1, lat2, lon2)
    
    return df





# Question 7:
def rotate_and_multiply_matrix(matrix: List[List[int]]) -> List[List[int]]:
    """
    Rotate the given matrix by 90 degrees clockwise, then multiply each element 
    by the sum of its original row and column index before rotation.
    
    Args:
    - matrix (List[List[int]]): 2D list representing the matrix to be transformed.
    
    Returns:
    - List[List[int]]: A new 2D list representing the transformed matrix.
    """
    # Your code here
    matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]
    n = len(matrix)
    
    # Step 1: Rotate the matrix 90 degrees clockwise
    rotated_matrix = [[0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            rotated_matrix[j][n - 1 - i] = matrix[i][j]
    
    # Step 2: Replace each element with the sum of all elements in the same row and column, excluding itself
    final_matrix = [[0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            # Sum of row elements excluding the current element
            row_sum = sum(rotated_matrix[i]) - rotated_matrix[i][j]
            
            # Sum of column elements excluding the current element
            col_sum = sum(rotated_matrix[k][j] for k in range(n)) - rotated_matrix[i][j]
            
            # Set the new value as the sum of row_sum and col_sum
            final_matrix[i][j] = row_sum + col_sum
    
    return final_matrix



transformed_matrix = rotate_and_multiply_matrix(matrix)
for row in transformed_matrix:
    print(row)





# Question 8:
def time_check(df) -> pd.Series:
    """
    Use shared dataset-2 to verify the completeness of the data by checking whether the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour and 7 days period

    Args:
        df (pandas.DataFrame)

    Returns:
        pd.Series: return a boolean series
    """
    # Write your logic here
    # Ensure the timestamp columns are in datetime format
    df['start'] = pd.to_datetime(df['startDay'] + ' ' + df['startTime'])
    df['end'] = pd.to_datetime(df['endDay'] + ' ' + df['endTime'])
    
    # Create a MultiIndex based on (id, id_2)
    df.set_index(['id', 'id_2'], inplace=True)
    
    # Function to check completeness for each group
    def check_group(group):
        # Create a full 24-hour time range for each day of the week
        full_range = pd.date_range(start='2021-01-01', end='2021-01-07', freq='D')
        all_times = []
        
        for day in full_range:
            # Create a 24-hour period for each day
            day_range = pd.date_range(start=day, end=day + pd.Timedelta(hours=23, minutes=59, seconds=59), freq='S')
            all_times.append(day_range)
        
        all_times = pd.Index(pd.concat(all_times))  # Combine all time ranges for the week

        # Collect all timestamps in the group
        timestamps = pd.concat([group['start'], group['end']]).drop_duplicates()

        # Check if all 24-hour periods are covered
        complete = all_times.isin(timestamps).all()
        
        return not complete  # Return True if there are gaps (incorrect timestamps)
    
    # Apply the completeness check to each group and return a boolean series
    result = df.groupby(level=[0, 1]).apply(check_group)

    return result
