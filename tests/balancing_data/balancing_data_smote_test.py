from eis_toolkit.balancing_data import balance_data, data_info, drop_columns, load_data, plot_data_distribution

# defining the paths for two csv data files
path1 = "/home/dipak/Desktop/Final_EIS_Data/2M_raster_points.csv"
path2 = "/home/dipak/Desktop/Final_EIS_Data/17_annoted_points.csv"


data_frame = load_data(path1, path2)
data_frame = drop_columns(data_frame, ["E", "N"])

# Define data as X and class as y
X = data_frame.drop("class", axis=1)
y = data_frame["class"]
print(f"X.shape--->{X.shape} and y.shape--->{y.shape}")

# plotting the data_distribution
plot_data_distribution(data_frame)

# calling the data_info function
data_info(data_frame)

# calling the function balance_data to address the issue for class imbalance
X_SMOTET_balance, y_SMOTET_balance = balance_data(X, y)
