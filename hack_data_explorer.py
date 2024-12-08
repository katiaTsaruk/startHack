import pandas as pd
import numpy as np
import random
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, mean_squared_error
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans


# Set random seed for reproducibility
random.seed(42)

#############################################################################################
# Step 1: Generate a larger synthetic dataset
battery_types = ['AA', 'AAA', 'C', 'D', '9V', 'CR2032', 'CR123A', 'AA NiMH', 'AAA NiMH', 'D NiMH']
brands = ['VARTA', 'Energizer', 'Duracell', 'Panasonic', 'AmazonBasics']

# Generate synthetic data
# Generate more meaningful synthetic data
data = {
    'Battery_Type': [random.choice(battery_types) for _ in range(100)],

    # Adjust capacity ranges based on battery type
    'Capacity_mAh': [
        random.randint(800, 10000) if battery != 'CR2032' else random.randint(200, 250)
        # CR2032 is a coin battery with lower capacity
        for battery in [random.choice(battery_types) for _ in range(100)]
    ],

    # Voltage is related to the battery type
    'Voltage_V': [
        random.choice([1.5, 3.7, 9]) if battery != 'CR2032' else 3.0  # CR2032 typically has 3V
        for battery in [random.choice(battery_types) for _ in range(100)]
    ],

    # Weight ranges adjusted for battery type
    'Weight_g': [
        random.uniform(20, 30) if battery == 'CR2032' else random.uniform(30,
                                                                          100) if battery != '9V' else random.uniform(
            30, 45)
        for battery in [random.choice(battery_types) for _ in range(100)]
    ],

    # Price is generally higher for larger capacity and more reputable brands
    'Price_USD': [
        random.uniform(1.99, 5.99) if battery == 'CR2032' else random.uniform(3.99, 15.99)
        for battery in [random.choice(battery_types) for _ in range(100)]
    ],

    # Battery life depends on capacity and rechargeability
    'Battery_Life_h': [
        random.randint(10, 50) if battery != 'CR2032' else random.randint(1, 10)  # Coin batteries have shorter life
        for battery in [random.choice(battery_types) for _ in range(100)]
    ],

    # Brand influences customer ratings and price
    'Brand': [random.choice(brands) for _ in range(100)],

    # Rechargeable or non-rechargeable, rechargeable batteries tend to have higher capacity
    'Rechargeable': [
        random.choice([1, 0]) if battery != 'CR2032' else 0  # CR2032 are typically non-rechargeable
        for battery in [random.choice(battery_types) for _ in range(100)]
    ],

    # Only rechargeable batteries have charge time
    'Charge_Time_h': [
        random.uniform(1, 5) if x == 1 else 0
        for x in [random.choice([1, 0]) for _ in range(100)]
    ],

    # Eco-friendly batteries are usually rechargeable and slightly more expensive
    'Eco_Friendly': [
        random.choice([1, 0]) if battery != 'CR2032' else 0  # CR2032 tends to be non-eco-friendly
        for battery in [random.choice(battery_types) for _ in range(100)]
    ],

    # Customer ratings, influenced by brand reputation and battery quality
    'Customer_Rating': [
        random.uniform(3.5, 5) if brand != 'AmazonBasics' else random.uniform(3.0, 4.5)
        for brand in [random.choice(brands) for _ in range(100)]
    ]
}


# Convert to DataFrame
df = pd.DataFrame(data)

# Display a sample of the data
print(df.head())



##########################################################
# Step 2: Store the Data in SQLite Database
def create_database(df):
    conn = sqlite3.connect('../varta_battery_data.db')
    df.to_sql('batteries', conn, if_exists='replace', index=False)
    conn.close()

# Create and store the data in the database
create_database(df)


##########################################################
# Step 3: Train ML Models to Predict Product Features
def predict_battery_price(df):
    # Features and target
    X = df[['Capacity_mAh', 'Voltage_V', 'Weight_g', 'Battery_Life_h', 'Rechargeable', 'Charge_Time_h', 'Eco_Friendly', 'Customer_Rating']]
    y = df['Price_USD']

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a regression model (RandomForestRegressor)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Output the performance
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error for Price Prediction: {mse}')

# Predict battery price
predict_battery_price(df)

def predict_rechargeable(df):
    # Select features and target variable
    X = df[['Capacity_mAh', 'Voltage_V', 'Weight_g', 'Battery_Life_h', 'Price_USD', 'Charge_Time_h', 'Eco_Friendly']]
    y = df['Rechargeable']

    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the classifier model (RandomForestClassifier)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the model
    clf.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = clf.predict(X_test)

    # Output the classification performance
    print("Classification Report for Rechargeable Prediction:")
    print(classification_report(y_test, y_pred))

    # Feature importance - showing the most important features for predicting rechargeability
    feature_importance = clf.feature_importances_
    features = X.columns
    feature_df = pd.DataFrame({'Feature': features, 'Importance': feature_importance})
    print("\nFeature Importance:")
    print(feature_df.sort_values(by='Importance', ascending=False))

    # Fancy Plot for Feature Importance
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_df.sort_values(by='Importance', ascending=False),
                palette='Blues_d')  # Using a blue color palette for a cool, professional look
    plt.title('Feature Importance for Predicting Battery Rechargeability', fontsize=16, fontweight='bold')
    plt.xlabel('Importance Score', fontsize=14)
    plt.ylabel('Feature', fontsize=14)

    # Add value labels on the bars for better readability
    for index, value in enumerate(feature_df['Importance']):
        plt.text(value + 0.01, index, f'{value:.3f}', color='black', ha="center", va="center", fontsize=12,
                 fontweight='bold')

    # Enhance plot with a grid, tight layout, and a dark background
    plt.grid(True, axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Show the plot
    plt.show()

# Step 4: Run the prediction function
predict_rechargeable(df)

# Step 5: Product Segmentation using KMeans Clustering

# Apply KMeans clustering
# Select features for clustering
X_clustering = df[['Capacity_mAh', 'Voltage_V', 'Weight_g', 'Battery_Life_h', 'Price_USD', 'Charge_Time_h', 'Eco_Friendly']]

# Normalize the data (important for clustering algorithms)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_clustering)

# Apply KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Define a custom color palette with distinct colors for each cluster
custom_palette = {0: '#1f77b4', 1: '#ff7f0e', 2: '#2ca02c'}  # Blue, Orange, Green (distinct colors)

plt.figure(figsize=(10, 6))

# Scatterplot of the clusters with custom color palette
sns.scatterplot(x='Capacity_mAh', y='Price_USD', hue='Cluster', data=df, palette=custom_palette, s=100)

# Customize plot to show titles, axis labels, and legend
plt.title(
    "Product Segmentation: Battery Features vs Price\n"
    "Cluster 0: Low price, moderate capacity, short battery life.\n"
    "Cluster 1: High capacity, long battery life, high price.\n"
    "Cluster 2: Eco-friendly, rechargeable, medium price."
)
plt.xlabel("Capacity (mAh)")
plt.ylabel("Price (USD)")
plt.legend(title='Cluster', labels=["Cluster 0", "Cluster 1", "Cluster 2"])  # Adding explicit cluster labels

# Add text annotations for cluster descriptions
cluster_descriptions = {
    0: "Cluster 0: Low price, moderate capacity, short battery life.",
    1: "Cluster 1: High capacity, long battery life, high price.",
    2: "Cluster 2: Eco-friendly, rechargeable, medium price."
}

# Calculate the centroid of each cluster for annotation
centroids = kmeans.cluster_centers_
plt.show()


# Step 5: Analyze the characteristics of each cluster
# Calculate mean values for each cluster
numeric_columns = df.select_dtypes(include=[np.number]).columns

# Calculate mean values for each cluster for only numeric columns
cluster_summary = df.groupby('Cluster')[numeric_columns].mean()

print("Cluster Summary (Mean Values for Each Cluster):")
print(cluster_summary)

# Optionally, you can also examine the distribution of each feature within clusters
# This can help identify trends and patterns.
for column in X_clustering.columns:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Cluster', y=column, data=df)
    plt.title(f'Distribution of {column} by Cluster')
    plt.xlabel('Cluster')
    plt.ylabel(column)
    plt.show()


# Step 7: Analyze the product segments
print("\nProduct Segmentation Analysis:")
for cluster in range(3):
    cluster_data = df[df['Cluster'] == cluster]
    print(f"\nCluster {cluster} (Size: {len(cluster_data)}):")
    print(cluster_data[['Capacity_mAh', 'Voltage_V', 'Weight_g', 'Battery_Life_h', 'Price_USD', 'Charge_Time_h', 'Eco_Friendly']].describe())


# For example, we can find which features optimize the battery price while keeping high customer ratings.
def optimize_battery_features(df):
    # This can be an optimization problem to find the best battery configuration.
    # Let's say we want to maximize price while keeping customer rating high.

    # Objective function: maximize price while keeping customer rating high
    def objective(x):
        # x[0]: Capacity, x[1]: Weight, x[2]: Life, x[3]: Rechargeable (0 or 1)
        return -(x[0] * 0.1 + x[1] * 0.2 + x[2] * 0.3 + x[3] * 10)  # Example function to maximize price

    # Constraints: Keep customer rating above a certain threshold
    cons = ({'type': 'ineq', 'fun': lambda x: np.array([5 - (x[0] * 0.2 + x[1] * 0.3)])})  # Example constraint

    # Initial guess
    initial_guess = [3000, 20, 20, 1]  # [Capacity, Weight, Life, Rechargeable]

    # Minimize (optimization process)
    result = minimize(objective, initial_guess, constraints=cons)

    # Print the optimized battery configuration
    print(f"Optimized battery configuration: {result.x}")

    # Return the result for plotting
    return result.x


# Get optimized battery configuration
optimized_config = optimize_battery_features(df)


# Visualizing the optimized result in 3D
def plot_optimization(df, optimized_config):
    # Plotting the original data in 3D
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot of the original data
    ax.scatter(df['Capacity_mAh'], df['Weight_g'], df['Battery_Life_h'], c=df['Price_USD'], cmap='viridis', s=50,
               edgecolors='k', alpha=0.7)

    # Scatter the optimized configuration
    ax.scatter(optimized_config[0], optimized_config[1], optimized_config[2], c='r', marker='x', s=100,
               label='Optimized Configuration')

    # Adding labels and title
    ax.set_xlabel('Capacity (mAh)')
    ax.set_ylabel('Weight (g)')
    ax.set_zlabel('Battery Life (h)')
    ax.set_title('Battery Feature Optimization', fontsize=16)

    # Adding a color bar
    cbar = plt.colorbar(ax.collections[0], ax=ax, shrink=0.5, aspect=10)
    cbar.set_label('Price (USD)', fontsize=12)

    # Show the legend
    ax.legend()

    plt.show()


# Plot the optimization results
plot_optimization(df, optimized_config)
