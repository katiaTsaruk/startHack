import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# The data can also be extracted from the "PitchBook"
# Extended sample startup data with performance metrics
data = {
    'Startup Name': [
        'EnergyCell', 'VoltCraft', 'BatteryBoost', 'GreenPowerTech', 'CellFusion',
        'BatteryGen', 'PowerPlus', 'NextGenCells', 'EcoBattery', 'ReVolt',
        'LithiumHub', 'SuperCharge', 'Batterizer', 'TechPower', 'PowerGrid'
    ],
    'Focus Area': [
        'Battery Storage', 'Electric Vehicles', 'Energy Storage', 'Solar Batteries',
        'Battery Recycling', 'Battery Storage', 'Electric Vehicles', 'Battery Storage',
        'Battery Recycling', 'Energy Storage', 'Electric Vehicles', 'Battery Storage',
        'Battery Recycling', 'Electric Vehicles', 'Energy Storage'
    ],
    'Technology': [
        'Lithium-Ion', 'Solid-State', 'Graphene', 'Lithium-Sulfur', 'Recycling Technology',
        'Lithium-Ion', 'Solid-State', 'Graphene', 'Recycling Technology', 'Lithium-Ion',
        'Solid-State', 'Supercapacitors', 'Graphene', 'Lithium-Ion', 'Battery Management Systems'
    ],
    'Funding ($M)': [
        10, 50, 25, 30, 15, 20, 40, 18, 8, 35,
        60, 45, 12, 50, 27
    ],
    'Founded Year': [
        2020, 2018, 2021, 2019, 2022, 2017, 2016, 2021, 2022, 2019,
        2018, 2020, 2019, 2021, 2020
    ],
    'Location': [
        'Germany', 'USA', 'China', 'Germany', 'France',
        'UK', 'USA', 'Germany', 'Germany', 'USA',
        'Japan', 'Sweden', 'Canada', 'USA', 'India'
    ],
    'Market Focus': [
        'Residential', 'Automotive', 'Commercial', 'Solar Energy', 'Recycling',
        'Residential', 'Automotive', 'Residential', 'Recycling', 'Commercial',
        'Automotive', 'Residential', 'Recycling', 'Automotive', 'Commercial'
    ],
    'Employee Count': [
        50, 200, 80, 45, 30, 70, 150, 60, 25, 90,
        250, 100, 40, 130, 60
    ],
    # Simulated performance data
    'Performance Score': [
        85, 92, 80, 75, 65, 70, 90, 78, 82, 88,
        95, 90, 74, 81, 89
    ],  # Performance based on customer satisfaction, product effectiveness, etc.
    'Revenue Growth (%)': [
        20, 35, 15, 10, 8, 18, 25, 12, 10, 22,
        30, 28, 12, 20, 18
    ],  # Percentage revenue growth in the last year
    'Market Adoption (%)': [
        65, 80, 70, 55, 50, 60, 75, 68, 45, 70,
        85, 72, 40, 80, 60
    ]  # Market adoption rate based on client acquisition
}

# Create a DataFrame with more startup details
df = pd.DataFrame(data)


# Function to recommend startups based on industry focus and technology
def recommend_startups(industry_focus, technology=None):
    # Filter the startups that match the desired industry focus
    relevant_startups = df[df['Focus Area'].str.contains(industry_focus, case=False)]

    # Optionally filter by technology type
    if technology:
        relevant_startups = relevant_startups[relevant_startups['Technology'].str.contains(technology, case=False)]

    return relevant_startups


# Function to summarize startups with key points and performance
def summarize_startups(startups):
    if startups.empty:
        return "No startups found in this industry."
    else:
        # Summarize the key trends and facts about the startups
        total_startups = len(startups)
        total_funding = startups['Funding ($M)'].sum()
        avg_funding = startups['Funding ($M)'].mean()
        avg_employees = startups['Employee Count'].mean()

        # Performance metrics
        avg_performance_score = startups['Performance Score'].mean()
        avg_revenue_growth = startups['Revenue Growth (%)'].mean()
        avg_market_adoption = startups['Market Adoption (%)'].mean()

        focus_area_counts = startups['Focus Area'].value_counts()
        technology_counts = startups['Technology'].value_counts()

        # Summary message
        summary = f"\nSummary of Battery Industry Startups:\n"
        summary += f"\nTotal Startups Found: {total_startups}\n"
        summary += f"Total Funding Across All Startups: ${total_funding}M\n"
        summary += f"Average Funding Per Startup: ${avg_funding:.2f}M\n"
        summary += f"Average Number of Employees: {avg_employees:.2f}\n"

        summary += f"\nPerformance Metrics:\n"
        summary += f" - Average Performance Score: {avg_performance_score:.2f}\n"
        summary += f" - Average Revenue Growth: {avg_revenue_growth:.2f}%\n"
        summary += f" - Average Market Adoption Rate: {avg_market_adoption:.2f}%\n"

        summary += "\nKey Focus Areas in the Battery Industry:\n"
        for area, count in focus_area_counts.items():
            summary += f" - {area}: {count} startup(s)\n"

        summary += "\nKey Technologies in the Battery Industry:\n"
        for tech, count in technology_counts.items():
            summary += f" - {tech}: {count} startup(s)\n"

        return summary


# Example use case: Inform decision-makers about battery startups
industry_focus = 'Battery'
technology_focus = 'Solid-State'  # For example, if VARTA is interested in solid-state battery technologies
relevant_startups = recommend_startups(industry_focus, technology_focus)
startup_summary = summarize_startups(relevant_startups)

# Display the summary (could be emailed or stored in a report in a real-world application)
print(startup_summary)


# Reusing the startup data from the previous code
data = {
    'Startup Name': [
        'EnergyCell', 'VoltCraft', 'BatteryBoost', 'GreenPowerTech', 'CellFusion',
        'BatteryGen', 'PowerPlus', 'NextGenCells', 'EcoBattery', 'ReVolt',
        'LithiumHub', 'SuperCharge', 'Batterizer', 'TechPower', 'PowerGrid'
    ],
    'Focus Area': [
        'Battery Storage', 'Electric Vehicles', 'Energy Storage', 'Solar Batteries',
        'Battery Recycling', 'Battery Storage', 'Electric Vehicles', 'Battery Storage',
        'Battery Recycling', 'Energy Storage', 'Electric Vehicles', 'Battery Storage',
        'Battery Recycling', 'Electric Vehicles', 'Energy Storage'
    ],
    'Technology': [
        'Lithium-Ion', 'Solid-State', 'Graphene', 'Lithium-Sulfur', 'Recycling Technology',
        'Lithium-Ion', 'Solid-State', 'Graphene', 'Recycling Technology', 'Lithium-Ion',
        'Solid-State', 'Supercapacitors', 'Graphene', 'Lithium-Ion', 'Battery Management Systems'
    ],
    'Funding ($M)': [
        10, 50, 25, 30, 15, 20, 40, 18, 8, 35,
        60, 45, 12, 50, 27
    ],
    'Founded Year': [
        2020, 2018, 2021, 2019, 2022, 2017, 2016, 2021, 2022, 2019,
        2018, 2020, 2019, 2021, 2020
    ],
    'Location': [
        'Germany', 'USA', 'China', 'Germany', 'France',
        'UK', 'USA', 'Germany', 'Germany', 'USA',
        'Japan', 'Sweden', 'Canada', 'USA', 'India'
    ],
    'Market Focus': [
        'Residential', 'Automotive', 'Commercial', 'Solar Energy', 'Recycling',
        'Residential', 'Automotive', 'Residential', 'Recycling', 'Commercial',
        'Automotive', 'Residential', 'Recycling', 'Automotive', 'Commercial'
    ],
    'Employee Count': [
        50, 200, 80, 45, 30, 70, 150, 60, 25, 90,
        250, 100, 40, 130, 60
    ],
    # Simulated performance data
    'Performance Score': [
        85, 92, 80, 75, 65, 70, 90, 78, 82, 88,
        95, 90, 74, 81, 89
    ],  # Performance based on customer satisfaction, product effectiveness, etc.
    'Revenue Growth (%)': [
        20, 35, 15, 10, 8, 18, 25, 12, 10, 22,
        30, 28, 12, 20, 18
    ],  # Percentage revenue growth in the last year
    'Market Adoption (%)': [
        65, 80, 70, 55, 50, 60, 75, 68, 45, 70,
        85, 72, 40, 80, 60
    ]  # Market adoption rate based on client acquisition
}

df = pd.DataFrame(data)


# Function to visualize key metrics
def visualize_startup_data(df):
    plt.figure(figsize=(14, 10))

    # 1. Performance Score Bar Chart
    plt.subplot(2, 2, 1)
    sns.barplot(x='Startup Name', y='Performance Score', data=df.sort_values('Performance Score', ascending=False))
    plt.title('Performance Score of Startups')
    plt.xticks(rotation=90)

    # 2. Revenue Growth Scatter Plot
    plt.subplot(2, 2, 2)
    sns.scatterplot(x='Startup Name', y='Revenue Growth (%)', data=df, hue='Technology', palette='viridis', s=100)
    plt.title('Revenue Growth vs Startups')
    plt.xticks(rotation=90)

    # 3. Market Adoption Pie Chart
    plt.subplot(2, 2, 3)
    market_adoption_avg = df['Market Adoption (%)'].mean()
    adoption_category = ['Above Average', 'Below Average']
    adoption_values = [df[df['Market Adoption (%)'] > market_adoption_avg].shape[0],
                       df[df['Market Adoption (%)'] <= market_adoption_avg].shape[0]]

    plt.pie(adoption_values, labels=adoption_category, autopct='%1.1f%%', startangle=90, colors=['#66b3ff', '#ff9999'])
    plt.title('Market Adoption Distribution')

    # 4. Funding vs Employee Count Scatter Plot
    plt.subplot(2, 2, 4)
    sns.scatterplot(x='Funding ($M)', y='Employee Count', data=df, hue='Focus Area', palette='Set1', s=100)
    plt.title('Funding vs Employee Count')

    plt.tight_layout()
    plt.show()


# Call the function to visualize the data
visualize_startup_data(df)
