import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import numpy as np

# Download the VADER model if not already installed
nltk.download('vader_lexicon')

# Initialize the Sentiment Intensity Analyzer
sia = SentimentIntensityAnalyzer()

varta_reviews = [
    "VARTA batteries are very durable and last a long time.",
    "I wasn't happy with VARTA. They drained quickly in my flashlight.",
    "Affordable and reliable, VARTA batteries are my go-to choice.",
    "These batteries leaked after a month of use. Very disappointing!",
    "Perfect for my remote control. VARTA batteries lasted over a year.",
    "Not suitable for high-drain devices like cameras.",
    "Great value for the price. I would definitely recommend VARTA batteries.",
    "The packaging was excellent, and the batteries worked as expected.",
    "They run out faster than Energizer batteries. Not impressed.",
    "Solid performance for basic household electronics.",
    "VARTA batteries gave me consistent results in my gaming controller.",
    "Failed to power my high-tech gadgets. Not worth the money.",
    "I love that VARTA offers environmentally friendly options.",
    "Very dependable in my wall clock and small devices.",
    "I switched to VARTA from Duracell, and I am very happy with the results.",
    "The batteries died after just two weeks. I expected more.",
    "VARTA batteries have been a lifesaver for my emergency flashlights.",
    "Great product for the price point. Works well in toys and remotes.",
    "Not as durable as advertised. My flashlight lasted only an hour.",
    "Highly recommended for low-power devices like remotes.",
    "These batteries are lightweight and easy to store.",
    "I have never experienced leaks with VARTA batteries. Great quality.",
    "The batteries arrived with low charge. Very unsatisfactory.",
    "VARTA batteries last longer than most generic brands.",
    "These batteries struggle in cold weather conditions.",
    "Good performance for medium-drain applications like clocks and radios.",
    "The shelf life is excellent, even after two years of storage.",
    "VARTA batteries are reliable and very affordable.",
    "I used these in my wireless mouse, and they worked perfectly.",
    "Not great for continuous usage. Drains faster than expected.",
    "The price is reasonable, but the performance could be better.",
    "No complaints so far. These batteries are working great.",
    "I love how long these batteries last in my portable fan.",
    "They stopped working after just one week. Very disappointing.",
    "I trust VARTA for all my low-power electronics at home.",
    "Good quality but lacks the longevity of premium brands.",
    "Best value for money if you're on a budget.",
    "The batteries corroded after a few months. Poor quality.",
    "Highly recommend for standard use cases like TV remotes.",
    "These batteries drained quickly in my camera flash.",
    "I appreciate that these are eco-friendly options for batteries.",
    "The durability exceeded my expectations for such a low price.",
    "These batteries are reliable and have never let me down.",
    "VARTA batteries are lightweight and perfect for travel.",
    "Good for clocks but not suitable for high-drain devices.",
    "Consistent power output for small gadgets.",
    "They last longer than some other affordable brands.",
    "I used these in my kids' toys, and they lasted for months.",
    "Decent performance, but I'd prefer Energizer for longer usage.",
    "I had a positive experience with VARTA in all my household devices."
]
energizer_reviews = [
    "Energizer batteries always perform well and never let me down.",
    "The new Energizer batteries were disappointing compared to the old ones.",
    "These batteries are reliable for all my electronic devices.",
    "I love the long-lasting power of Energizer. Worth the price.",
    "Energizer batteries drained faster than expected in my flashlight.",
    "Perfect for my wireless keyboard and mouse. Energizer rocks!",
    "I’ve trusted Energizer for years, and they continue to deliver quality.",
    "These batteries are too expensive for the performance they offer.",
    "They worked well in my kid’s toys and lasted a long time.",
    "The packaging was damaged, but the batteries performed well.",
    "Energizer batteries are excellent for high-drain devices like cameras.",
    "I found that these last longer than VARTA batteries in my remotes.",
    "Not as long-lasting as advertised, but still better than generic brands.",
    "The durability of these batteries is unmatched. Highly recommend!",
    "Energizer batteries are a bit pricey, but they are worth every penny.",
    "I wasn’t happy with the performance of these batteries in cold weather.",
    "Great for travel! Energizer batteries powered all my devices effortlessly.",
    "The longevity of these batteries is why I keep buying them.",
    "Energizer has been my go-to brand for years. Never disappoints!",
    "These batteries failed to power my high-drain flashlight effectively.",
    "Super dependable for everyday gadgets like remotes and clocks.",
    "I prefer Energizer over Duracell for the consistent quality.",
    "The batteries leaked after three months. Very disappointing experience.",
    "I love how long Energizer batteries last in my gaming controller.",
    "The rechargeables are excellent, but the disposable ones could be better.",
    "I used these in my smart thermostat, and they worked flawlessly.",
    "Energizer batteries lasted longer than expected in my camera flash.",
    "These are the best batteries for heavy-duty applications.",
    "The performance is great, but I wish they were more affordable.",
    "Energizer batteries have exceeded my expectations in every way.",
    "I recommend these for low-power devices, but not for high-power gadgets.",
    "The batteries didn’t last as long as promised. Disappointed.",
    "These are solid batteries for emergency use. Reliable and durable.",
    "My Energizer batteries lasted for over a year in my wall clock.",
    "The quality is top-notch, but they could improve the pricing.",
    "Energizer batteries gave me consistent results in all my devices.",
    "I love how lightweight and efficient these batteries are.",
    "Not suitable for high-drain devices like my DSLR camera.",
    "These are the most reliable batteries I’ve used so far.",
    "Energizer outperformed every other brand I’ve tried.",
    "The batteries arrived quickly and worked as expected. Great job!",
    "I used these in my flashlight, and they lasted for weeks.",
    "The batteries drained faster than I hoped. Not the best value.",
    "I’ve never had issues with Energizer batteries in my remotes.",
    "Super reliable for travel. These batteries saved me during emergencies.",
    "These lasted much longer than other brands I’ve tried in my devices.",
    "Good quality, but they are slightly overpriced for regular use.",
    "Energizer is my trusted brand for all my high-tech gadgets.",
    "I had a positive experience using these in my gaming console.",
    "Durable and consistent performance. Highly recommended."
]

# Function to analyze sentiment
def analyze_reviews(reviews, company_name):
    results = []
    for review in reviews:
        sentiment = sia.polarity_scores(review)
        results.append({
            "Review": review,
            "Positive": sentiment['pos'],
            "Negative": sentiment['neg'],
            "Neutral": sentiment['neu'],
            "Compound": sentiment['compound'],
            "Company": company_name
        })
    return results

# Analyze sentiment for both companies
varta_results = analyze_reviews(varta_reviews, "VARTA")
energizer_results = analyze_reviews(energizer_reviews, "Energizer")

# Combine results
all_results = varta_results + energizer_results

# Display results
for result in all_results:
    print(f"Company: {result['Company']}")
    print(f"Review: {result['Review']}")
    print(f"Sentiment: Positive={result['Positive']}, Negative={result['Negative']}, Neutral={result['Neutral']}, Compound={result['Compound']}")
    print("-" * 60)

# Categorize and count sentiment
sentiment_counts = {"VARTA": {"Positive": 0, "Negative": 0, "Neutral": 0},
                    "Energizer": {"Positive": 0, "Negative": 0, "Neutral": 0}}

for result in all_results:
    sentiment_category = (
        "Positive" if result['Compound'] > 0.05 else
        "Negative" if result['Compound'] < -0.05 else
        "Neutral"
    )
    sentiment_counts[result["Company"]][sentiment_category] += 1

print("\nSentiment Summary:")
for company, counts in sentiment_counts.items():
    print(f"{company} - Positive: {counts['Positive']}, Neutral: {counts['Neutral']}, Negative: {counts['Negative']}")


# Data for plotting (sentiment counts)
labels = ["Positive", "Neutral", "Negative"]
varta_values = [sentiment_counts["VARTA"][label] for label in labels]
energizer_values = [sentiment_counts["Energizer"][label] for label in labels]

x = np.arange(len(labels))  # Label locations
width = 0.35  # Bar width

# Create the plot
fig, ax = plt.subplots(figsize=(10, 6))

# Add bars for VARTA and Energizer
varta_bars = ax.bar(x - width/2, varta_values, width, label="VARTA", color="#1f77b4", edgecolor="black", alpha=0.8)
energizer_bars = ax.bar(x + width/2, energizer_values, width, label="Energizer", color="#ff7f0e", edgecolor="black", alpha=0.8)

# Add titles and labels
ax.set_title("Sentiment Analysis of Customer Reviews", fontsize=16, fontweight='bold')
ax.set_xlabel("Sentiment Category", fontsize=14)
ax.set_ylabel("Number of Reviews", fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=12)
ax.legend(fontsize=12, loc="upper right")

# Add bar value annotations
def add_value_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # Offset for text position
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

add_value_labels(varta_bars)
add_value_labels(energizer_bars)

# Add gridlines for better readability
ax.yaxis.grid(True, linestyle="--", alpha=0.6)

# Show the plot
plt.tight_layout()
plt.show()
"""# Visualize sentiment counts
labels = ["Positive", "Neutral", "Negative"]
varta_values = [sentiment_counts["VARTA"][label] for label in labels]
energizer_values = [sentiment_counts["Energizer"][label] for label in labels]

x = range(len(labels))
plt.bar(x, varta_values, width=0.4, label="VARTA", color="blue", alpha=0.7)
plt.bar([i + 0.4 for i in x], energizer_values, width=0.4, label="Energizer", color="green", alpha=0.7)

plt.xlabel("Sentiment")
plt.ylabel("Counts")
plt.title("Sentiment Analysis of Customer Reviews")
plt.xticks([i + 0.2 for i in x], labels)
plt.legend()
plt.show()"""
