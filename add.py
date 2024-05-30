import matplotlib.pyplot as plt

# Define the sections and their frequencies
sections = [
    "Introduction to Anthropology",
    "Sub-fields of Anthropology",
    "Human Culture",
    "Cultural Unity and Variations",
    "Evaluating Cultural Differences",
    "Culture Change",
    "Marriage, Family, and Kinship"
]
frequencies = [6, 4, 5, 3, 3, 4, 4]

# Create the bar chart
plt.figure(figsize=(10, 6))
plt.bar(sections, frequencies, color='skyblue')
plt.xlabel('Sections')
plt.ylabel('Frequency')
plt.title('Distribution of Topics in Anthropology Document')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Show the plot
plt.show()

