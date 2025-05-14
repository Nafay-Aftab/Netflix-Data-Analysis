import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Data Loading and Preprocessing ---
# Load the Netflix dataset
df = pd.read_csv('Netflix.csv', encoding='utf-8', lineterminator='\n')

# Convert 'Release_Date' to just the year
df['Release_Date'] = pd.to_datetime(df['Release_Date'], errors='coerce').dt.year

# Drop unnecessary columns that don't add value to the analysis
df.drop(['Overview', 'Original_Language', 'Poster_Url'], axis=1, inplace=True)

# --- Vote Labeling Based on 'Vote_Average' ---
# Define a function to label movies based on their vote average
def label_rating(score):
    if score >= 7.1:
        return 'Popular'
    elif score >= 6.5:
        return 'Above Average'
    elif score >= 5.9:
        return 'Average'
    else:
        return 'Below Average'

# Apply the function to create a new column 'Vote_Label'
df['Vote_Label'] = df['Vote_Average'].apply(label_rating)

# Reorder columns to place 'Vote_Label' right after 'Vote_Average'
cols = df.columns.tolist()
idx = cols.index('Vote_Average')
cols.remove('Vote_Label')
cols.insert(idx + 1, 'Vote_Label')
df = df[cols]

# --- Genre Exploding ---
# Split the 'Genre' column by commas and explode the list so each genre gets its own row
df['Genre'] = df['Genre'].str.split(', ')
df = df.explode('Genre').reset_index(drop=True)

# Convert the 'Genre' column to a categorical type for efficiency
df['Genre'] = df['Genre'].astype('category')

# --- Visualization 1: Genre Distribution ---
# Create a countplot to visualize the distribution of genres
# This will show how often each genre appears in the dataset
plt.figure(figsize=(12, 6))
sns.set_style('whitegrid')
bars = sns.countplot(
    y='Genre',
    data=df,
    palette='viridis',  # Applied a different color palette
    order=df['Genre'].value_counts().index
)

# Add bar labels showing the count of each genre
for container in bars.containers:
    bars.bar_label(container)

plt.title('Genre Column Distribution', fontsize=18)
plt.xlabel('Count', fontsize=14)
plt.ylabel('Genre', fontsize=14)
plt.tight_layout()
plt.savefig('genre_column.png')  # Save the plot as an image
plt.show()

# --- Visualization 2: Movie Distribution by Vote Label ---
# Create a bar plot to visualize the distribution of movies by vote label (Popular, Above Average, etc.)
# First, drop duplicate movie titles to avoid overcounting
plt.figure(figsize=(10, 5))
sns.set_style('whitegrid')

# Count the vote labels after dropping duplicate movie titles
gb = df.drop_duplicates(subset='Title')['Vote_Label'].value_counts()

bars2 = sns.barplot(x=gb.index, y=gb.values, palette='coolwarm')  # Applied a coolwarm color palette
for container in bars2.containers:
    bars2.bar_label(container)

plt.title('Movie Distribution by Vote Label', fontsize=18)
plt.xlabel('Vote Label', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.tight_layout()
plt.savefig('movie_popularity.png')  # Save the plot as an image
plt.show()

# --- Visualization 3: Top 10 Most Popular Movies ---
# Create a bar plot showing the top 10 most popular movies based on their 'Popularity' score
# Drop duplicates based on movie titles and sort them by popularity
top_10 = df.drop_duplicates(subset='Title').sort_values(by='Popularity', ascending=False).head(10)

sns.set_style('whitegrid')
plt.figure(figsize=(12, 6))

# Plot the top 10 movies with their popularity scores
bars3 = sns.barplot(x='Popularity', y='Title', data=top_10, palette='plasma')  # Applied a plasma color palette

# Add bar labels to show the popularity score
for container in bars3.containers:
    bars3.bar_label(container)

# Rotate y-ticks for better readability
plt.yticks(rotation=45, ha='right')

# Adjust layout to prevent clipping of titles
plt.tight_layout()
plt.subplots_adjust(top=0.9)

# Add titles and labels to the plot
plt.title('Top 10 Most Popular Movies on Netflix', fontsize=16)
plt.xlabel('Popularity Score', fontsize=12)
plt.ylabel('Movie Title', fontsize=12)

# Save the plot
plt.savefig('top_10_movies.png')
plt.show()

# --- Visualization 4: Distribution of Movies by Release Year ---
# Create a histogram to visualize the distribution of movies based on their release year
# This will show which years had the most movie releases
plt.figure(figsize=(12, 6))
sns.histplot(df['Release_Date'], kde=False, bins=40, color='skyblue')  # Applied a custom color
plt.title('Distribution of Movies by Release Year', fontsize=18)
plt.xlabel('Release Year', fontsize=14)
plt.ylabel('Number of Movies', fontsize=14)
plt.tight_layout()
plt.savefig('movies_by_year.png')
plt.show()
