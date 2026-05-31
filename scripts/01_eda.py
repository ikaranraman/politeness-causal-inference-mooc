#!/usr/bin/env python
# coding: utf-8

# # Notebook 01: Exploratory Data Analysis
# 
# **Project:** Quantifying Politeness in Online Educational Forums  
# **Purpose:** First-pass exploration of the Stanford MOOC forum dataset to understand structure, distributions and data quality before modelling  
# **Author:** Karan Raman  
# **Input:** data/raw/stanfordMOOCForumPostsSet.xlsx 
# **Output:** data/processed/posts_eda.parquet

# ## 0. Imports & configuration

# In[1]:


import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from pathlib import Path

sns.set_theme(style='whitegrid', font_scale=1.1)
mpl.rcParams.update({
    'figure.figsize'    : (10, 6),
    'figure.dpi'        : 120,
    'savefig.dpi'       : 300,
    'savefig.bbox'      : 'tight',
    'savefig.facecolor' : 'white',
    'font.size'         : 11,
    'axes.titlesize'    : 13,
    'axes.spines.top'   : False,
    'axes.spines.right' : False,
    'grid.alpha'        : 0.3,
})

BLUE, RED, GREEN, GREY = '#2166AC', '#D6604D', '#1A9641', '#878787'
sns.set_palette([BLUE, RED, GREEN, GREY, '#F4A582'])

DATA_RAW       = Path('../data/raw')
DATA_PROCESSED = Path('../data/processed')
DATA_OUT       = Path('../data/outputs')
DATA_OUT.mkdir(parents=True, exist_ok=True)

print('Import successful')


# 
# ## 1. Load the data

# In[2]:


# Load dataset
df = pd.read_excel(DATA_RAW / 'stanfordMOOCForumPostsSet.xlsx')

print(f'Rows: {len(df):,}')
print(f'Columns: {df.shape[1]}')
print(f'\nColumn names:')
for col in df.columns:
    print(f'  {col}: {df[col].dtype}')


# In[3]:


# Preview first three rows
df.head(3)


# ## 2. Data quality: missing values, duplicates, empty/short posts

# In[4]:


# Missing values per column
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_summary = pd.DataFrame({'missing_count': missing, 'missing_pct': missing_pct})
missing_summary[missing_summary.missing_count > 0].sort_values('missing_pct', ascending=False)


# In[5]:


# Duplicate posts
dupes = df.duplicated(subset='forum_post_id').sum()
print(f'Duplicate post IDs: {dupes}')

# Empty / very short text posts
df['text_length'] = df['Text'].fillna('').str.split().str.len()
print(f'Posts with 0 words: {(df.text_length == 0).sum()}')
print(f'Posts with <5 words: {(df.text_length < 5).sum()}')


# 
# ## 3. Post type distribution
# Understanding the balance of opinions, questions and answers.

# In[6]:


df = df.rename(columns={
    'Opinion(1/0)': 'Opinion',
    'Question(1/0)': 'Question',
    'Answer(1/0)': 'Answer',
    'Sentiment(1-7)': 'Sentiment',
    'Confusion(1-7)': 'Confusion',
    'Urgency(1-7)': 'Urgency'
})


# In[7]:


# Structural post type
print('Structural post type distribution:')
print(df['post_type'].value_counts().to_string())


# In[8]:


# Content flags (what a post is about (non-exclusive))
for col in ['Opinion', 'Question', 'Answer']:
    n = int(df[col].sum())
    pct = n / len(df) * 100
    print(f'{col:<10}: {n:,} ({pct:.1f}%)')


# In[9]:


fig, axes = plt.subplots(1, 2, figsize=(12, 4))
type_counts = {col: int(df[col].sum()) for col in ['Opinion', 'Question', 'Answer']}

def add_labels(ax):
    for bar in ax.patches:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 100,
                f'{int(bar.get_height()):,}',
                ha='center', va='bottom', fontsize=9)

pd.Series(type_counts).plot(kind='bar', ax=axes[0], color=[BLUE, GREY, GREEN])
axes[0].set_title('Content flags per post')
axes[0].set_ylabel('Number of posts')
axes[0].tick_params(axis='x', rotation=0)
add_labels(axes[0])

df['post_type'].dropna().value_counts().plot(kind='bar', ax=axes[1], color=['#8856a7', '#F4A582'])
axes[1].set_title('Post position in thread')
axes[1].set_ylabel('Number of posts')
axes[1].set_xlabel('Post Type')
axes[1].tick_params(axis='x', rotation=0)
add_labels(axes[1])

plt.tight_layout()
plt.savefig(DATA_OUT / 'fig_post_type_distribution.png')
plt.show()


# 
# ## 4. Course distribution
# Understanding how posts spread across courses and course types.

# In[10]:


# Note: counts below are from raw data before cleaning
print(f'Unique course display names : {df.course_display_name.nunique()}')
print(f'Unique user IDs             : {df.forum_uid.nunique():,}')
print(f'Unique thread IDs           : {df.comment_thread_id.nunique():,}')

print('\nCourse type distribution (raw data):')
print(df['CourseType'].value_counts().to_string())


# In[11]:


top_courses = df['course_display_name'].value_counts().head(10).sort_values()

fig, ax = plt.subplots(figsize=(10, 5))
top_courses.plot(kind='barh', ax=ax, color=BLUE)
ax.set_title('Top 10 courses by post count')
ax.set_xlabel('Number of posts')
ax.set_ylabel('')
plt.tight_layout()
plt.savefig(DATA_OUT / 'fig_top_courses.png')
plt.show()


# ## 5. Engagement metrics
# Exploring upvotes, reads, and anonymity patterns.

# In[12]:


for col, label in [('up_count', 'Upvotes'), ('reads', 'Reads')]:
    print(f'{label}:')
    print(df[col].describe().round(2).to_string())
    print()

print(f'Anonymous posts     : {int(df.anonymous.sum()):,} ({df.anonymous.mean()*100:.1f}%)')
print(f'Anonymous to peers  : {int(df.anonymous_to_peers.sum()):,} ({df.anonymous_to_peers.mean()*100:.1f}%)')


# 
# ## 6. Existing annotation distributions
# Sentiment, confusion, and urgency are pre-labelled on a 1–7 scale

# In[13]:


print('Annotation score statistics:')
print(df[['Sentiment', 'Confusion', 'Urgency']].describe().round(2).to_string())


# In[14]:


corr = df[['Sentiment', 'Confusion', 'Urgency']].corr()

fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='Blues',
            square=True, linewidths=0.5, ax=ax, vmin=-1, vmax=1)
ax.set_title('Correlation between annotation scores')
plt.tight_layout()
plt.savefig(DATA_OUT / 'fig_annotation_correlation.png')
plt.show()


# ## 7. Text length distribution

# In[15]:


df['word_count'] = df['Text'].str.split().str.len()

print('Word count per post:')
print(df['word_count'].describe().round(1).to_string())

fig, ax = plt.subplots(figsize=(10, 4))
df['word_count'].clip(upper=500).plot(kind='hist', bins=50, ax=ax,
                                      color=BLUE, edgecolor='white')
ax.set_title('Post length distribution (clipped at 500 words)')
ax.set_xlabel('Word count')
ax.set_ylabel('Number of posts')
plt.tight_layout()
plt.savefig(DATA_OUT / 'fig_text_length.png')
plt.show()


# ## 8. Temporal distribution
# When are posts being made? Useful for understanding response timing patterns.

# In[16]:


df['created_at']  = pd.to_datetime(df['created_at'])
df['month']       = df['created_at'].dt.to_period('M')
df['hour_of_day'] = df['created_at'].dt.hour
df['day_of_week'] = df['created_at'].dt.day_name()

print(f'Date range  : {df.created_at.min().date()} to {df.created_at.max().date()}')
print(f'Months      : {df.month.nunique()}')
print(f'Posts/month : {len(df) / df.month.nunique():.0f} avg')


# In[17]:


# Posts per month
fig, ax = plt.subplots(figsize=(12, 4))
monthly = df.groupby('month').size()
monthly.index = monthly.index.to_timestamp()
ax.plot(monthly.index, monthly.values, color=BLUE, marker='o', markersize=4)
for x, y in zip(monthly.index, monthly.values):
    ax.annotate(f'{y:,}', (x, y), textcoords='offset points',
                xytext=(0, 6), ha='center', fontsize=7)
ax.set_title('Posts per month')
ax.set_ylabel('Number of posts')
ax.xaxis.set_major_formatter(mpl.dates.DateFormatter('%b %y'))
ax.tick_params(axis='x', rotation=30)
plt.tight_layout()
plt.savefig(DATA_OUT / 'fig_posts_per_month.png')
plt.show()


# In[18]:


# Posts by hour of day
fig, ax = plt.subplots(figsize=(12, 4))
hourly = df['hour_of_day'].value_counts().sort_index()
ax.fill_between(hourly.index, hourly.values, color=GREEN, alpha=0.6)
ax.plot(hourly.index, hourly.values, color=GREEN)
for x, y in zip(hourly.index, hourly.values):
    ax.annotate(f'{y:,}', (x, y), textcoords='offset points',
                xytext=(0, 6), ha='center', fontsize=7)
ax.set_title('Posts by hour of day')
ax.set_xlabel('Hour (UTC)')
ax.set_ylabel('Number of posts')
ax.set_xticks(range(0, 24))
plt.tight_layout()
plt.savefig(DATA_OUT / 'fig_posts_by_hour.png')
plt.show()


# ## 9. Role Label Investigation

# In[19]:


print('Role label check')
role_cols = [c for c in df.columns if any(
    kw in c.lower() for kw in ['role', 'instructor', 'student', 'staff', 'teacher']
)]
print(f'Role columns found : {role_cols if role_cols else "none"}')
print(f'Unique user IDs    : {df["forum_uid"].nunique():,} anonymised')
print(f'post_type values   : {df["post_type"].unique().tolist()}')
print(f'Anonymous posts    : {int(df["anonymous"].sum()):,}')
print('No role metadata present in this dataset.')


# ## 10. Response Timing Exploration

# In[20]:


df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')

thread_first = (df.groupby('comment_thread_id')['created_at']
                .min().rename('thread_start_time'))
df = df.merge(thread_first, on='comment_thread_id', how='left')
df['response_time_hours'] = (
    df['created_at'] - df['thread_start_time']
).dt.total_seconds() / 3600

bins   = [-0.01, 1, 24, 168, float('inf')]
labels = ['< 1 hr', '1-24 hrs', '1-7 days', '> 7 days']
df['response_time_bin'] = pd.cut(df['response_time_hours'], bins=bins, labels=labels)

thread_sizes = df.groupby('comment_thread_id').size()
print(f'Unique threads     : {thread_sizes.shape[0]:,}')
print(f'Single post threads: {(thread_sizes == 1).sum():,}')
print(f'Multi post threads : {(thread_sizes >= 2).sum():,}')
print(df['response_time_hours'].describe().round(2).to_string())


# In[21]:


valid_rt = df['response_time_hours'].dropna()
valid_rt = valid_rt[valid_rt > 0].clip(upper=168)

fig, ax = plt.subplots(figsize=(10, 4))
valid_rt.plot(kind='hist', bins=40, ax=ax, color=BLUE, edgecolor='white')
ax.axvline(valid_rt.median(), color='black', linestyle='--',
           label=f'median = {valid_rt.median():.1f} hrs')
ax.set_title('Response time distribution (clipped at 1 week)')
ax.set_xlabel('Hours since thread start')
ax.set_ylabel('Number of posts')
ax.legend()
plt.tight_layout()
plt.savefig(DATA_OUT / 'fig_response_timing.png')
plt.show()


# In[22]:


rt_counts = df['response_time_bin'].value_counts().reindex(labels)
fig, ax = plt.subplots(figsize=(8, 4))
rt_counts.plot(kind='bar', ax=ax, color=GREEN, edgecolor='white')
ax.set_title('Posts by response time category')
ax.set_ylabel('Number of posts')
ax.tick_params(axis='x', rotation=20)

for bar in ax.patches:
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 50,
            f'{int(bar.get_height()):,}',
            ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(DATA_OUT / 'fig_response_bins.png')
plt.show()


# ## 11. Save output file

# In[23]:


# Summary of notebook
print(f'Total posts        : {len(df):,}')
print(f'Date range         : {df.created_at.min().date()} to {df.created_at.max().date()}')
print(f'Unique users       : {df.forum_uid.nunique():,}')
print(f'Anonymous posts    : {int(df.anonymous.sum()):,}')
print(f'Multi-post threads : {(df.groupby("comment_thread_id").size() >= 2).sum():,}')
print(f'Median response    : {df["response_time_hours"][df["response_time_hours"]>0].median():.1f} hrs')


# In[24]:


# Save a clean copy of the dataframe 
# Convert Text to string to fix mixed types
df['Text'] = df['Text'].astype(str)


# In[25]:


# Fix all object columns with mixed types
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].astype(str)

# Save
df.to_parquet(DATA_OUT / 'posts_eda.parquet', index=False)
print('Saved to data/processed/posts_eda.parquet')

