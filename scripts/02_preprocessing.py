#!/usr/bin/env python
# coding: utf-8

# # Notebook 02: Text Preprocessing
# 
# **Author:** Karan Raman  
# **Purpose:** Clean and filter raw forum posts to produce an analysis-ready dataset  
# **Run after:** 01_eda.ipynb
# **Input:** data/processed/posts_eda.parquet  
# **Output:** data/processed/posts_cleaned.parquet

# ## 0. Imports & configuration

# In[1]:


import pandas as pd
import numpy as np
import re
import html
import matplotlib as mpl
import matplotlib.pyplot as plt
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

print('Import successful')


# ## 1. Load data from EDA

# In[2]:


df = pd.read_parquet(DATA_PROCESSED / 'posts_eda.parquet')
print(f'Loaded {len(df):,} posts')
df.head(3)


# ## 2. Inspect raw text
# Sample posts before cleaning to understand what we are working with.

# In[3]:


pd.set_option('display.max_colwidth', 300)
df[['forum_post_id', 'Text', 'post_type']].sample(10, random_state=42)


# ## 3. Noise Analysis and Text cleaning

# In[4]:


# Noise analysis
# scan raw text before cleaning to understand what needs to be handled
noise_patterns = {
    'HTML tags'          : r'<[^>]+>',
    'Redacted name tags' : r'<nameRedac_[^>]*>',
    'Anon screen name'   : r'<anon_screen_name_redacted>',
    'URLs'               : r'http\S+|www\.\S+',
    'Email addresses'    : r'\S+@\S+',
    'HTML entities'      : r'&[a-z]+;',
    '_x0007_ artefacts'  : r'_x0007_',
    'Markdown links'     : r'\[.*?\]\(.*?\)',
    'Broken URL fragments': r'[a-f0-9]{8,}&\S+',
    'Non-ASCII chars'    : r'[^\x00-\x7F]',
}

print('Noise pattern frequency (raw text):')
for label, pattern in noise_patterns.items():
    count = df['Text'].str.contains(pattern, regex=True, na=False).sum()
    pct   = count / len(df) * 100
    print(f'  {label:<25}: {count:>6,} ({pct:.1f}%)')


# In[5]:


# Text cleaning function
# handles all noise patterns identified in the analysis above

def clean_text(text):
    if not isinstance(text, str) or text.strip().lower() in ('nan', 'none', 'null', ''):
        return ''

    # nested redacted name tags must go first before general HTML removal
    text = re.sub(r'<nameRedac_<[^>]*>>', '', text)
    text = re.sub(r'<nameRedac_[^>]*>', '', text)

    text = html.unescape(text)
    text = re.sub(r'<[^>]+>', ' ', text)

    # urls and links
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    text = re.sub(r'\[([^\]]+)\]\[[^\]]*\]', r'\1', text)

    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[a-f0-9]{8,}&\S+', '', text)
    text = text.replace('_x0007_', ' ')
    text = re.sub(r'[^\x00-\x7F]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text

df['text_clean'] = df['Text'].apply(clean_text)
print('Cleaning done.')


# ## 4. Filter short posts
# Posts with fewer than five words are excluded as they carry no reliable politeness signal.

# In[6]:


# word count after cleaning to decide the minimum length threshold

df['word_count'] = df['text_clean'].str.split().str.len()
print(df['word_count'].describe().round(1).to_string())
print()

for t in [0, 3, 5, 10]:
    n = (df['word_count'] <= t).sum()
    print(f'posts with  <= {t:>2} words: {n:,} ({n/len(df)*100:.1f}%)')


# In[7]:


# 5 words minimum is standard in the politeness literature
MIN_WORDS = 5

n_before = len(df)
df = df[df['word_count'] >= MIN_WORDS].copy()
n_after = len(df)

print(f'Removed {n_before - n_after:,} posts shorter than {MIN_WORDS} words')
print(f'Remaining posts: {n_after:,}')


# ## 5. Standardise columns
# Rename and recast columns for consistency throughout the rest of the project.

# In[8]:


# standardise column names
df = df.rename(columns={
    'Text'      : 'text_raw',
    'Opinion'   : 'is_opinion',
    'Question'  : 'is_question',
    'Answer'    : 'is_answer',
    'Sentiment' : 'sentiment',
    'Confusion' : 'confusion',
    'Urgency'   : 'urgency',
    'CourseType': 'course_type',
})

# cast binary flags to integer
for col in ['is_opinion', 'is_question', 'is_answer', 'anonymous', 'anonymous_to_peers']:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

# parse timestamps
df['created_at'] = pd.to_datetime(df['created_at'], utc=True, errors='coerce')

print('Column types after standardising:')
for col in df.columns:
    print(f'  {col}: {df[col].dtype}')


# ## 7. Response timing

# In[9]:


# compute response timing for each post relative to thread start

for col in ['thread_start_time', 'response_time_hours', 'response_time_bin']:
    if col in df.columns:
        df = df.drop(columns=[col])

thread_first = (
    df.groupby('comment_thread_id')['created_at']
    .min()
    .rename('thread_start_time')
)
df = df.merge(thread_first, on='comment_thread_id', how='left')

df['response_time_hours'] = (
    df['created_at'] - df['thread_start_time']
).dt.total_seconds() / 3600

bins   = [-0.01, 1, 24, 168, float('inf')]
labels = ['< 1 hour', '1-24 hours', '1-7 days', '> 7 days']
df['response_time_bin'] = pd.cut(df['response_time_hours'], bins=bins, labels=labels)

print(f'Response time computed for {df["response_time_hours"].notna().sum():,} posts')
print(df['response_time_bin'].value_counts().reindex(labels).to_string())


# ## 8. Select final columns

# In[10]:


# select final columns and save clean dataset

KEEP_COLS = [
    'forum_post_id', 'comment_thread_id', 'forum_uid',
    'course_display_name', 'course_type', 'created_at', 'post_type',
    'is_opinion', 'is_question', 'is_answer',
    'anonymous', 'anonymous_to_peers',
    'up_count', 'reads', 'sentiment', 'confusion', 'urgency',
    'text_raw', 'text_clean', 'word_count',
    'thread_start_time', 'response_time_hours', 'response_time_bin',
]

df_clean = df[KEEP_COLS].copy()
print(f'Final shape: {df_clean.shape}')
df_clean.head(3)


# ## 8. Sanity check
# Quick check to confirm the cleaned dataset looks as expected before saving.

# In[11]:


fig, ax = plt.subplots(figsize=(10, 4))
df_clean['word_count'].clip(upper=300).plot(kind='hist', bins=50, ax=ax, color=BLUE, edgecolor='white')
ax.set_title('Word count distribution after cleaning')
ax.set_xlabel('Word count')
ax.set_ylabel('Number of posts')
plt.tight_layout()
plt.savefig(DATA_OUT / 'fig_wordcount_clean.png')
plt.show()


# In[12]:


print('post_type breakdown:')
print(df_clean['post_type'].value_counts())

print('\ncourse_type breakdown:')
print(df_clean['course_type'].value_counts())


# ## 9. Save clean dataset

# In[13]:


# save clean dataset
for col in df_clean.select_dtypes(include=['object', 'str']).columns:
    df_clean[col] = df_clean[col].astype(str)

df_clean.to_parquet(DATA_PROCESSED / 'posts_clean.parquet', index=False)
print(f'Saved {len(df_clean):,} posts to posts_clean.parquet')

