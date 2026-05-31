#!/usr/bin/env python
# coding: utf-8

# # 06 LLM Intervention
# **Project:** Quantifying Politeness in Online Educational Forums  
# **Author:** Karan Raman  
# **Purpose:** Explore the feasibility of using a free local LLM to rewrite
# impolite or neutral forum posts into more polite versions.
# 
# **Evaluation dimensions:**
# 1. Politeness gain, does the rewritten post score higher with ConvoKit?
# 2. Semantic preservation, does the rewrite keep the original meaning?
# 3. Qualitative inspection, does it read naturally?
# 
# **Run After:** 05_causal_analysis.ipynb
# **Input:** data/processed/posts_analysis.parquet
# **Output:** data/outputs/llm_top_improved.csv

# ## 0. Install & imports

# In[2]:


import subprocess
subprocess.run(['pip', 'install', 'sentence-transformers', '-q'], check=True)
print('sentence-transformers installed.')


# In[3]:


import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import time
from pathlib import Path
from convokit import PolitenessStrategies
from transformers import pipeline
import spacy

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

DATA_PROCESSED = Path('../data/processed')
DATA_OUT       = Path('../data/outputs')
DATA_OUT.mkdir(parents=True, exist_ok=True)

N_SAMPLES   = 100
RANDOM_SEED = 42

nlp = spacy.load('en_core_web_sm')
ps  = PolitenessStrategies()

print('Setup complete.')


# ## 1. Load rewriting model
# Loading the T5-based paraphrasing model used to rewrite impolite posts into more polite versions.

# In[4]:


# load T5 paraphrasing model
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

model_name = 'humarin/chatgpt_paraphraser_on_T5_base'
tokenizer  = T5Tokenizer.from_pretrained(model_name)
model      = T5ForConditionalGeneration.from_pretrained(model_name)
model.eval()

def rewrite_post(text):
    prompt = f"make this more polite and respectful: {str(text)[:400]}"
    inputs = tokenizer(prompt, return_tensors='pt',
                       max_length=256, truncation=True, padding=True)
    with torch.no_grad():
        outputs = model.generate(**inputs, num_beams=4,
                                 max_length=256, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

print(f'Test: {rewrite_post("Give me the answer now.")}')


# ## 2. Load and sample impolite/neutral posts

# In[5]:


# load data and sample impolite/neutral posts for rewriting

df = pd.read_parquet(DATA_PROCESSED / 'posts_scored.parquet')
df['politeness_score'] = pd.to_numeric(df['politeness_score'], errors='coerce')

candidates = df[df['politeness_score'] <= 0].copy()
sample     = candidates.sample(n=N_SAMPLES, random_state=RANDOM_SEED).reset_index(drop=True)

print(f'Candidates : {len(candidates):,}')
print(f'Sample size: {len(sample)}')
print(sample['politeness_score'].value_counts().sort_index().to_string())


# ## 3. Rewrite posts with HuggingFace T5

# In[6]:


# rewrite all sampled posts
rewritten_texts = []
for i, row in sample.iterrows():
    rewritten_texts.append(rewrite_post(row['text_clean']))
    if (i + 1) % 10 == 0:
        print(f'  {i+1}/{len(sample)} done')

sample['text_rewritten'] = rewritten_texts
print(f'Successfully rewritten: {sample["text_rewritten"].notna().sum()}/{len(sample)}')


# In[7]:


pd.set_option('display.max_colwidth', 200)
sample[['text_clean', 'text_rewritten', 'politeness_score']].head(5)


# ## 4. Re-score rewritten posts with ConvoKit

# In[8]:


# feature lists needed for scoring
positive = [
    'feature_politeness_==Please==',
    'feature_politeness_==Please_start==',
    'feature_politeness_==Hedges==',
    'feature_politeness_==HASHEDGE==',
    'feature_politeness_==Gratitude==',
    'feature_politeness_==Deference==',
    'feature_politeness_==Apologizing==',
    'feature_politeness_==1st_person_pl.==',
    'feature_politeness_==Indirect_(btw)==',
    'feature_politeness_==Indirect_(greeting)==',
    'feature_politeness_==HASPOSITIVE==',
]

negative = [
    'feature_politeness_==Direct_question==',
    'feature_politeness_==Direct_start==',
    'feature_politeness_==2nd_person==',
    'feature_politeness_==2nd_person_start==',
    'feature_politeness_==HASNEGATIVE==',
]

# score original and rewritten posts with ConvoKit
def score_convokit(text):
    try:
        strategies = ps.transform_utterance(str(text), spacy_nlp=nlp).meta['politeness_strategies']
        pos = sum(strategies.get(k, 0) for k in positive)
        neg = sum(strategies.get(k, 0) for k in negative)
        return pos - neg
    except:
        return np.nan

rewritten_valid = sample.dropna(subset=['text_rewritten']).copy()
rewritten_valid['politeness_original']  = rewritten_valid['text_clean'].apply(score_convokit)
rewritten_valid['politeness_rewritten'] = rewritten_valid['text_rewritten'].apply(score_convokit)
rewritten_valid['improvement']          = rewritten_valid['politeness_rewritten'] - rewritten_valid['politeness_original']

print(f'Mean original score  : {rewritten_valid["politeness_original"].mean():.3f}')
print(f'Mean rewritten score : {rewritten_valid["politeness_rewritten"].mean():.3f}')
print(f'Mean improvement     : {rewritten_valid["improvement"].mean():.3f}')


# ## 5. Semantic preservation cosine similarity
# Checks whether the rewrite kept the original meaning.

# In[9]:


import logging
import warnings
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

logging.getLogger('sentence_transformers').setLevel(logging.ERROR)
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.disable(logging.WARNING)
warnings.filterwarnings('ignore')

st_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

orig_emb = st_model.encode(rewritten_valid['text_clean'].tolist(), batch_size=32, convert_to_numpy=True)
rew_emb  = st_model.encode(rewritten_valid['text_rewritten'].tolist(), batch_size=32, convert_to_numpy=True)

similarities = [cosine_similarity([o], [r])[0][0] for o, r in zip(orig_emb, rew_emb)]
rewritten_valid['semantic_similarity'] = similarities

print(f'Mean similarity    : {np.mean(similarities):.3f}')
print(f'Posts with sim>0.8 : {(np.array(similarities) > 0.8).sum()}/{len(similarities)}')
print(f'Posts with sim>0.9 : {(np.array(similarities) > 0.9).sum()}/{len(similarities)}')


# ## 6. Visualise results

# In[15]:


# scatter plot: original vs rewritten politeness scores
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(rewritten_valid['politeness_original'],
           rewritten_valid['politeness_rewritten'],
           alpha=0.6, color=BLUE)
lim = max(abs(rewritten_valid['politeness_original'].min()),
          abs(rewritten_valid['politeness_rewritten'].max())) + 0.5
ax.plot([-lim, lim], [-lim, lim], 'k--', linewidth=1, label='no change')
ax.set_xlabel('Original score')
ax.set_ylabel('Rewritten score')
ax.set_title('Politeness: original vs rewritten')
ax.legend()
plt.tight_layout()
plt.savefig(DATA_OUT / 'fig_llm_scatter.png')
plt.show()


# In[16]:


# improvement distribution
fig, ax = plt.subplots(figsize=(7, 5))
rewritten_valid['improvement'].plot(kind='hist', bins=15, ax=ax, color=GREEN, edgecolor='white')
ax.axvline(rewritten_valid['improvement'].mean(), color='black', linestyle='--',
           label=f"mean={rewritten_valid['improvement'].mean():.2f}")
ax.axvline(0, color=GREY, linestyle='-', linewidth=0.8)
ax.set_title('Improvement distribution')
ax.set_xlabel('Score change')
ax.legend()
plt.tight_layout()
plt.savefig(DATA_OUT / 'fig_llm_improvement.png')
plt.show()


# In[17]:


# semantic similarity distribution
fig, ax = plt.subplots(figsize=(7, 5))
pd.Series(similarities).plot(kind='hist', bins=15, ax=ax, color=GREY, edgecolor='white')
ax.axvline(np.mean(similarities), color='black', linestyle='--',
           label=f'mean={np.mean(similarities):.2f}')
ax.set_title('Semantic similarity')
ax.set_xlabel('Cosine similarity')
ax.legend()
plt.tight_layout()
plt.savefig(DATA_OUT / 'fig_llm_similarity.png')
plt.show()


# ## 7. Qualitative inspection
# Read the best and worst rewrites to judge naturalness.

# In[11]:


pd.set_option('display.max_colwidth', 200)
cols = ['text_clean', 'text_rewritten', 'politeness_original',
        'politeness_rewritten', 'improvement', 'semantic_similarity']

print('Top 5 most improved:')
display(rewritten_valid.nlargest(5, 'improvement')[cols].reset_index(drop=True))

print('Top 5 lowest similarity:')
display(rewritten_valid.nsmallest(5, 'semantic_similarity')[cols].reset_index(drop=True))


# In[12]:


# save top 5 most improved and lowest similarity rewrites

cols = ['text_clean', 'text_rewritten', 'politeness_original',
        'politeness_rewritten', 'improvement', 'semantic_similarity']

top_improved = rewritten_valid.nlargest(5, 'improvement')[cols].reset_index(drop=True)
top_improved.to_csv(DATA_OUT / 'llm_top_improved.csv', index=False)

low_similarity = rewritten_valid.nsmallest(5, 'semantic_similarity')[cols].reset_index(drop=True)
low_similarity.to_csv(DATA_OUT / 'llm_low_similarity.csv', index=False)

print(f'Saved top 5 most improved to llm_top_improved.csv')
print(f'Saved top 5 lowest similarity to llm_low_similarity.csv')


# ## 8. Summary statistics

# In[13]:


# LLM intervention summary
n_improved = (rewritten_valid['improvement'] > 0).sum()
n_same     = (rewritten_valid['improvement'] == 0).sum()
n_degraded = (rewritten_valid['improvement'] < 0).sum()
total      = len(rewritten_valid)

print(f'Posts rewritten          : {total}')
print(f'Mean original score      : {rewritten_valid["politeness_original"].mean():.3f}')
print(f'Mean rewritten score     : {rewritten_valid["politeness_rewritten"].mean():.3f}')
print(f'Mean improvement         : {rewritten_valid["improvement"].mean():.3f}')
print(f'Improved                 : {n_improved} ({n_improved/total*100:.1f}%)')
print(f'Unchanged                : {n_same} ({n_same/total*100:.1f}%)')
print(f'Degraded                 : {n_degraded} ({n_degraded/total*100:.1f}%)')
print(f'Mean similarity          : {np.mean(similarities):.3f}')
print(f'Posts with sim > 0.8     : {(np.array(similarities) > 0.8).sum()}/{total}')


# In[14]:


# save intervention results

for col in rewritten_valid.select_dtypes(include=['object', 'str']).columns:
    rewritten_valid[col] = rewritten_valid[col].astype(str)

rewritten_valid.to_parquet(DATA_PROCESSED / 'posts_intervention.parquet', index=False)
print(f'Saved {len(rewritten_valid):,} posts to posts_intervention.parquet')

