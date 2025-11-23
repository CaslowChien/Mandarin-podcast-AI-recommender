# 🎧 Chinese Podcast Search Engine

An AI-powered semantic search engine for Chinese podcasts, built with embeddings and deployed as a static website. Search through thousands of Chinese podcast episodes using natural language queries in both English and Chinese.

![Search Demo](https://img.shields.io/badge/Status-Live-success)
![Language](https://img.shields.io/badge/Languages-中文%20%7C%20English-blue)

## 📋 Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Architecture](#architecture)
4. [Setup Guide](#setup-guide)
5. [Detailed Workflow](#detailed-workflow)
6. [Project Structure](#project-structure)
7. [Customization](#customization)
8. [Troubleshooting](#troubleshooting)

---

## 🎯 Overview

This project creates a semantic search engine for Chinese podcasts by:
1. Extracting podcast metadata from the Podcast Index database
2. Fetching episode details via the Podcast Index API
3. Generating AI embeddings for semantic search
4. Creating a fast, client-side search interface that runs entirely in the browser

**Live Demo**: [Your deployment URL here]

---

## ✨ Features

- 🔍 **Semantic Search**: Find podcasts by meaning, not just keywords
- 🌏 **Multilingual**: Search in Chinese (Simplified/Traditional) or English
- ⚡ **Fast**: Client-side search with no backend required
- 🎨 **Beautiful UI**: Modern, responsive design
- 📱 **Mobile Friendly**: Works on all devices
- 🔒 **Privacy**: All processing happens in your browser
- 🆓 **Free to Host**: Deploy on GitHub Pages or any static host

---

## 🏗️ Architecture

```
┌─────────────────┐
│ Podcast Index   │  ← SQL query for Chinese podcasts
│ Database (SQL)  │
└────────┬────────┘
         │
         ├─> Export: podcasts_ch.csv (creator data)
         │
         ▼
┌─────────────────┐
│ Python Script   │  ← Fetch episode details via API
│ (API Fetching)  │
└────────┬────────┘
         │
         ├─> Output: podcast_ch_episodes.csv
         │
         ▼
┌─────────────────┐
│ Sentence        │  ← Generate embeddings
│ Transformers    │     Model: paraphrase-multilingual-MiniLM-L12-v2
└────────┬────────┘
         │
         ├─> Output: podcast_embeddings.json (~400MB)
         │
         ▼
┌─────────────────┐
│ Sharding Script │  ← Split into 163 files (~23MB each)
│                 │
└────────┬────────┘
         │
         ├─> Output: shards/shard_0.json to shard_162.json
         │
         ▼
┌─────────────────┐
│ Static Website  │  ← HTML + Xenova Transformers.js
│ (index.html)    │     100% client-side search
└─────────────────┘
```

---

## 🚀 Setup Guide

### Prerequisites

- **Python 3.8+** with pip
- **Google Colab** (recommended) or local GPU for faster embedding generation
- **Podcast Index API credentials** (free at [podcastindex.org](https://podcastindex.org))
- **SQL database access** to Podcast Index database
- **GitHub account** (for free hosting via GitHub Pages)

### Step 1: Extract Podcast Data (SQL)

Use the following SQL query to extract Chinese podcast metadata from the Podcast Index database:

```sql
-- Database source: https://podcastindex.org/
-- Output: podcasts_ch.csv

SELECT 
    id, link, lastUpdate, lastHttpStatus, dead, itunesId, explicit, 
    itunesType, newestItemPubdate, language, oldestItemPubdate, 
    episodeCount, popularityScore, priority, createdOn, updateFrequency, 
    newestEnclosureDuration, 
    category1, category2, category3, category4, category5, 
    category6, category7, category8, category9, category10,
    
    -- Clean problematic columns (remove quotes, newlines, carriage returns)
    REPLACE(REPLACE(REPLACE(title, '"', '""'), CHAR(10), ' '), CHAR(13), ' ') AS clean_title,
    REPLACE(REPLACE(REPLACE(link, '"', '""'), CHAR(10), ' '), CHAR(13), ' ') AS clean_link,
    REPLACE(REPLACE(REPLACE(itunesAuthor, '"', '""'), CHAR(10), ' '), CHAR(13), ' ') AS clean_itunesAuthor,
    REPLACE(REPLACE(REPLACE(imageUrl, '"', '""'), CHAR(10), ' '), CHAR(13), ' ') AS clean_imageUrl,
    REPLACE(REPLACE(REPLACE(host, '"', '""'), CHAR(10), ' '), CHAR(13), ' ') AS clean_host,
    REPLACE(REPLACE(REPLACE(description, '"', '""'), CHAR(10), ' '), CHAR(13), ' ') AS description

FROM podcasts

WHERE (
    -- Match various Chinese language codes
    LOWER(language) LIKE '%zh%'
    OR LOWER(language) LIKE '%zho%'
    OR LOWER(language) LIKE '%zhhans%'
    OR LOWER(language) LIKE '%zhhant%'
    OR LOWER(language) LIKE '%zh-cn%'
    OR LOWER(language) LIKE '%zh-tw%'
    OR LOWER(language) LIKE '%zh-sg%'
    OR LOWER(language) LIKE '%zh-rtw%'
    OR LOWER(language) LIKE '%zh-chs%'
    OR LOWER(language) LIKE '%zh-cht%'
    OR LOWER(language) LIKE '%zh-t%'
)
AND lastUpdate > 1731628800;  -- Only active podcasts (updated after 2024-11-15)
```

**Output**: Save as `podcasts_ch.csv`

---

### Step 2: Fetch Episode Details

Use the Podcast Index API to get episode information for each podcast.

#### 2.1 Install Dependencies

```bash
pip install python-podcastindex pandas requests
```

#### 2.2 Run the Fetching Script

Open `Podcast.ipynb` in Google Colab or Jupyter, and run the **Fetching** section:

```python
import pandas as pd
import podcastindex
import time
import requests

# Load podcast IDs
df = pd.read_csv("podcasts_ch.csv", usecols=['id'])

# Configure API (get your keys from podcastindex.org)
config = {
    "api_key": "YOUR_API_KEY_HERE",
    "api_secret": "YOUR_API_SECRET_HERE"
}
index = podcastindex.init(config)

# Fetch episodes
all_items = []
failed_ids = []
feed_ids = df['id'].tolist()

for fid in feed_ids:
    try:
        result = index.episodesByFeedId(fid)
        
        for item in result.get("items", []):
            item["creatorId"] = fid  # Link episode to creator
            all_items.append(item)
        
        # Save progress every 1000 episodes
        if len(all_items) % 1000 == 0:
            pd.DataFrame(all_items).to_csv("podcasts_partial.csv", index=False)
            print(f"💾 Saved progress — {len(all_items)} records so far.")
            
    except requests.exceptions.HTTPError as e:
        status = e.response.status_code
        print(f"❌ Error {status} for ID {fid}")
        
        if status in [429, 503]:  # Rate limit
            print("⚠️ Rate limit hit. Sleeping for 60 seconds...")
            time.sleep(60)
            continue
            
        failed_ids.append(fid)
        continue

# Final save
df_final = pd.DataFrame(all_items)
df_final.to_csv("podcast_ch_episodes.csv", index=False)
print(f"✅ Scraping complete! Saved {len(df_final)} records")
```

**Output**: `podcast_ch_episodes.csv` (contains all episode details)

---

### Step 3: Generate Embeddings

Generate AI embeddings for semantic search using a multilingual model.

#### 3.1 Install Dependencies

```bash
pip install sentence-transformers torch pandas numpy
```

#### 3.2 Run the Embedding Script

Run the **Embedding** section in `Podcast.ipynb`:

```python
import pandas as pd
import torch
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from decimal import Decimal

# ========================================
# LOAD DATA
# ========================================
episodes = pd.read_csv("podcast_ch_episodes.csv")
creators = pd.read_csv("podcasts_ch.csv")

# Rename columns to avoid conflicts
episodes = episodes.rename(columns={
    "title": "episode_title", 
    "description": "episode_description"
})
creators = creators.rename(columns={
    "description": "creator_description"
})

# ========================================
# MERGE EPISODES + CREATORS
# ========================================
df = episodes.merge(
    creators, 
    left_on="creatorId", 
    right_on="id", 
    how="left", 
    suffixes=("", "_creator")
)

# ========================================
# EXTRACT CATEGORIES
# ========================================
category_cols = [f"category{i}" for i in range(1, 11)]

def extract_categories(row):
    categories = []
    for c in category_cols:
        val = row.get(c)
        if isinstance(val, str) and val.lower() not in ["", "n/a", "none"]:
            categories.append(val.strip())
    return categories

# ========================================
# BUILD EMBEDDING TEXT
# ========================================
def build_text(row):
    title = row.get("episode_title", "") or "Untitled Episode"
    edesc = row.get("episode_description", "") or ""
    cdesc = row.get("creator_description", "") or ""
    lang = row.get("language", "") or ""
    
    categories = extract_categories(row)
    cat_text = ", ".join(categories) if categories else "General"
    
    combined = (
        f"Episode Title: {title}\n"
        f"Episode Description: {edesc}\n"
        f"Podcast Description: {cdesc}\n"
        f"Categories: {cat_text}\n"
        f"Language: {lang}"
    )
    
    return combined.strip()

df["embedding_text"] = df.apply(build_text, axis=1)

# ========================================
# GENERATE EMBEDDINGS
# ========================================
print("Loading embedding model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

print("Generating embeddings...")
embeddings = model.encode(
    df["embedding_text"].tolist(), 
    normalize_embeddings=True,
    batch_size=256, 
    show_progress_bar=True
)

# ========================================
# BUILD FINAL JSON OUTPUT
# ========================================
def sanitize(obj):
    """Convert numpy/pandas types to JSON-serializable types"""
    if obj is None or obj is pd.NA:
        return None
    
    if isinstance(obj, np.floating):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    
    if isinstance(obj, np.integer):
        return int(obj)
    
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    
    if isinstance(obj, (str, int, bool)):
        return obj
    
    if isinstance(obj, Decimal):
        return float(obj)
    
    if isinstance(obj, (list, tuple)):
        return [sanitize(x) for x in obj]
    
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    
    if isinstance(obj, np.ndarray):
        return [sanitize(x) for x in obj.tolist()]
    
    if isinstance(obj, np.object_):
        return sanitize(obj.item())
    
    try:
        return sanitize(obj.__dict__)
    except:
        return str(obj)

output = []

for (_, row), emb in zip(df.iterrows(), embeddings):
    entry = {
        "creator": row.get("clean_title", ""),
        "title": row.get("episode_title", ""),
        "description": row.get("episode_description", ""),
        "link": row.get("link", None),
        "image": row.get("clean_imageUrl", ""),
        "lastUpdate": row.get("lastUpdate", None),
        "categories": extract_categories(row),
        "language": row.get("language", ""),
        "embedding": emb.tolist()
    }
    
    entry = sanitize(entry)
    output.append(entry)

# ========================================
# SAVE JSON
# ========================================
with open("podcast_embeddings.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"✅ Saved podcast_embeddings.json ({len(output)} episodes)")
```

**Output**: `podcast_embeddings.json` (~400MB file with all embeddings)

**⏱️ Time Estimate**: 
- **With GPU (Colab)**: 10-30 minutes for 40,000+ episodes
- **With CPU**: 1-3 hours

---

### Step 4: Split into Shards

The large JSON file needs to be split into smaller chunks (~23MB each) for efficient browser loading.

#### 4.1 Install Dependencies

```bash
pip install ijson
```

#### 4.2 Run the Sharding Script

```python
import ijson
import json
import os

# Configuration
MAX_MB = 23
MAX_BYTES = MAX_MB * 1024 * 1024

# Create output directory
os.makedirs("shards", exist_ok=True)

shard = []
shard_id = 0
size = 0

# Stream and split the large JSON file
with open("podcast_embeddings.json", "r") as f:
    items = ijson.items(f, "item", use_float=True)
    
    for item in items:
        entry = json.dumps(item)
        entry_size = len(entry.encode("utf-8"))
        
        # If adding this item exceeds shard size, save current shard
        if size + entry_size > MAX_BYTES:
            with open(f"shards/shard_{shard_id}.json", "w") as out:
                out.write("[" + ",".join(shard) + "]")
            
            print(f"✅ Saved shard_{shard_id}.json ({size / 1024 / 1024:.2f} MB)")
            
            shard = []
            shard_id += 1
            size = 0
        
        shard.append(entry)
        size += entry_size

# Write the last shard
with open(f"shards/shard_{shard_id}.json", "w") as out:
    out.write("[" + ",".join(shard) + "]")

print(f"✅ Created {shard_id + 1} shard files")
```

**Output**: `shards/` directory containing `shard_0.json` through `shard_162.json`

**📦 Optional**: Zip the shards for easier transfer:
```bash
zip -r shards.zip shards/
```

---

### Step 5: Deploy the Website

#### 5.1 Create Project Structure

```
your-project/
├── index.html              ← Your search interface (the improved HTML file)
├── shards/
│   ├── shard_0.json
│   ├── shard_1.json
│   ├── ...
│   └── shard_162.json
└── README.md              ← This file
```

#### 5.2 Update Configuration

Open `index.html` and update the `SHARD_COUNT` in the configuration:

```javascript
const CONFIG = {
  SHARD_COUNT: 163,  // Match the number of shard files you created
  TOP_K: 100,
  FINAL_K: 10,
  // ... other settings
};
```

#### 5.3 Deploy to GitHub Pages

1. **Create a new GitHub repository**
   ```bash
   git init
   git add .
   git commit -m "Initial commit: Chinese Podcast Search"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
   git push -u origin main
   ```

2. **Enable GitHub Pages**
   - Go to your repository on GitHub
   - Click **Settings** > **Pages**
   - Under "Source", select **main** branch
   - Click **Save**
   - Your site will be live at: `https://YOUR_USERNAME.github.io/YOUR_REPO/`

3. **Alternative Hosting Options**
   - **Netlify**: Drag and drop your folder at [netlify.com/drop](https://app.netlify.com/drop)
   - **Vercel**: Connect your GitHub repo at [vercel.com](https://vercel.com)
   - **Cloudflare Pages**: Push to GitHub and connect at [pages.cloudflare.com](https://pages.cloudflare.com)

---

## 📁 Project Structure

```
podcast-search/
│
├── index.html                    # Main search interface (100% client-side)
│
├── shards/                       # 163 JSON files with embeddings
│   ├── shard_0.json              # ~23MB each
│   ├── shard_1.json
│   └── ...
│
├── Podcast.ipynb                 # Jupyter notebook for data processing
│
├── podcasts_ch.csv               # Raw podcast creator data from SQL
├── podcast_ch_episodes.csv       # Episode details from API
├── podcast_embeddings.json       # Full embeddings file (not deployed)
│
└── README.md                     # This documentation
```

---

## 🎨 Customization

### Change the Number of Search Results

In `index.html`, modify:

```javascript
const CONFIG = {
  FINAL_K: 10,  // Change to 20, 30, etc. for more results
};
```

### Adjust Search Speed vs Quality

```javascript
const CONFIG = {
  TOP_K: 100,      // Increase for better quality (slower)
  BATCH_SIZE: 10,  // Increase for faster loading (more memory)
};
```

### Use a Different Embedding Model

In the Python embedding script, change:

```python
model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
```

To another model like:
- `paraphrase-multilingual-MiniLM-L12-v2` - Better for Chinese
- `distiluse-base-multilingual-cased-v2` - Faster, multilingual
- `sentence-transformers/LaBSE` - Best for Chinese but slower

⚠️ **Important**: If you change the model in Python, you must also update the model in `index.html`:

```javascript
const CONFIG = {
  MODEL_NAME: 'Xenova/paraphrase-multilingual-MiniLM-L12-v2'
};
```

### Customize the UI

Edit the CSS in `index.html`:

```css
/* Change color scheme */
body { 
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}

/* Change result card appearance */
.result-item {
  background: #f9f9f9;
  padding: 20px;
  border-radius: 12px;
}
```

---

## 🔧 Troubleshooting

### Problem: Search results are not relevant for Chinese queries

**Solution**: The current HTML uses `paraphrase-multilingual-MiniLM-L12-v2` which has better Chinese support. If results are still poor:

1. Try increasing keyword weight in the hybrid search
2. Use a Chinese-specific embedding model
3. Add more keyword matching patterns

### Problem: Website loads slowly

**Solutions**:
- Reduce `BATCH_SIZE` to load fewer shards simultaneously
- Increase shard size (e.g., 50MB instead of 23MB) to create fewer files
- Use a CDN for faster shard loading
- Enable gzip compression on your hosting platform

### Problem: "Failed to load shard" errors

**Causes**:
- Incorrect `SHARD_COUNT` in config
- Shards not uploaded to the correct folder
- CORS issues (if testing locally)

**Solutions**:
1. Check that `SHARD_COUNT` matches your actual number of files
2. Verify all shards are in the `shards/` folder
3. Use a local server for testing: `python -m http.server 8000`

### Problem: Memory errors during embedding generation

**Solutions**:
- Reduce `batch_size` in the embedding script (e.g., from 256 to 64)
- Use Google Colab with GPU for more memory
- Process data in smaller chunks

### Problem: API rate limits when fetching episodes

**Solutions**:
- Add delays between requests: `time.sleep(1)`
- Save progress frequently (every 100 requests)
- Resume from failed IDs

---

## 📊 Performance Metrics

Based on typical usage:

| Metric | Value |
|--------|-------|
| Total Episodes | ~47,000 |
| Total Shards | 163 files |
| Shard Size | ~23MB each |
| Total Data Size | ~3.7GB |
| Initial Load Time | 3-5 seconds |
| Search Time | 2-4 seconds |
| Embedding Dimension | 384 (MiniLM-L6-v2) |
| Browser Memory Usage | ~200-500MB |

---

## 🤝 Contributing

Contributions are welcome! Here are some ideas:

- [ ] Add podcast playback preview
- [ ] Implement filters (category, language, date)
- [ ] Add bookmark/favorite functionality
- [ ] Create a backend API for faster searches
- [ ] Support for English podcasts
- [ ] Add podcast ratings/reviews

---

## 📄 License

This project is open source and available under the MIT License.

---

## 🙏 Acknowledgments

- **Podcast Index**: For providing the podcast database and API
- **Sentence Transformers**: For the embedding models
- **Xenova/Transformers.js**: For browser-based AI inference
- **Hugging Face**: For model hosting

---

## 📞 Support

If you encounter issues:

1. Check the [Troubleshooting](#troubleshooting) section
2. Open an issue on GitHub
3. Review browser console (F12) for error messages

---

## 🚀 Next Steps

After completing the setup:

1. **Test locally**: Run a local server and verify search works
2. **Optimize**: Adjust configuration for your needs
3. **Deploy**: Push to GitHub Pages or other hosting
4. **Share**: Let people know about your podcast search engine!
5. **Iterate**: Gather feedback and improve the search quality

---

## 👨‍💻 Author

**Caslow Chien**  
📧 Email: [caslowchien@gmail.com](mailto:caslowchien@gmail.com)

Feel free to reach out for questions, suggestions, or collaborations!

---

**Built with ❤️ for the Chinese podcast community**
