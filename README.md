# dsci560-lab5-Reddit-program
# DSCI 560 – Lab 5  
## Reddit Cybersecurity Topic Clustering System

### Author
group zll
Member:
Benson Luo, 9649157234
Yanbing Zhu, 2695304141
Jicheng Liu, 4656726511


## 1. Project Overview

This project builds an end-to-end data pipeline for scraping, processing, embedding, clustering, and querying Reddit posts within the cybersecurity domain.

The system:

- Scrapes posts from Old Reddit (HTML parsing)
- Stores raw data in MySQL
- Generates fixed-dimension embeddings using TF-IDF + SVD
- Clusters posts using K-Means
- Extracts topic keywords per cluster
- Visualizes clusters using PCA
- Supports automated periodic updates
- Allows interactive user queries during automation

The system processes approximately 5,000 unique Reddit posts across multiple cybersecurity-related subreddits.

## 2. Technology Stack

- Python 3
- MySQL
- BeautifulSoup (HTML parsing)
- Scikit-learn (TF-IDF, SVD, KMeans, PCA)
- Matplotlib (visualization)

## 3. Database Schema

### posts table
Stores raw scraped data.

Key fields:
- `post_id` (UNIQUE)
- `subreddit`
- `title`
- `clean_text`
- `author_masked`
- `created_at`
- `post_url`
- `image_url`
- `is_ad`
- `cluster_id`

Upsert logic ensures no duplication using: INSERT … ON DUPLICATE KEY UPDATE


### embeddings table
Stores vector representations.

Key fields:
- `post_row_id`
- `vector_json`
- `dim`
- `model_version`

Embeddings are version-controlled via `model_version` to avoid dimension mismatch.


### cluster_topics table
Stores top keywords for each cluster.

Key fields:
- `model_version`
- `k`
- `cluster_id`
- `top_terms`


## 4. Pipeline Workflow

### Step 1 – Scraping

Scrape posts using Old Reddit: python scraper.py 3000 –subs cybersecurity,netsec,sysadmin,privacy –sleep 3.5

To increase coverage, we switched from `/new/` to `/top/?t=year` when recent posts became saturated.


### Step 2 – Preprocessing
python preprocess.py

Cleans text and prepares `clean_text`.

### Step 3 – Embedding Generation
python embed.py –limit 5000 –dim 128 –model_version tfidf_svd_v3

- TF-IDF vectorization
- Dimensionality reduction using TruncatedSVD
- L2 normalization
- Stored in MySQL

### Step 4 – Clustering
for 5000 data: python cluster_from_embeddings.py –k 12 –limit 5000 –model_version tfidf_svd_v3

K-Means clustering is performed directly on stored embeddings.
Cluster assignments are written back to the `posts` table.

### Step 5 – Keyword Extraction
python keywords.py –model_version tfidf_svd_v3 –k 12 –topn 10
Extracts representative keywords per cluster.

### Step 6 – Visualization
python visualize.py –model_version tfidf_svd_v3 –out cluster_pca_v3_k12.png
PCA reduces embeddings to 2D for visualization.

## 5. Automation Mode

Run the full automated pipeline: python main.py 5 –scrape_n 200 –subs cybersecurity,netsec,sysadmin –embed_limit 5000 –cluster_limit 5000 –model_version tfidf_svd_v3 –k 12

Features:
- Runs full pipeline every 5 minutes
- Uses incremental scraping
- Keeps system interactive
- Allows query input during waiting period

To exit: :exit

## 6. Interactive Query

While automation is running, users can input natural language queries:

Example: phishing email campaign detection

The system:
1. Embeds the query
2. Compares it with cluster centroids
3. Returns the nearest cluster
4. Displays top keywords and representative posts


## 7. Design Decisions

- Old Reddit scraping avoids strict API limitations.
- Upsert logic prevents data duplication.
- Model versioning ensures embedding reproducibility.
- Increasing k from 8 to 12 improved cluster balance.
- Automation simulates a lightweight real-time topic clustering system.

## 8. Scalability

The system successfully processes approximately 5,000 unique posts.
The modular design allows:
- Larger scraping volumes
- Different embedding models
- Alternative clustering algorithms


## 9. Known Limitations

- HTML scraping may be rate-limited (HTTP 429).
- Short text titles reduce semantic richness.
- PCA visualization shows approximate separation only.


## 10. Conclusion

This project demonstrates a complete unsupervised topic modeling pipeline with database integration, versioned embeddings, clustering, visualization, automation, and interactive querying.

The system simulates a small-scale production data workflow rather than a static notebook experiment.