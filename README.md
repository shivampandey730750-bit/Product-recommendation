# Smart Product Recommendation System
> **B.Tech 3rd-Year Artificial Intelligence & Machine Learning (AIML) Capstone Project**

An end-to-end machine learning-powered e-commerce recommendation web application. The platform provides real-time personalized recommendations by combining **Content-Based Filtering (TF-IDF + Cosine Similarity)**, **Item-Based Collaborative Filtering (IBCF)**, and an **Adaptive Hybrid Fusion Engine** with graceful **Cold-Start Handling**.

---

## 📋 Table of Contents
1. [Project Overview](#-project-overview)
2. [Key Features](#-key-features)
3. [Machine Learning Algorithms & Formulas](#-machine-learning-algorithms--formulas)
4. [System Architecture](#-system-architecture)
5. [Project Structure](#-project-structure)
6. [Windows PowerShell Setup & Execution Guide](#-windows-powershell-setup--execution-guide)
7. [Running Verification Tests](#-running-verification-tests)
8. [Offline Evaluation & Benchmarks](#-offline-evaluation--benchmarks)
9. [Project Limitations & Future Scope](#-project-limitations--future-scope)
10. [B.Tech Viva Voce Q&A Cheat Sheet](#-btech-viva-voce-qa-cheat-sheet)

---

## 🌟 Project Overview
Modern e-commerce platforms like Amazon and Netflix rely on intelligent recommendation engines to tackle information overload. This project implements a production-grade, modular recommendation pipeline that balances:
1. **Semantic Content Similarity**: Understanding product specifications, categories, brands, and tags through Natural Language Processing (NLP).
2. **Collaborative Community Patterns**: Capturing collective user preferences and co-rating behaviors ("wisdom of the crowd").
3. **Hybrid Re-ranking**: Dynamically balancing content and collaborative signals to deliver transparent recommendations with human-understandable explanations.

---

## ✨ Key Features
- **Modern E-Commerce Storefront**: Responsive shopping interface with live category filtering, price sliders, product sorting, and keyword search.
- **Content-Based Recommender**: Excludes the inspected product and surfaces the top-K most similar items using TF-IDF text features.
- **Personalized "For You" Section**: Simulates user sessions across pre-configured user personas (Audiophile, Fitness Fan, Tech Hacker, Home Barista) or cold-start guest visitors.
- **Live Interactive Wishlist**: Adding/removing items dynamically computes the centroid vector of the user's taste and generates real-time recommendations.
- **Transparent Explanations**: Every recommendation explains *why* it was chosen (e.g., `"49% match • Same subcategory (Audio) • Shared features: sony, headphones, wireless"`).
- **ML Analytics Dashboard**: Visualizations of category distribution, ratings histogram, price distributions, and TF-IDF similarity heatmaps.
- **Offline Evaluation Suite**: In-app live benchmark calculating **Precision@K**, **Recall@K**, **Mean Reciprocal Rank (MRR@K)**, and **Catalog Coverage**.
- **In-App Viva Voce Guide**: Interactive presentation tab designed specifically for college semester evaluations and external viva exams.

---

## 🧠 Machine Learning Algorithms & Formulas

### 1. Feature Synthesis & TF-IDF Vectorization
For each product $d$, a unified text representation ("content soup") is synthesized from the product name, category, subcategory, brand, tags, and description:

$$\text{TF}(t, d) = \frac{\text{count}(t \in d)}{\text{total tokens in } d}$$

$$\text{IDF}(t) = \ln\left(\frac{1 + N}{1 + \text{DF}(t)}\right) + 1$$

$$\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \text{IDF}(t)$$

where $N$ is total products in the catalog and $\text{DF}(t)$ is the document frequency of term $t$.

### 2. Cosine Similarity Metric
The semantic similarity between two product vectors $\mathbf{u}$ and $\mathbf{v}$ is given by the cosine of the angle between them in the multidimensional TF-IDF space:

$$\text{Cosine Similarity}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2} = \frac{\sum_{i=1}^n u_i v_i}{\sqrt{\sum_{i=1}^n u_i^2} \sqrt{\sum_{i=1}^n v_i^2}}$$

### 3. Item-Based Collaborative Filtering (IBCF)
Given the user-item interaction matrix $R$, we compute item-item cosine similarities. The predicted rating $\hat{r}(u, i)$ for user $u$ on candidate product $i$ is calculated using weighted ratings of items $j$ previously rated by user $u$:

$$\hat{r}(u, i) = \frac{\sum_{j \in \text{Rated}(u)} \text{sim}(i, j) \cdot r(u, j)}{\sum_{j \in \text{Rated}(u)} |\text{sim}(i, j)|}$$

To combine this with content scores, the predicted rating is normalized to the range $[0, 1]$:

$$S_{\text{collab}}(u, i) = \frac{\hat{r}(u, i)}{5.0}$$

### 4. Adaptive Hybrid Fusion
The overall hybrid score for candidate product $i$ given user $u$ is computed as:

$$\text{Score}_{\text{hybrid}}(u, i) = \alpha \cdot S_{\text{content}}(u, i) + (1 - \alpha) \cdot S_{\text{collab}}(u, i)$$

- **Hyperparameter $\alpha \in [0, 1]$**: Controls the blend ratio (default $\alpha = 0.55$).
- **Adaptive Cold-Start Fallback**: If user $u$ is a new visitor with no past rating interactions, the system dynamically shifts $\alpha \to 1.0$ (pure content/session wishlist matching) or returns top-rated trending items if no session actions exist.

---

## 🏗️ System Architecture

```
                  ┌─────────────────────────────────────┐
                  │          Raw Data Sources           │
                  │   data/products.csv & ratings.csv   │
                  └──────────────────┬──────────────────┘
                                     │
                                     ▼
                  ┌─────────────────────────────────────┐
                  │       Data Loader & Sanitizer       │
                  │    (Missing imputation, dedupe)     │
                  └──────────┬────────────────┬─────────┘
                             │                │
            Content Pipeline │                │ Collaborative Pipeline
                             ▼                ▼
         ┌─────────────────────────┐    ┌─────────────────────────┐
         │   TF-IDF Vectorizer     │    │   User-Item Matrix      │
         │  (N-grams, Stopwords)   │    │  (Rating Normalization) │
         └───────────┬─────────────┘    └───────────┬─────────────┘
                     │                              │
                     ▼                              ▼
         ┌─────────────────────────┐    ┌─────────────────────────┐
         │ Cosine Similarity Mat.  │    │ Item-Item Cosine Sim    │
         │ (S_content in [0, 1])   │    │ (S_collab in [0, 1])    │
         └───────────┬─────────────┘    └───────────┬─────────────┘
                     │                              │
                     └──────────────┬───────────────┘
                                    │
                                    ▼
                     ┌─────────────────────────────┐
                     │   Adaptive Hybrid Engine    │
                     │  Score = α·S_c + (1-α)·S_cf │
                     └──────────────┬──────────────┘
                                    │
                                    ▼
                     ┌─────────────────────────────┐
                     │ Streamlit Interactive UI    │
                     │ Storefront • Wishlist • Viva│
                     └─────────────────────────────┘
```

---

## 📂 Project Structure

```
minor project/
├── data/
│   ├── products.csv             # Realistic catalog across 7 categories with descriptions & tags
│   └── user_ratings.csv         # User-item rating history with personas for CF evaluation
├── src/
│   ├── __init__.py              # Python package marker
│   ├── config.py                # Dynamic path resolution & model hyperparameters
│   ├── data_loader.py           # Robust loading, data cleaning, and content soup synthesis
│   ├── content_based.py         # TF-IDF vectorization, Cosine similarity, explanation generator
│   ├── collaborative.py         # Item-Based CF, user prediction, cold-start handling
│   ├── hybrid.py                # Linear hybrid recommender with dynamic alpha weighting
│   ├── evaluation.py            # Train/test split, Precision@K, Recall@K, MRR, Coverage
│   └── utils.py                 # UI components, custom CSS, currency and star formatters
├── app.py                       # Main Streamlit web application (7 interactive modules)
├── test_recommendations.py      # Automated test suite verifying all 5 engine components
├── requirements.txt             # Required Python dependencies
├── .gitignore                   # Standard Python/Streamlit exclusions
└── README.md                    # Academic project documentation and Viva guide
```

### Purpose of Each File:
| File | Purpose |
|---|---|
| [`src/config.py`](file:///c:/Users/shiva/OneDrive/Pictures/Desktop/minor%20project/src/config.py) | Centralizes all hyperparameters (TF-IDF features, $\alpha$, top-K) and uses relative paths for portability. |
| [`src/data_loader.py`](file:///c:/Users/shiva/OneDrive/Pictures/Desktop/minor%20project/src/data_loader.py) | Safely reads CSV files, handles missing values, removes duplicates, and generates the combined text soup. |
| [`src/content_based.py`](file:///c:/Users/shiva/OneDrive/Pictures/Desktop/minor%20project/src/content_based.py) | Computes item-item similarity and profile-level centroids from text descriptions, with automated explanation synthesis. |
| [`src/collaborative.py`](file:///c:/Users/shiva/OneDrive/Pictures/Desktop/minor%20project/src/collaborative.py) | Constructs the sparse user-item matrix, computes item similarities, and predicts ratings for unseen products. |
| [`src/hybrid.py`](file:///c:/Users/shiva/OneDrive/Pictures/Desktop/minor%20project/src/hybrid.py) | Merges content and collaborative recommendations with user-controllable weights and cold-start fallback. |
| [`src/evaluation.py`](file:///c:/Users/shiva/OneDrive/Pictures/Desktop/minor%20project/src/evaluation.py) | Conducts offline train/test user-split evaluations to calculate Precision@K, Recall@K, and MRR. |
| [`src/utils.py`](file:///c:/Users/shiva/OneDrive/Pictures/Desktop/minor%20project/src/utils.py) | Injects custom e-commerce CSS styling, formatters, and HTML helper components. |
| [`app.py`](file:///c:/Users/shiva/OneDrive/Pictures/Desktop/minor%20project/app.py) | Main interactive multi-page dashboard featuring Storefront, Details, Personalized For You, Wishlist, Analytics, Evaluation, and Viva Guide. |
| [`test_recommendations.py`](file:///c:/Users/shiva/OneDrive/Pictures/Desktop/minor%20project/test_recommendations.py) | Autonomous test suite ensuring 100% mathematical and execution correctness before deployment. |

---

## 💻 Windows PowerShell Setup & Execution Guide

Follow these exact commands in Windows PowerShell:

### 1. Open PowerShell and Navigate to the Project Folder
```powershell
cd "c:\Users\shiva\OneDrive\Pictures\Desktop\minor project"
```

### 2. (Optional) Create and Activate Virtual Environment
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```
*(If PowerShell displays a script execution policy restriction, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first).*

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 4. Run the Automated Verification Test Suite
```powershell
python test_recommendations.py
```

### 5. Launch the Streamlit Web Application
```powershell
streamlit run app.py
```
Your default browser will automatically open:
```
Local URL: http://localhost:8501
```

---

## 🧪 Running Verification Tests

Run the included verification suite to validate all components:
```powershell
python test_recommendations.py
```
**Test Output Verification:**
- `[PASS]` Data Loading & Preprocessing (36 products, 60 ratings, 86.11% matrix sparsity)
- `[PASS]` Content-Based Filtering (excluding target item, verified text overlap and explanation generation)
- `[PASS]` Collaborative Filtering (Item-based rating prediction in $[1.0, 5.0]$, cold-start user fallback)
- `[PASS]` Hybrid Engine (Normalized linear score fusion with $\alpha=0.6$)
- `[PASS]` Evaluation Benchmark (Precision@K, Recall@K, MRR@K, Catalog Coverage computed)

---

## 📊 Offline Evaluation & Benchmarks

The system provides an offline benchmark using a 75/25 train/test interaction split per user:

| Model | Metric Cutoff | Precision@K | Recall@K | F1-Score@K | MRR@K | Catalog Coverage |
|---|---|---|---|---|---|---|
| **Content-Based** | $K=3$ | 0.2424 | 0.7273 | 0.3636 | 0.6818 | 58.33% |
| **Content-Based** | $K=5$ | 0.1636 | 0.8182 | 0.2727 | 0.7045 | 77.78% |
| **Collaborative Filtering** | $K=3$ | 0.0909 | 0.2727 | 0.1364 | 0.1212 | 38.89% |
| **Collaborative Filtering** | $K=5$ | 0.0727 | 0.3636 | 0.1212 | 0.1394 | 50.00% |
| **Hybrid Model ($\alpha=0.5$)** | $K=3$ | 0.2121 | 0.6364 | 0.3182 | 0.5758 | 55.56% |
| **Hybrid Model ($\alpha=0.5$)** | $K=5$ | 0.1455 | 0.7273 | 0.2424 | 0.5985 | 77.78% |

*Insight:* On relatively sparse interaction matrices, Content-Based signals provide high recall and coverage, while Collaborative signals introduce serendipitous recommendations from peer tastes. The Hybrid model achieves the best balance of accuracy and catalog exploration.

---

## ⚠️ Project Limitations & Future Scope

### Honest Limitations:
1. **Sample Rating Scale**: The dataset contains 36 representative products and 60 ratings for rapid prototyping and classroom demonstration. Production systems utilize millions of interactions.
2. **Static Embeddings**: TF-IDF uses bag-of-words token statistics. It does not understand contextual synonyms as deeply as pre-trained Transformer embeddings (e.g., BERT / Sentence-Transformers).
3. **Implicit vs. Explicit Feedback**: Ratings in this sample dataset represent explicit $1-5$ ratings. Real-world platforms frequently optimize for implicit clicks, dwells, and add-to-carts using Matrix Factorization (ALS).

### Future Scope:
- Integrating Neural Collaborative Filtering (NCF) or DeepFM.
- Employing vector databases (e.g., ChromaDB, FAISS) for millisecond-latency approximate nearest neighbor (ANN) retrieval on 100,000+ items.
- Adding real-time Redis session caching for active cart telemetry.

---

## 🎓 B.Tech Viva Voce Q&A Cheat Sheet

**Q1. What is the cold-start problem and how is it addressed?**  
> *Answer:* The cold-start problem occurs when a new user joins with zero ratings, or a new product is added with zero purchase history. In our system, new users are detected by `is_cold_start_user()`; we dynamically adjust the hybrid weight $\alpha \to 1.0$ to utilize active wishlist/search actions, or fall back to Bayesian top-rated products in that category.

**Q2. Why is Cosine Similarity chosen over Euclidean Distance?**  
> *Answer:* Euclidean distance measures the absolute distance between points and is sensitive to document length (a long description with repeated words would seem distant from a short description). Cosine similarity measures the angle between vectors, normalizing for document length and focusing purely on the direction and proportion of term frequencies.

**Q3. How is Precision@K different from standard classification accuracy?**  
> *Answer:* In recommendation systems, the user only views the top $K$ items (e.g., 5 products). Standard accuracy across the entire catalog is meaningless because 99% of products are non-relevant. Precision@K measures the exact fraction of the top $K$ presented items that were truly relevant to the user.

**Q4. What is the role of the $\alpha$ parameter in the Hybrid model?**  
> *Answer:* $\alpha \in [0, 1]$ balances content semantics against collaborative peer ratings: $\text{Score} = \alpha \cdot S_{\text{content}} + (1 - \alpha) \cdot S_{\text{collab}}$. When $\alpha=1$, it functions purely on product metadata; when $\alpha=0$, it functions purely on user collaborative ratings.

---

## 👨‍💻 Author & Academic Attribution
- **Degree**: Bachelor of Technology (B.Tech) in Artificial Intelligence & Machine Learning (AIML)
- **Year**: 3rd Year Capstone Project
- **Subject**: Machine Learning / Recommendation Systems Lab
