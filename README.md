# ImageGeneration_from_Reviews

# Cat Tree Reviews to Image Generation

Generate product images from customer review text using LLM analysis and diffusion models.

## Overview

This project explores whether AI can reconstruct complex 3D furniture structures from textual descriptions. The pipeline: **scrape reviews → LLM extract visual features → generate images with DALL-E 3 & Stable Diffusion**.

## Product Selection

**Product:** BestPet 70in Cat Tree Tower, Dark Gray  
**Source:** [Walmart Product Page](https://www.walmart.com/ip/BestPet-70in-Cat-Tree-Tower-Dark-Gray-w-Scratch-Posts-House-Funny-Toys/2052331762)

**Product choice** Cat trees present an interesting challenge for text-to-image generation:
- Multi-level platforms with spatial relationships
- Varied textures (plush, sisal, carpet)
- Specific design elements (scratching posts, hanging toys, enclosed houses)
- Reviews often mention structural details ("sturdy base," "three-tier design") and tactile qualities ("soft plush")

This tests the LLM's ability to extract architectural features from unstructured text and translate them into coherent visual prompts.

## Data Collection

- **Tool:** Instant Data Scraper Chrome Extension
- **Scope:** 106 pages of customer reviews (~730 valid reviews)
- **Processing:** Extracted only the review text column, removed ratings, dates, usernames, etc.

## Methodology

### 1. Preprocessing
- Loaded raw CSV, retained only `review` column
- Removed NaN values → 730 clean reviews
- Combined all reviews (~28k tokens) — within GPT-4's context window, so no sampling needed

### 2. Embedding & Clustering (Visualization)
Used sentence-transformers + KMeans to understand review topic distribution:

```python
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(reviews)
kmeans = KMeans(n_clusters=15)
```

This was to visualize the semantic structure of reviews across 15 topic clusters.
<img width="1064" height="442" alt="cattree_reviews_clustering" src="https://github.com/user-attachments/assets/cb2b7c2f-2200-449e-bb32-45b7fe751e87" />



### 3. LLM Feature Extraction (GPT-4o)
Fed all 730 reviews to GPT-4o to extract visual features in JSON format:
- Structure & dimensions
- Materials & textures  
- Components (scratching posts, condos, platforms, toys)
- Color scheme
- Generated image prompt

### 4. Prompt Iteration

**Problem:** The LLM-generated prompt included negative sentiment words like "wobbly," "unstable," "not very sturdy" — faithfully reflecting the mixed reviews but not ideal for product image generation.

**Solution:** Created 3 prompts for comparison:

| Prompt | Description |
|--------|-------------|
| `review_based` | Raw LLM output (includes negative words) |
| `positive_extracted` | Prompted LLM to extract only positive features (still had "wobbly") |
| `ideal_manual` | Manually curated prompt with only positive visual descriptors |

### 5. Image Generation
Generated images using both **DALL-E 3** and **Stable Diffusion XL** with each of the 3 prompts.

## Results

### DALL-E 3
<img width="1042" height="370" alt="dalle_cattree" src="https://github.com/user-attachments/assets/95bea857-55f9-4d14-b789-67c0ff02a7f6" />


### Stable Diffusion XL
<img width="1040" height="364" alt="sd_cattree" src="https://github.com/user-attachments/assets/847ff412-94ba-4ea3-b682-9c288ad46891" />


## Observations

1. **Stable Diffusion produces more realistic cat trees** — DALL-E 3 outputs look more stylized/artistic, while SD generates images closer to actual product photos.

2. **Negative words don't visually manifest** — Despite prompts containing "wobbly" or "unstable," neither model actually rendered a visually unstable structure. Text-to-image models don't seem to interpret abstract quality descriptors into visual representations.

3. **Manual prompt curation matters** — The `ideal_manual` prompt with explicit visual details (e.g., "sisal-wrapped scratching posts," "circular entrances," "white background, product photography style") produced the most product-photo-like results.

4. **LLM is too "honest"** — When analyzing mixed reviews, GPT-4 faithfully extracts both positive and negative features. For marketing/product visualization use cases, explicit prompt engineering to filter sentiment is necessary.


```

## Tech Stack

- **Data Collection:** Instant Data Scraper (Chrome Extension)
- **Embeddings:** sentence-transformers (`all-MiniLM-L6-v2`)
- **Clustering:** scikit-learn KMeans + t-SNE visualization
- **LLM:** OpenAI GPT-4o
- **Image Generation:** DALL-E 3, Stable Diffusion XL (via Stability AI API)
- **Environment:** Google Colab

## Prompts Used

<details>
<summary>Click to expand prompts</summary>

**review_based:**
> A 70-inch dark gray cat tree tower with multiple compact tiers. It features small platforms and enclosed spaces, covered in soft plush fabric. The base is not very stable, and the structure includes limited sisal scratching posts. Hanging toys like balls are attached but easily detachable. The overall aesthetic is compact and affordable, suitable for small cats or kittens.

**positive_extracted:**
> A tall, dark gray cat tree tower with multiple compact levels and small platforms. It features plush fabric covering with a soft texture. The structure includes small enclosed cat houses and a few low-placed scratching posts. Hanging toys like balls are attached but appear easily detachable. The overall design looks wobbly and is more suitable for kittens or small cats. The color scheme is primarily dark gray with some navy blue and light gray accents.

**ideal_manual:**
> A 70-inch tall multi-tiered cat tree tower in dark gray plush fabric, featuring 5-6 spacious platforms at different heights, sisal-wrapped scratching posts, two cozy enclosed cat condos with circular entrances, soft carpeted perches, hanging pom-pom toys, sturdy wide base for stability, set against a clean white background, product photography style, high quality, detailed textures

</details>






# Image Generation Agentic Workflow

This repository implements an end-to-end agentic workflow that converts raw customer reviews into structured visual prompts and final product images.
It includes these stages:
	1.	IngestionAgent – load & clean CSV
	2.	EmbeddingAgent – compute MiniLM embeddings
	3.	ClusteringAgent – (optional) cluster reviews
	4.	RAGStoreAgent – build FAISS vector store
	5.	RetrievalAgent – retrieve visual/positive/negative review corpora
	6.	PromptAgent – LLM → JSON feature extraction + image prompts
	7.	ImageAgent – image generation (DALL-E / Stable Diffusion / Google Imagen)
	8.	run_workflow() – orchestrates the entire pipeline

⸻

🚀 Quick Start

1. Install Dependencies

pip install -r requirements.txt

Make sure you have:
	•	openai
	•	google-genai
	•	sentence-transformers
	•	faiss-cpu
	•	matplotlib
	•	pandas
	•	tqdm
	•	python-dotenv

⸻
