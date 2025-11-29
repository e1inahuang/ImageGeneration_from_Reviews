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

| Prompt               | Description                                                         |
| -------------------- | ------------------------------------------------------------------- |
| `review_based`       | Raw LLM output (includes negative words)                            |
| `positive_extracted` | Prompted LLM to extract only positive features (still had "wobbly") |
| `ideal_manual`       | Manually curated prompt with only positive visual descriptors       |

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

# Egg cooker Reviews to Image Generation
Generate product images from customer review text using LLM analysis and diffusion models.
## Overview
This part explores how three different diffusion models(DALL-E, Stable Diffusion, Nano Banana)function under three different data sampling strategies(baseline, cluster-based, RAG-based)
## Product Selection
**Product:** MyMini Premium 7-Egg Cooker (Kitchen Appliances Category)
**Source:** [Walmart Product Page](https://www.walmart.com/ip/Nostalgia-MyMini-Premium-350-Watt-7-Egg-Cooker-Teal/666985734)

**Product choice:** egg cooker is an unconventional electrical appliance but has a relatively similar appearance. Using an egg cooker as my target could test the model's pure understanding of text description. 
## Data Collection
- **Tool:** Instant Data Scraper Chrome Extension
- **Scope:** 277 pages of customer reviews (~2102 valid reviews)
- **Processing:** Extracted only the review text column, removed ratings, dates, usernames.
## Methodology
### 1. Data loading and prepossing 

- Loaded raw CSV, transformed to dataframe
    
- retained only review column
    
- Removed NaN values → 2102 clean reviews
    
- Because gpt-4o’s max input tokens is 30k, we have to sample the original dataset
    
#### 1) Base-line sampling

- Randomly selected 800 reviews from 2102 clean reviews

#### 2) Cluster-based sampling 

- Use SentenceTransformer all-MiniLM-L6-v2 for embedding
- Use KMeans to cluster all the 2102 clean reviews into 8 different clusters
- Visualizatized cluster distribution and cluster size distribution(see below figure)
- Use Silhouette Score, CH Score, DB Score to evaluate clustering quality: 
```
Silhouette Score: 0.04490712285041809

CH Score: 95.15264892578125

DB Score: 3.682014979654795 
```
- Randomly select 100 from each cluster, in total 800 reviews
     
<img width="1064" height="442" alt="cluster_outcome" src="https://github.com/user-attachments/assets/979be6ed-1d3d-4a0a-91a9-88a0735981ce" />

#### 3）RAG-based sampling

- Use SentenceTransformer all-MiniLM-L6-v2 for embedding
    
- Use FAISS vector base to storage embedding, transform to float21, normalize to L2, build index
    
- Write 5 different queries(visual-related, functional-related, materials-related, positive design reviews, negative design reviews) to retrieve the vector base
    
-  Combine three types of visual-related reviews and remove duplicate, get 500 visual reviews
    
- Separately remove duplicate in top positive reviews and top negative reviews, get 250 reviews each, 500 in total
    

5 RAG Queries: 

```
visual_query = """

Reviews describing how the egg cooker operates: boiling eggs, steaming eggs, omelet tray, poaching cups, noise level, doneness levels, heating performance, capacity, steam vent, buzzer sound, cooking consistency, speed, ease of use.

"""

functional_query = """

Descriptions of the egg cooker’s visual appearance and physical structure: shape of the cooker, dome lid, transparent or opaque lid, egg tray design, heating plate, color of the unit, size and dimensions, number of eggs it holds, materials, buttons, lights, overall build, body shape, and design style.

"""

materials_query = """

Comments about the materials, textures, and build quality plastic lid, stainless steel heating plate, thickness of the base, durability, sturdiness, quality of the egg tray, condensation on lid,

how solid or cheap the cooker feels, and overall craftsmanship.

"""

positive_query = """

Reviews praising the egg cooker’s visual design: clean look, sleek appearance, modern style, compact size, neat egg tray layout, nice color, visually appealing lid, cute or attractive design.

"""

negative_query = """

Complaints about the egg cooker’s appearance or structure: too small, looks cheap, flimsy lid, poorly designed egg tray, unstable base, low-quality plastic, uneven cooking, foggy lid,

messy condensation, scratches, dents, color mismatch.

"""
```

### 2. Prompts generating 

| Prompt               | Description                               |
| -------------------- | ----------------------------------------- |
| `review_based`       | Extract VISUAL and STRUCTURAL information |
| `positive_extracted` | Analyze positive features of product      |
| `realistic_manual`   | Analyze negative features of product      |
| `ideal_manual`       | Manual prompt to produce the ideal prompt |


### 3. Image Generation:

- Set up Model (e.g., DALL-E 3)
    
- Generate 3-5 images using crafted prompts
    
- Compare with actual product images
    
- Document findings for Product A
    
#### Models:

1. Dall-e-3 from OpenAi
    
2. Stable Diffusion XL from [stability.ai](http://stability.ai)
    
3. imagen-4.0-generate-001(Nano Banana) from Google
    
## Result

### base line sampling
#### DALL-E
<img width="5370" height="1416" alt="dalle3_comparison" src="https://github.com/user-attachments/assets/c6c9c6a9-0ffd-46f0-a151-9538d85e6bb0" />

#### Stable Diffusion
<img width="5370" height="1416" alt="sd_comparison" src="https://github.com/user-attachments/assets/fccb274d-a825-477b-8d3e-db95456ca77e" />

#### Nano Banana
<img width="5370" height="1416" alt="gimg_comparison" src="https://github.com/user-attachments/assets/afec55e3-5378-49dd-b7f1-96c92bed0a87" />

### Cluster-based sampling
#### DALL-E
<img width="5370" height="1416" alt="dalle3_comparison" src="https://github.com/user-attachments/assets/cf982bd9-9b8b-45d9-97d1-1c5d8dee8853" />

#### Stable Diffusion
<img width="5370" height="1416" alt="sd_comparison" src="https://github.com/user-attachments/assets/f6618f20-d945-45db-a50c-63f672b84031" />

#### Nano Banana
<img width="5370" height="1416" alt="gimg_comparison" src="https://github.com/user-attachments/assets/453a8698-6c5e-4a01-9491-00b4ca1efa7d" />

### RAG-based sampling
#### DALL-E
<img width="5370" height="1416" alt="dalle3_comparison" src="https://github.com/user-attachments/assets/bbb67240-eb2d-46c3-a718-c1ed48715b24" />

#### Stable Diffusion
<img width="5370" height="1416" alt="sd_comparison" src="https://github.com/user-attachments/assets/f0c10257-910c-49ea-b45d-ab1cda5fc26a" />

#### Nano Banana
<img width="5370" height="1416" alt="gimg_comparison" src="https://github.com/user-attachments/assets/79f4e5ea-82e0-41d9-8d2f-f9b7d0cbdb07" />


# GenAI Product Review → Image Generation Agentic Workflow

This repository implements an end-to-end agentic workflow that converts raw customer reviews into structured visual prompts and final product images.
## Component: 
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

## 🚀 Quick Start

1. Install Dependencies

```
pip install -r requirements.txt
%%  Make sure you have:
	•	openai
	•	google-genai
	•	sentence-transformers
	•	faiss-cpu
	•	matplotlib
	•	pandas
	•	tqdm
	•	python-dotenv %%
```


⸻

## 🔑 2. Configure API Keys

**Create a .env file in the project root:**

```
OPENAI_API_KEY=your_openai_key_here
STABILITY_API_KEY=your_stability_key_here   # for Stable Diffusion
GOOGLE_API_KEY=your_google_genai_key_here  # for Google Imagen / Nano Banana
```

**Load them inside notebooks:**

```
from dotenv import load_dotenv
load_dotenv()
```

**On macOS/Linux, you can also export:**

```
export OPENAI_API_KEY=...
export STABILITY_API_KEY=...
export GOOGLE_API_KEY=...
```


⸻

### 📄 3. Prepare Your CSV File

Your CSV must contain a single text column (default: "review").

Example:

```
review
"I love this product..."
"The lid broke after 3 uses..."
```

If your column name is different (e.g., "text"), you can override using:

```
run_workflow(..., text_col="text")
```


⸻

## 🛠 4. Set the Product Name

You can pass the product name directly to the workflow:

```
results = run_workflow(
    csv_path="walmart.csv",
    output_root="./outputs/my_run",
    product_name="Dash Rapid Egg Cooker",
)
```
If you are calling from command line, use --product_name (if enabled):

```
python pipeline.py --csv walmart.csv --out outputs --product_name "Dash Rapid Egg Cooker"
```

The product name will be injected into:
	•	RetrievalAgent queries
	•	PromptAgent’s feature extraction prompts
	•	Final image generation prompts

⸻

## 🧪 5. Choose Sampling Mode

The workflow offers three ways to choose ~800 reviews for prompt generation:

```
sampling_mode="rag" (Recommended)
```

Retrieves reviews by semantic queries → Best accuracy, least noise.:
	•	visual structure
	•	materials
	•	functional structure
	•	positive appearance
	•	negative appearance


```
sampling_mode="cluster"
```

Samples evenly across KMeans clusters.

```
sampling_mode="random"
```

Random 800 reviews.

Set it like this:

```
run_workflow(..., sampling_mode="rag")
```


⸻

## 🎨 6. Choose Image Generation Model

The workflow supports 3 image models:

**DALL-E 3	"dalle3" / "dalle"	OpenAI**
**Stable Diffusion XL	"sd"	Stability AI**
**Google Imagen 4.0 (Nano Banana)	"imagen"	Google GenAI**

Choose like this:

```
run_workflow(
    ...,
    image_model="dalle",   # or: "sd" / "nano"
)
```

Generated images go into automatically separated folders:

outputs/egg_run/images_dalle3_rag/
outputs/egg_run/images_sd_cluster/
outputs/egg_run/images_imagen_random/


⸻

## 🧩 7. Modifying the Prompt Templates

Prompt templates are defined at the top of pipeline.py:

```
feature_prompt_template = """
...
{PRODUCT_NAME}
{FULL_TEXT}
{EXAMPLES}
{COUNT}
...
"""
```

And:

```
posneg_prompt_template = """
...
{PRODUCT_NAME}
{POS_FULL}
{NEG_FULL}
...
"""
```

Guidelines for Modifying Prompt Templates

✔ Keep placeholders ({FULL_TEXT}, {PRODUCT_NAME}, etc.)
✔ Escape JSON braces using {{ and }}
✔ Make sure the final output must remain valid JSON

Example: Adding new fields

You can simply extend the JSON template:

"lighting_conditions": "any descriptions related to reflections, gloss, shine"


⸻

## 📦 8. Running the Workflow

**From Python Notebook:**

```
from pipeline import run_workflow

results = run_workflow(
    csv_path="walmart.csv",
    output_root="./outputs/egg_run",
    product_name="Dash Rapid Egg Cooker",
    sampling_mode="rag",
    image_model="dalle3",
)
```
**From Command Line:**

```
python pipeline.py \
    --csv walmart.csv \
    --out ./outputs/egg_run \
    --image_model dalle3 \
    --sampling_mode rag \
    --product_name "Dash Rapid Egg Cooker"
```


⸻

## 📁 9. Workflow Output Structure

Example folder:

outputs/egg_run/
  artifacts/
    embeddings.npy
    reviews_with_clusters.csv
  images_dalle3_rag/
    dalle3_feature_based.png
    dalle3_ideal_from_pos.png
    dalle3_realistic_from_pos_neg.png
    dalle3_comparison.png
  prompts/
    feature_summary.json
    posneg_summary.json


⸻

## 🧠 10. Agentic Workflow Diagram (for report)

           CSV
            │
            ▼
     IngestionAgent
            │
            ▼
     EmbeddingAgent ───► FAISS Vector Store (RAGStoreAgent)
            │                     │
            │                     ▼
            ▼               RetrievalAgent
     ClusteringAgent            │
            │                   ▼
            └────────────► PromptAgent
                                │
                                ▼
                           ImageAgent
                                │
                                ▼
                         Generated Images


⸻
