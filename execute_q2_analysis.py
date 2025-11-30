"""
Q2 Analysis Script - Bike Helmet Reviews to Visual Features
Executes the complete analysis workflow and saves results
"""

import pandas as pd
import numpy as np
import json
import os
from sentence_transformers import SentenceTransformer
import faiss
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
from openai import OpenAI

# Set up
print("=" * 60)
print("BIKE HELMET REVIEW ANALYSIS - Q2")
print("=" * 60)

# Load environment
load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Part 1: Load and preprocess reviews
print("\n[1/9] Loading reviews data...")
df = pd.read_csv('reviews.csv')
print(f"Loaded {len(df)} rows")

# Extract reviews from 'tl-m' column (index 3)
df_reviews = df.iloc[:, [3]].copy()
df_reviews = df_reviews.dropna()
df_reviews.columns = ['review']
df_reviews = df_reviews.reset_index(drop=True)

print(f"Cleaned reviews: {len(df_reviews)}")

# Save cleaned data
df_reviews.to_csv('cleaned_helmet_reviews.csv', index=False)

# Part 2: Generate embeddings
print("\n[2/9] Generating embeddings...")
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(df_reviews['review'].tolist(), show_progress_bar=True)
print(f"Generated {len(embeddings)} embeddings")

# Part 3: Create FAISS index
print("\n[3/9] Creating FAISS index...")
embeddings_np = np.array(embeddings, dtype='float32')
faiss.normalize_L2(embeddings_np)

d = embeddings_np.shape[1]
index = faiss.IndexFlatIP(d)
index.add(embeddings_np)
print(f"FAISS index created with {index.ntotal} vectors")

# Part 4: K-means clustering
print("\n[4/9] Performing K-means clustering...")
n_clusters = 8
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(embeddings_np)
df_reviews['cluster'] = cluster_labels

print(f"Created {n_clusters} clusters")
print("\nCluster distribution:")
print(df_reviews['cluster'].value_counts().sort_index())

# Save cluster info
cluster_info = {}
for cluster_id in range(n_clusters):
    cluster_reviews = df_reviews[df_reviews['cluster'] == cluster_id]
    sample = cluster_reviews['review'].iloc[0]
    cluster_info[f"cluster_{cluster_id}"] = {
        "count": len(cluster_reviews),
        "sample": sample[:200]
    }

with open('cluster_analysis.json', 'w') as f:
    json.dump(cluster_info, f, indent=2)

# Part 5: Semantic search function
def search_reviews(query, top_k=400):
    """Search reviews using FAISS"""
    q_emb = model.encode([query], convert_to_numpy=True)
    q_emb = q_emb.astype('float32')
    faiss.normalize_L2(q_emb)
    
    scores, indices = index.search(q_emb, top_k)
    indices = indices[0]
    scores = scores[0]
    
    results = df_reviews.iloc[indices].copy()
    results['score'] = scores
    return results

# Part 6: Retrieve reviews for different aspects
print("\n[5/9] Retrieving reviews for visual analysis...")

visual_query = """
Reviews describing the bike helmet's visual appearance and features:
magnetic visor design, detachable goggles, USB rechargeable light, LED light positions,
helmet shape and form, ventilation holes pattern, color options, overall design style,
adjustable dial system, material texture.
"""

function_query = """
Descriptions of how the bike helmet operates and functions:
visor attachment mechanism, goggles magnetic system, light brightness and modes,
ventilation airflow, fit adjustment, padding comfort, safety protection,
ease of use, weight and balance.
"""

materials_query = """
Comments about materials, textures, and build quality:
EPS foam structure, PC shell material, padding quality and comfort,
visor and goggle materials, light housing, overall durability and sturdiness,
quality of construction.
"""

top_visual_reviews = search_reviews(visual_query, top_k=400)
top_function_reviews = search_reviews(function_query, top_k=400)
top_materials_reviews = search_reviews(materials_query, top_k=400)

print(f"Retrieved {len(top_visual_reviews)} visual reviews")
print(f"Retrieved {len(top_function_reviews)} function reviews")
print(f"Retrieved {len(top_materials_reviews)} material reviews")

# Part 7: Combine and sample
print("\n[6/9] Combining and sampling reviews...")
core_visual_df = pd.concat(
    [top_visual_reviews, top_function_reviews, top_materials_reviews],
    ignore_index=True
)

# Remove duplicates
core_visual_df = core_visual_df.drop_duplicates(subset=['review']).reset_index(drop=True)
print(f"After deduplication: {len(core_visual_df)} reviews")

# Sample if needed
max_reviews = 500
if len(core_visual_df) > max_reviews:
    core_visual_df = core_visual_df.sample(n=max_reviews, random_state=42).reset_index(drop=True)
    print(f"Sampled to: {len(core_visual_df)} reviews")

# Prepare text
all_reviews_text = "\\n\\n---REVIEW---\\n\\n".join(core_visual_df['review'].tolist())
print(f"Total characters: {len(all_reviews_text)}")
print(f"Estimated tokens: ~{len(all_reviews_text) // 4}")

# Part 8: OpenAI feature extraction
print("\n[7/9] Calling OpenAI API for feature extraction...")

features_json = {}
image_prompt = ""

if not OPENAI_API_KEY:
    print("WARNING: OPENAI_API_KEY not found in environment")
    if os.path.exists('helmet_extracted_features.json'):
        print("Loading cached features from 'helmet_extracted_features.json' instead...")
        with open('helmet_extracted_features.json', 'r') as f:
            features_json = json.load(f)
        print("✓ Cached features loaded successfully")
        image_prompt = features_json.get('image_prompt', '')
        print(f"\nImage prompt generated ({len(image_prompt)} chars)")
    else:
        print("ERROR: No API key and no cached features found.")
        print("Please create a .env file with your API key or ensure 'helmet_extracted_features.json' exists.")
else:
    client = OpenAI(api_key=OPENAI_API_KEY)
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert product analyst and visual designer."},
                {"role": "user", "content": analysis_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.7
        )
        
        response_text = response.choices[0].message.content
        features_json = json.loads(response_text)
        
        print("✓ Features extracted successfully")
        
        # Save features
        with open('helmet_extracted_features.json', 'w') as f:
            json.dump(features_json, f, indent=2)
        
        image_prompt = features_json.get('image_prompt', '')
        print(f"\nImage prompt generated ({len(image_prompt)} chars)")
        
    except Exception as e:
        print(f"ERROR calling OpenAI API: {e}")

# Part 9: Generate image with DALL-E
print("\n[8/9] Generating product image with DALL-E...")

if image_prompt and OPENAI_API_KEY:
    try:
        image_response = client.images.generate(
            model="dall-e-3",
            prompt=image_prompt,
            size="1024x1024",
            quality="standard",
            n=1
        )
        
        image_url = image_response.data[0].url
        print("✓ Image generated successfully")
        
        # Save image URL
        with open('generated_helmet_image.txt', 'w') as f:
            f.write(f"Image URL: {image_url}\n")
            f.write(f"\nPrompt: {image_prompt}\n")
        
        # Download image
        import requests
        img_data = requests.get(image_url).content
        with open('generated_helmet_image.png', 'wb') as handler:
            handler.write(img_data)
        print("✓ Image saved as generated_helmet_image.png")
        
    except Exception as e:
        print(f"ERROR generating image: {e}")
        image_url = ""
else:
    if not image_prompt:
        print("Skipping image generation (no prompt available)")
    elif not OPENAI_API_KEY:
        print("Skipping image generation (no API key)")
        print(f"Prompt that would be used:\n{image_prompt}")
    image_url = ""

# Part 10: Create visualizations
print("\n[9/9] Creating visualizations...")

# Cluster distribution chart
plt.figure(figsize=(10, 6))
cluster_counts = df_reviews['cluster'].value_counts().sort_index()
plt.bar(cluster_counts.index, cluster_counts.values, color='steelblue', alpha=0.8)
plt.xlabel('Cluster ID', fontsize=12)
plt.ylabel('Number of Reviews', fontsize=12)
plt.title('Review Distribution Across Clusters', fontsize=14, fontweight='bold')
plt.grid(axis='y', alpha=0.3)
for i, v in enumerate(cluster_counts.values):
    plt.text(i, v + 5, str(v), ha='center', fontsize=10)
plt.tight_layout()
plt.savefig('cluster_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Saved cluster_distribution.png")

# Summary
print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
print(f"\nTotal reviews analyzed: {len(df_reviews)}")
print(f"Clusters identified: {n_clusters}")
print(f"Reviews for visual analysis: {len(core_visual_df)}")
print(f"\nGenerated files:")
print("  - cleaned_helmet_reviews.csv")
print("  - cluster_analysis.json")
print("  - helmet_extracted_features.json")
print("  - cluster_distribution.png")
if image_url:
    print("  - generated_helmet_image.png")
    print("  - generated_helmet_image.txt")

print("\n✓ Ready for Q2 documentation")
