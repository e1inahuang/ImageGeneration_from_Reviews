# ========== Core Libraries ==========
import os
import json
import base64
import requests
import numpy as np
import pandas as pd

# ========== For Images ==========
from PIL import Image
from io import BytesIO

# ========== ML / NLP ==========
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
import faiss

# ========== OpenAI LLM & Google Imagen ==========
from openai import OpenAI
from google import genai
from google.genai import types

# ========== Visualization ==========
import matplotlib.pyplot as plt
# ====== basic config ======
OUTPUT_ROOT = "./output"  

os.makedirs(OUTPUT_ROOT, exist_ok=True)

# ====== Prompt Templates ======
feature_prompt_template = """
You are an expert product analyst.

We are analyzing customer reviews for the product "{PRODUCT_NAME}".
Your job is to extract **physical, structural, and visual features** of the product, based ONLY on what customers describe.

Below are several representative review snippets:
{EXAMPLES}

And here is the FULL SET of {COUNT} visual/structural-related reviews:
{FULL_TEXT}

From these reviews, infer **all visually relevant product attributes**. 
Focus only on attributes that affect how the product LOOKS or is PHYSICALLY BUILT, such as:

- **Overall structure & geometry** (shape, form factor, layout, tier/layer structure, proportions)
- **Dimensions & size impressions** (height, width, depth, compact/large feel)
- **Components & parts** (buttons, trays, modules, attachments, handles, lids, screens, shelves, stands, compartments)
- **Materials & textures** (plastic, wood, metal, fabric, smooth, glossy, matte, soft, rigid, flexible)
- **Color & style** (primary colors, accents, visual themes, perceived aesthetic)
- **Build quality & sturdiness** only if described visually
- **Visual cues of functionality** (lights, indicators, openings, vents, display panels, surface layout)

DO NOT include:
- Emotional opinions (“I love it”, “I hate it”)
- Non-visual functional comments (performance, taste, sound, battery life, speed, etc.)
- Complaints unrelated to appearance
- Guesses or hallucinations — use ONLY what reviews mention.

Produce **ONLY valid JSON** in the following structure:

{{
  "structure": {{
    "overall_shape": "description of overall form factor",
    "layout": "how components or sections are arranged",
    "dimensions": "described or implied size characteristics"
  }},
  "materials": {{
    "primary_materials": ["materials mentioned"],
    "surface_textures": ["smooth / matte / glossy / soft / rough etc."],
    "quality_impression": "visual quality cues described by customers"
  }},
  "components": {{
    "major_parts": "description of key physical parts or modules",
    "interactive_elements": "buttons, switches, lids, handles, levers, trays, etc.",
    "special_features": "any notable visual physical feature mentioned"
  }},
  "colors": {{
    "primary": "main visible color(s)",
    "accents": ["secondary colors or trims"]
  }},
  "keywords": ["important descriptive adjectives from reviews"],
  "image_prompt": "A concise, accurate image-generation prompt describing the physical and visual attributes of '{PRODUCT_NAME}'."
}}

Output ONLY this JSON. No explanations or markdown.
"""

posneg_prompt_template = """ You are an expert product analyst.
We have a set of customer reviews for an {PRODUCT_NAME}.
Your task:
1. POSITIVE VISUAL FEATURES:
   - Only describe physical and visual features that customers LIKED
   - Example aspects: shape, color, compactness, trays, lid, materials, lights, overall look

2. NEGATIVE VISUAL FEATURES:
   - Only describe physical and visual issues that customers DISLIKED
   - Example aspects: looks cheap, too small, flimsy, unstable, poor materials, ugly colors

3. IDEAL_IMAGE_PROMPT:
   - A rich image generation prompt that shows the product at its BEST,
     based ONLY on positive visual features.

4. REALISTIC_IMAGE_PROMPT:
   - A balanced prompt that reflects both positive features and common complaints,
     representing how the product truly looks in real life.

Output valid JSON in this format (no extra text):

{{
  "positive_features": {{...}},
  "negative_features": {{...}},
  "ideal_image_prompt": "prompt focusing on best features",
  "realistic_image_prompt": "balanced prompt"
}}

Use:
{PRODUCT_NAME}
{POS_EXAMPLES}
{NEG_EXAMPLES}
{POS_FULL}
{NEG_FULL}
{POS_COUNT}
{NEG_COUNT}
"""

#IngestionAgent
class IngestionAgent:
    """Load and clean raw review data from CSV."""

    def __init__(self, review_col: str = "review"):
        self.review_col = review_col

    def load_csv(self, csv_path: str) -> pd.DataFrame:
        """Read raw CSV into a DataFrame."""
        df = pd.read_csv(csv_path)
        if self.review_col not in df.columns:
            raise ValueError(f"CSV must contain a '{self.review_col}' column.")
        return df

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic cleaning: strip, drop NA, drop empty, dedup."""
        df = df.dropna(subset=[self.review_col]).copy()
        df[self.review_col] = df[self.review_col].astype(str).str.strip()
        df = df[df[self.review_col] != ""]
        df = df.drop_duplicates(subset=[self.review_col]).reset_index(drop=True)
        return df

    def run(self, csv_path: str) -> pd.DataFrame:
        """Full ingestion step: load + clean."""
        df = self.load_csv(csv_path)
        df = self.clean(df)
        print(f"[Ingestion] Loaded {len(df)} cleaned reviews.")
        return df
    
    
#EmbeddingAgent
class EmbeddingAgent:
    """Compute and persist sentence embeddings for reviews."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode a list of texts into embeddings."""
        emb = self.model.encode(texts, show_progress_bar=True)
        return np.array(emb, dtype="float32")

    def save_embeddings(self, embeddings: np.ndarray, path: str) -> None:
        np.save(path, embeddings)
        print(f"[Embedding] Saved embeddings to {path}")

    def run(self, df: pd.DataFrame, text_col: str, save_path: str | None = None) -> np.ndarray:
        """Encode df[text_col]; optionally save to disk."""
        texts = df[text_col].tolist()
        embeddings = self.encode(texts)
        if save_path is not None:
            self.save_embeddings(embeddings, save_path)
        return embeddings
    

#ClusteringAgent

class ClusteringAgent:
    """Cluster review embeddings and attach cluster labels."""

    def __init__(self, n_clusters: int = 8): # number of clusters
        self.n_clusters = n_clusters

    def fit_predict(self, embeddings: np.ndarray) -> np.ndarray:
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10) # n_init=10 for sklearn >=1.4
        labels = kmeans.fit_predict(embeddings)
        return labels

    def compute_metrics(self, embeddings: np.ndarray, labels: np.ndarray) -> dict:
        sil = silhouette_score(embeddings, labels)
        ch_score = calinski_harabasz_score(embeddings, labels)
        db_score = davies_bouldin_score(embeddings, labels)
        return {"silhouette": sil , "calinski_harabasz": ch_score, "davies_bouldin": db_score}

    def run(self, df: pd.DataFrame, embeddings: np.ndarray, out_csv: str | None = None) -> pd.DataFrame:
        """Add cluster labels to df, compute metrics, optionally save."""
        labels = self.fit_predict(embeddings)
        df = df.copy()
        df["cluster"] = labels
        metrics = self.compute_metrics(embeddings, labels)
        print(f"[Clustering] n_clusters={self.n_clusters}, silhouette={metrics['silhouette']:.4f}, ch_score={metrics['calinski_harabasz']:.4f}, db_score={metrics['davies_bouldin']:.4f}")
        if out_csv is not None:
            df.to_csv(out_csv, index=False)
            print(f"[Clustering] Saved clustered reviews to {out_csv}")
        return df



#RAGStoreAgent
class RAGStoreAgent:
    """Manage FAISS index for semantic search."""

    def build_index(self, embeddings: np.ndarray) -> faiss.IndexFlatIP:
        faiss.normalize_L2(embeddings)
        d = embeddings.shape[1]
        index = faiss.IndexFlatIP(d)
        index.add(embeddings)
        print(f"[RAGStore] Built FAISS index with {index.ntotal} vectors.")
        return index

    def save_index(self, index: faiss.IndexFlatIP, path: str) -> None:
        faiss.write_index(index, path)
        print(f"[RAGStore] Saved FAISS index to {path}")

    def load_index(self, path: str) -> faiss.IndexFlatIP:
        index = faiss.read_index(path)
        print(f"[RAGStore] Loaded FAISS index from {path}")
        return index

#RetrievalAgent
class RetrievalAgent:
    """High-level retrieval over FAISS using query templates."""

    def __init__(self, 
                 embedding_agent: EmbeddingAgent,
                 index: faiss.IndexFlatIP,
                 df: pd.DataFrame,
                 product_name: str,
                 text_col: str = "review"):
        """
        product_name: e.g. 'Dash Rapid Egg Cooker' or 'Egg cooker'
        """
        self.embedding_agent = embedding_agent
        self.index = index
        self.df = df
        self.product_name = product_name
        self.text_col = text_col
        

    def search(self, query: str, top_k: int = 400) -> pd.DataFrame:
        q_emb = self.embedding_agent.encode([query])
        faiss.normalize_L2(q_emb)
        k = min(top_k, len(self.df))   # 防止 top_k > 总文档数
        scores, idx = self.index.search(q_emb, k)
        idx = idx[0]
        scores = scores[0]
        res = self.df.iloc[idx].copy()
        res["score"] = scores
        return res

    def build_corpora(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Return:
            core_visual_df: reviews about visual + structural aspects
            pos_df: positive visual/design reviews
            neg_df: negative visual/design reviews
        All queries are conditioned on self.product_name.
        """

        # ---------- Query templates with product_name injected ----------
        visual_query = f"""
        Visual and structural description of the product "{self.product_name}":
        shape of the cooker, lid style (dome / flat / transparent), egg tray layout,
        number of eggs it can hold, base size, buttons, indicator lights,
        overall geometry and how it looks on the counter.
        """

        materials_query = f"""
        Materials, textures and build quality of the product "{self.product_name}":
        plastic vs. stainless steel, finish (glossy / matte), thickness, sturdiness,
        quality of the egg tray, lid material, heating plate material, and how solid or cheap it feels.
        """

        functional_query = f"""
        Functional descriptions of "{self.product_name}" that imply structure:
        boiling or steaming eggs, number of layers, presence of poaching tray or omelet tray,
        water measuring cup, steam vent placement, buzzer or alarm, lid behavior when cooking.
        Focus only on details that help infer physical layout or components.
        """

        positive_query = f"""
        Positive comments about the visual design and appearance of "{self.product_name}":
        clean or compact look, cute or modern style, nice colors, visually appealing lid,
        attractive proportions, good-looking size on the countertop.
        """

        negative_query = f"""
        Complaints about the appearance or structure of "{self.product_name}":
        looks cheap, too small or bulky, flimsy lid, unstable base, poor-quality plastic,
        ugly color, awkward proportions, confusing layout of buttons or trays.
        """

        # ---------- Retrieve for each intent ----------
        top_visual = self.search(visual_query,     top_k=300)
        top_func   = self.search(functional_query, top_k=250)
        top_mat    = self.search(materials_query,  top_k=250)

        # core visual reviews
        core_visual_df = pd.concat([top_visual, top_func, top_mat], ignore_index=True)
        core_visual_df = core_visual_df.drop_duplicates(subset=[self.text_col]).reset_index(drop=True)

        # positive visual
        top_pos = self.search(positive_query, top_k=200)
        pos_df = top_pos.drop_duplicates(subset=[self.text_col]).reset_index(drop=True)

        # nagetive visual
        top_neg = self.search(negative_query, top_k=120)
        neg_df = top_neg.drop_duplicates(subset=[self.text_col]).reset_index(drop=True)

        print(f"[Retrieval] core_visual={len(core_visual_df)}, pos={len(pos_df)}, neg={len(neg_df)}")
        return core_visual_df, pos_df, neg_df


#===== PromptAgent ======

client = OpenAI()
class PromptAgent:
    """
    Use LLM to convert review corpora into structured prompt JSON.
    Supports injecting product_name and example review snippets.
    """

    def __init__(self, model_name: str = "gpt-4o"):
        self.model_name = model_name

    def _call_llm_json(self, system_prompt: str, user_prompt: str, max_tokens: int = 1800) -> dict:
        """Call LLM and force the result to be valid JSON."""
        resp = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",  "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=max_tokens,
        )

        content = resp.choices[0].message.content.strip()
        # Remove markdown ```json ... ``` wrapper if present
        if content.startswith("```"):
            # split on first ```
            parts = content.split("```", 1)
            # parts[1] might conatin "json\n{...}"
            content = parts[1]
            if content.lstrip().startswith("json"):
                content = content.lstrip()[4:] # remove 'json'
        content = content.strip()

            # Try to parse JSON
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # Attempt to extract JSON object from the text
            start = content.find("{")
            end = content.rfind("}")
            if start != -1 and end != -1 and end > start:
                json_str = content[start:end+1]
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError as e2:
                    print("❌ Failed to parse JSON after trimming to outer braces.")
                    print("Excerpt of model output:")
                    print(content[:1000])
                    raise e2
            else:
                print("❌ Model output does not contain a clear JSON object.")
                print("Excerpt of model output:")
                print(content[:1000])
                raise

    # ------------ BUILD FEATURE SUMMARY ------------------------
    def build_feature_summary(
        self,
        core_visual_df: pd.DataFrame,
        product_name: str,
        feature_prompt_template: str,
        text_col: str = "review",
        n_examples: int = 3,
    ) -> dict:
        """
        core_visual_df → feature JSON + global image_prompt.
        You pass your own prompt template via feature_prompt_template.
        """

        # Combine all reviews
        full_text = "\n\n---REVIEW---\n\n".join(core_visual_df[text_col].tolist())

        # Take a few examples for context
        examples = core_visual_df[text_col].head(n_examples).tolist()
        examples_block = "\n".join(f"- {r}" for r in examples)

        # Inject into your template
        user_prompt = feature_prompt_template.format(
            PRODUCT_NAME=product_name,
            FULL_TEXT=full_text,
            EXAMPLES=examples_block,
            COUNT=len(core_visual_df)
        )

        return self._call_llm_json(
            system_prompt="You are an expert product analyst. Output only valid JSON.",
            user_prompt=user_prompt,
        )

    # ------------ BUILD POS / NEG PROMPTS ------------------------
    def build_pos_neg_prompts(
        self,
        pos_df: pd.DataFrame,
        neg_df: pd.DataFrame,
        product_name: str,
        posneg_prompt_template: str,
        text_col: str = "review",
        n_pos_examples: int = 5,
        n_neg_examples: int = 5,
    ) -> dict:
        """
        pos_df + neg_df → positive/negative features + ideal/realistic prompts.
        Template is fully controlled by the user.
        """

        # Combine full positive/negative text
        pos_full = "\n\n---POSITIVE REVIEW---\n\n".join(pos_df[text_col].tolist())
        neg_full = "\n\n---NEGATIVE REVIEW---\n\n".join(neg_df[text_col].tolist())

        # Example subsets
        pos_examples = "\n".join(f"- {r}" for r in pos_df[text_col].head(n_pos_examples))
        neg_examples = "\n".join(f"- {r}" for r in neg_df[text_col].head(n_neg_examples))

        user_prompt = posneg_prompt_template.format(
            PRODUCT_NAME=product_name,
            POS_EXAMPLES=pos_examples,
            NEG_EXAMPLES=neg_examples,
            POS_FULL=pos_full,
            NEG_FULL=neg_full,
            POS_COUNT=len(pos_df),
            NEG_COUNT=len(neg_df),
        )

        return self._call_llm_json(
            system_prompt="You are an expert product analyst. Output only valid JSON.",
            user_prompt=user_prompt,
        )

#==== ImageAgent ======
class ImageAgent:
    """Generate images from prompts using different back-end models."""

    def __init__(self, 
                 imagen_client: genai.Client | None = None,
                 sd_api_key_env: str = "STABILITY_API_KEY"):
        self.imagen_client = imagen_client
        self.sd_api_key_env = sd_api_key_env

    # ------------ DALL-E 3 ------------
    def generate_with_dalle3(
        self,
        prompts: dict[str, str],
        output_folder: str,
        prefix: str = "dalle3"
    ) -> dict:
        """
        Use OpenAI DALL-E 3 to generate one image per prompt.
        Saves images to output_folder and returns {prompt_name: image_path}.
        use previous `client = OpenAI()`。
        """
        os.makedirs(output_folder, exist_ok=True)

        dalle_images: dict[str, str] = {}

        for prompt_name, prompt_text in prompts.items():
            print(f"\n[DALL-E 3] Generating: {prompt_name}...")

            if not prompt_text:
                print(f"[DALL-E 3] Skipping '{prompt_name}' because prompt text is empty.")
                continue

            try:
                response = client.images.generate(
                    model="dall-e-3",
                    prompt=prompt_text,
                    size="1024x1024",
                    quality="standard",
                    n=1,
                )

                image_url = response.data[0].url
                dalle_images[prompt_name] = image_url  

                # download and save image locally
                img_response = requests.get(image_url)
                img = Image.open(BytesIO(img_response.content))

                filename = f"{prefix}_{prompt_name}.png"
                save_path = os.path.join(output_folder, filename)
                img.save(save_path)

                # update path instead of URL
                dalle_images[prompt_name] = save_path

                print(f"[DALL-E 3] Saved: {save_path}")

            except Exception as e:
                print(f"[DALL-E 3] Error for '{prompt_name}': {e}")

        # generate comparison image
        if dalle_images:
            try:
                n = len(dalle_images)
                fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))
                if n == 1:
                    axes = [axes]

                for ax, (prompt_name, img_path) in zip(axes, dalle_images.items()):
                    img = Image.open(img_path)
                    ax.imshow(img)
                    ax.set_title(f"DALL-E 3: {prompt_name}", fontsize=10)
                    ax.axis("off")

                comp_path = os.path.join(output_folder, f"{prefix}_comparison.png")
                plt.tight_layout()
                plt.savefig(comp_path, dpi=300, bbox_inches="tight")
                plt.close(fig)
                print(f"[DALL-E 3] Saved comparison image: {comp_path}")
            except Exception as e:
                print(f"[DALL-E 3] Failed to create comparison image: {e}")

        return dalle_images

    # ------------ Stable Diffusion (Stability API) ------------
    def generate_with_stable_diffusion(
        self,
        prompts: dict[str, str],
        output_folder: str,
        prefix: str = "sd"
    ) -> dict:
        """
        Use Stability AI SDXL API to generate one image per prompt.
        Saves images to output_folder and returns {prompt_name: image_path}.
        """
        os.makedirs(output_folder, exist_ok=True)

        api_key = os.environ.get(self.sd_api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Stable Diffusion API key not found. "
                f"Set environment variable {self.sd_api_key_env}."
            )

        sd_images: dict[str, str] = {}

        for prompt_name, prompt_text in prompts.items():
            print(f"\n[SD] Generating: {prompt_name}...")

            if not prompt_text:
                print(f"[SD] Skipping '{prompt_name}' because prompt text is empty.")
                continue

            try:
                response = requests.post(
                    "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image",
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {api_key}"
                    },
                    json={
                        "text_prompts": [{"text": prompt_text, "weight": 1}],
                        "cfg_scale": 7,
                        "height": 1024,
                        "width": 1024,
                        "samples": 1,
                        "steps": 30
                    },
                    timeout=60,
                )

                if response.status_code == 200:
                    data = response.json()
                    img_data = base64.b64decode(data["artifacts"][0]["base64"])
                    img = Image.open(BytesIO(img_data))

                    filename = f"{prefix}_{prompt_name}.png"
                    save_path = os.path.join(output_folder, filename)
                    img.save(save_path)
                    sd_images[prompt_name] = save_path
                    print(f"[SD] Saved: {save_path}")
                else:
                    print(f"[SD] Error for '{prompt_name}': {response.status_code} - {response.text}")

            except Exception as e:
                print(f"[SD] Exception for '{prompt_name}': {e}")

        # generate comparison image
        if sd_images:
            try:
                n = len(sd_images)
                fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))
                if n == 1:
                    axes = [axes]

                for ax, (prompt_name, img_path) in zip(axes, sd_images.items()):
                    img = Image.open(img_path)
                    ax.imshow(img)
                    ax.set_title(f"Stable Diffusion: {prompt_name}", fontsize=10)
                    ax.axis("off")

                comp_path = os.path.join(output_folder, f"{prefix}_comparison.png")
                plt.tight_layout()
                plt.savefig(comp_path, dpi=300, bbox_inches="tight")
                plt.close(fig)
                print(f"[SD] Saved comparison image: {comp_path}")
            except Exception as e:
                print(f"[SD] Failed to create comparison image: {e}")

        return sd_images

    # ------------ Google Imagen(Nano banana) ------------
    def generate_with_nano_banana(
        self,
        prompts: dict[str, str],
        output_folder: str,
        prefix: str = "gimg"
    ) -> dict:
        """
        Use Google GenAI Imagen (imagen-4.0-generate-001) to generate one image per prompt.
        Saves images to output_folder and returns {prompt_name: image_path}.
        Need GOOGLE_API_KEY set in env.
        """
        os.makedirs(output_folder, exist_ok=True)

        client = self.imagen_client or genai.Client()

        gimg_images: dict[str, str] = {}

        try:
            # Generate images with Imagen model
            for prompt_name, prompt_text in prompts.items():
                print(f"\n[Imagen] Generating: {prompt_name}...")

                if not prompt_text:
                    print(f"[Imagen] Skipping '{prompt_name}' because prompt text is empty.")
                    continue

                response = client.models.generate_images(
                    model="imagen-4.0-generate-001",
                    prompt=prompt_text,
                    config=types.GenerateImagesConfig(
                        number_of_images=1,
                    ),
                )

                if not response.generated_images:
                    print(f"[Imagen] No images returned for prompt: {prompt_name}")
                    continue

                generated_image = response.generated_images[0]
                img = generated_image.image  # PIL Image object

                filename = f"{prefix}_{prompt_name}.png"
                save_path = os.path.join(output_folder, filename)
                img.save(save_path)
                gimg_images[prompt_name] = save_path

                print(f"[Imagen] Saved: {save_path}")

            # -----------------------------
            # show generated images & save comparison
            # -----------------------------
            if not gimg_images:
                print("[Imagen] No images were generated, nothing to display.")
            else:
                print("\n--- Imagen (Google GenAI) Results ---")

                prompt_names = list(gimg_images.keys())
                n = len(prompt_names)

                fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))
                if n == 1:
                    axes = [axes]

                for ax, prompt_name in zip(axes, prompt_names):
                    img_path = gimg_images[prompt_name]
                    if os.path.exists(img_path):
                        img = Image.open(img_path)
                        ax.imshow(img)
                        ax.set_title(f"Imagen: {prompt_name}", fontsize=12)
                        ax.axis("off")
                    else:
                        ax.set_title(f"Missing: {prompt_name}", fontsize=12)
                        ax.axis("off")

                plt.tight_layout()
                comp_save_path = os.path.join(output_folder, f"{prefix}_comparison.png")
                plt.savefig(comp_save_path, dpi=300, bbox_inches="tight")
                plt.close(fig)
                print(f"[Imagen] Comparison image saved to: {comp_save_path}")

        except Exception as e:
            print(f"[Imagen] Error while generating images: {e}")

        return gimg_images
    
    
    def run(self, image_model: str, prompts: dict[str, str], output_folder: str) -> dict:
        """Dispatch to the chosen model."""
        os.makedirs(output_folder, exist_ok=True)
        model = image_model.lower()
        if model == "dalle":
            return self.generate_with_dalle3(prompts, output_folder, prefix="dalle")
        elif model == "sd":
            return self.generate_with_stable_diffusion(prompts, output_folder, prefix="sd")
        elif model == "nano":
            return self.generate_with_nano_banana(prompts, output_folder, prefix="nano")
        else:
            raise ValueError(f"Unknown image model: {image_model}")
        

def run_workflow(
    csv_path: str,
    output_root: str,
    image_model: str = "dalle",       # "dalle" / "sd" / "nano"
    sampling_mode: str = "random",    # "random" / "cluster" / "rag"
    text_col: str = "review",
    product_name: str = "Generic Product" 
):
    """
    Full pipeline: CSV → Clean → Embedding → Cluster → RAG → LLM Prompt → Image Generation.
    """

    print("\n======== RUN WORKFLOW ========")
    print(f"CSV File:       {csv_path}")
    print(f"Output Folder:  {output_root}")
    print(f"Image Model:    {image_model}")
    print(f"Sampling Mode:  {sampling_mode}")
    print("================================\n")

     # --- Create output paths ---
    os.makedirs(output_root, exist_ok=True)

    # artifacts
    artifacts_dir = os.path.join(output_root, "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)

    # images
    run_tag = f"{image_model}_{sampling_mode}"
    images_dir = os.path.join(output_root, f"images_{run_tag}")
    os.makedirs(images_dir, exist_ok=True)

    print(f"[Paths] Artifacts dir: {artifacts_dir}")
    print(f"[Paths] Images dir:    {images_dir}")

    # --- Initialize Agents ---
    ingestion_agent   = IngestionAgent(review_col=text_col)
    embedding_agent   = EmbeddingAgent(model_name="all-MiniLM-L6-v2")
    clustering_agent  = ClusteringAgent(n_clusters=8)
    rag_store_agent   = RAGStoreAgent()
    prompt_agent      = PromptAgent(model_name="gpt-4o")

    # Google Imagen Client
    imagen_client = genai.Client()
    image_agent = ImageAgent(imagen_client=imagen_client)

    # --- 1. Load & Clean ---
    df = ingestion_agent.run(csv_path)

    # --- 2. Embedding ---
    emb_path = os.path.join(artifacts_dir, "embeddings.npy")
    embeddings = embedding_agent.run(df, text_col=text_col, save_path=emb_path)

    # --- 3. Clustering ---
    df_clustered_csv = os.path.join(artifacts_dir, "reviews_with_clusters.csv")
    df = clustering_agent.run(df, embeddings, out_csv=df_clustered_csv)

    # --- 4. RAG Store ---
    index = rag_store_agent.build_index(embeddings)

    # --- 5. Retrieval (RAG / Cluster / Random) ---
    ret = RetrievalAgent(
        embedding_agent=embedding_agent,
        index=index,
        df=df,
        product_name=product_name,
        text_col=text_col
    )

    if sampling_mode == "rag":
        core_visual_df, pos_df, neg_df = ret.build_corpora()
    elif sampling_mode == "cluster":
        from random import sample
        # cluster sampling: sample up to 100 reviews per cluster
        sampled = pd.concat([
            df[df["cluster"] == c].sample(min(100, len(df[df["cluster"] == c])), random_state=42)
            for c in df["cluster"].unique()
        ])
        core_visual_df = sampled
        pos_df = sampled.sample(100, random_state=42)
        neg_df = sampled.sample(50, random_state=42)
    elif sampling_mode == "random":
        sampled = df.sample(750, random_state=42)
        core_visual_df = sampled
        pos_df = sampled.sample(100, random_state=42)
        neg_df = sampled.sample(50, random_state=42)
    else:
        raise ValueError(f"Invalid sampling_mode: {sampling_mode}")

    # --- 6. LLM Prompts ---
    feature_json = prompt_agent.build_feature_summary(
        core_visual_df=core_visual_df,
        product_name=product_name,
        feature_prompt_template=feature_prompt_template,
    )

    pos_neg_json = prompt_agent.build_pos_neg_prompts(
        pos_df=pos_df,
        neg_df=neg_df,
        product_name=product_name,
        posneg_prompt_template=posneg_prompt_template,
    )

    prompts = {
        "feature_based": feature_json.get("image_prompt", ""),
        "ideal_from_pos": pos_neg_json.get("ideal_image_prompt", ""),
        "realistic_from_pos_neg": pos_neg_json.get("realistic_image_prompt", "")
    }

    # --- 7. Image Generation ---
    image_agent.run(
        image_model=image_model,
        prompts=prompts,
        output_folder=images_dir,
    )

    print("\n======== WORKFLOW COMPLETED ========\n")
    return {
        "core_visual_df": core_visual_df,
        "pos_df": pos_df,
        "neg_df": neg_df,
        "prompts": prompts,
        "images_dir": images_dir
    }
    
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run full review-to-image workflow.")

    parser.add_argument("--csv", required=True, help="Path to review CSV file.")
    parser.add_argument("--out", required=True, help="Output directory root.")
    parser.add_argument("--image_model", default="sd",
                        choices=["sd", "dalle", "nano"],
                        help="Choose image model: sd / dalle / nano.")
    parser.add_argument("--sampling_mode", default="rag",
                        choices=["random", "cluster", "rag"],
                        help="How to select ~800 reviews.")

    args = parser.parse_args()

    run_workflow(
        csv_path=args.csv,
        output_root=args.out,
        image_model=args.image_model,
        sampling_mode=args.sampling_mode,
    )