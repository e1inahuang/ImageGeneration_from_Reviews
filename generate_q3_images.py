
import os
import json
import requests
import time
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

if not OPENAI_API_KEY or not HF_TOKEN:
    print("Error: Missing API keys in .env file")
    exit(1)

client = OpenAI(api_key=OPENAI_API_KEY)

# Define Prompts
# Prompt 1: Derived from Q2 Reviews (The "Voice of Customer" prompt)
prompt_1 = "A high-quality, professional product shot of a modern cycling helmet in matte black with electric blue accents. The helmet features a futuristic design with a magnetic dark-tinted eye shield (goggles) attached to the front. On the back, there is a glowing red LED safety light. The helmet has a streamlined shape with multiple ventilation holes for airflow. It sits on a clean, neutral studio background with soft lighting to highlight the sleek texture and the reflection on the magnetic goggles."

# Prompt 2: Derived from Q1 Description (The "Official Specs" prompt)
prompt_2 = "A VICTGOAL bike helmet with detachable magnetic goggles and a removable sun visor. The helmet is dual-tone fluorescent yellow and black, emphasizing high visibility and safety. It features a rechargeable USB LED light on the rear with 3 lighting modes. The design includes 21 breathable vents for cooling. Shown in a dynamic outdoor setting, resting on a park bench with a blurred bicycle in the background."

# Prompt 3: Lifestyle/Action Shot (The "In-Use" prompt)
prompt_3 = "Cinematic action shot of a cyclist wearing a sleek white VICTGOAL helmet with a magnetic grey visor. The cyclist is riding through a city street at twilight. The rear red LED light on the helmet is illuminated and glowing brightly. The image has a shallow depth of field, focusing on the helmet's aerodynamic shape and the reflection of city lights on the visor. High resolution, photorealistic."

prompts = [
    {"id": "p1_reviews", "text": prompt_1, "desc": "Review-Based (Q2)"},
    {"id": "p2_specs", "text": prompt_2, "desc": "Description-Based (Q1)"},
    {"id": "p3_action", "text": prompt_3, "desc": "Lifestyle/Action"}
]

def generate_dalle3(prompt, filename):
    print(f"Generating DALL-E 3 image for: {filename}...")
    try:
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1
        )
        image_url = response.data[0].url
        
        # Download
        img_data = requests.get(image_url).content
        with open(filename, 'wb') as f:
            f.write(img_data)
        print(f"✓ Saved {filename}")
        return True
    except Exception as e:
        print(f"✗ Failed DALL-E 3: {e}")
        return False

def generate_sdxl(prompt, filename):
    print(f"Generating SDXL image for: {filename}...")
    API_URL = "https://router.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    
    payload = {
        "inputs": prompt,
        "parameters": {
            "negative_prompt": "blurry, low quality, distorted, deformed, ugly, bad anatomy",
            "num_inference_steps": 30,
            "guidance_scale": 7.5
        }
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        
        if response.status_code == 200:
            with open(filename, 'wb') as f:
                f.write(response.content)
            print(f"✓ Saved {filename}")
            return True
        else:
            print(f"✗ Failed SDXL: {response.status_code} - {response.text}")
            # If model is loading, wait and retry
            if "estimated_time" in response.text:
                wait_time = response.json().get("estimated_time", 10)
                print(f"Model loading, waiting {wait_time:.1f}s...")
                time.sleep(wait_time + 2)
                return generate_sdxl(prompt, filename)
            return False
            
    except Exception as e:
        print(f"✗ Failed SDXL: {e}")
        return False

# Execution Loop
print("Starting Q3 Image Generation...")
print(f"Estimated Cost (DALL-E 3): ${len(prompts) * 0.040:.2f}")

results = []

for p in prompts:
    # Generate with DALL-E 3
    dalle_filename = f"q3_dalle3_{p['id']}.png"
    success_dalle = generate_dalle3(p['text'], dalle_filename)
    
    # Generate with SDXL
    sdxl_filename = f"q3_sdxl_{p['id']}.png"
    success_sdxl = generate_sdxl(p['text'], sdxl_filename)
    
    results.append({
        "prompt_id": p['id'],
        "description": p['desc'],
        "prompt_text": p['text'],
        "dalle_file": dalle_filename if success_dalle else "FAILED",
        "sdxl_file": sdxl_filename if success_sdxl else "FAILED"
    })

# Save results log
with open("q3_image_generation_log.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nGeneration Complete!")
