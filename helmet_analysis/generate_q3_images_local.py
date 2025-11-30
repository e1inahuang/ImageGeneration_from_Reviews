
import torch
from diffusers import StableDiffusionPipeline
import os

# Configuration
OUTPUT_DIR = "q3_generated_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Check for MPS (Apple Silicon) or CPU
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# Prompts from Q2/Q3 planning
prompts = [
    {
        "id": "p1_reviews",
        "text": "A high-quality, professional product shot of a modern cycling helmet in matte black with electric blue accents. The helmet features a futuristic design with a magnetic dark-tinted eye shield (goggles) attached to the front. On the back, there is a glowing red LED safety light. The helmet has a streamlined shape with multiple ventilation holes for airflow. It sits on a clean, neutral studio background with soft lighting to highlight the sleek texture and the reflection on the magnetic goggles.",
        "desc": "Review-Based"
    },
    {
        "id": "p2_specs",
        "text": "A VICTGOAL bike helmet with detachable magnetic goggles and a removable sun visor. The helmet is dual-tone fluorescent yellow and black, emphasizing high visibility and safety. It features a rechargeable USB LED light on the rear with 3 lighting modes. The design includes 21 breathable vents for cooling. Shown in a dynamic outdoor setting, resting on a park bench with a blurred bicycle in the background.",
        "desc": "Description-Based"
    },
    {
        "id": "p3_action",
        "text": "Cinematic action shot of a cyclist wearing a sleek white VICTGOAL helmet with a magnetic grey visor. The cyclist is riding through a city street at twilight. The rear red LED light on the helmet is illuminated and glowing brightly. The image has a shallow depth of field, focusing on the helmet's aerodynamic shape and the reflection of city lights on the visor. High resolution, photorealistic.",
        "desc": "Lifestyle/Action"
    }
]

# Model 1: Stable Diffusion v1.5
# print("\n" + "="*50)
# print("Loading Model 1: Stable Diffusion v1.5")
# print("="*50)

# model_id_1 = "runwayml/stable-diffusion-v1-5"
# # pipe1 = StableDiffusionPipeline.from_pretrained(model_id_1, torch_dtype=torch.float32)
# # pipe1 = pipe1.to(device)

# # if device == "mps":
# #     pipe1.enable_attention_slicing()

# # for p in prompts:
# #     filename = f"{OUTPUT_DIR}/model1_sd15_{p['id']}.png"
# #     if not os.path.exists(filename):
# #         print(f"Generating {p['id']} with Model 1...")
# #         image = pipe1(p['text'], num_inference_steps=30).images[0]
# #         image.save(filename)
# #         print(f"Saved {filename}")
# #     else:
# #         print(f"Skipping {filename} (already exists)")

# # del pipe1
# # torch.cuda.empty_cache() if torch.cuda.is_available() else None

# Model 2: OpenJourney (Midjourney Style) - Non-gated alternative
print("\n" + "="*50)
print("Loading Model 2: OpenJourney (prompthero/openjourney)")
print("="*50)

model_id_2 = "prompthero/openjourney"
pipe2 = StableDiffusionPipeline.from_pretrained(model_id_2, torch_dtype=torch.float32)
pipe2 = pipe2.to(device)

if device == "mps":
    pipe2.enable_attention_slicing()

for p in prompts:
    # Add "mdjrny-v4 style" to prompt as recommended for this model
    prompt_text = "mdjrny-v4 style " + p['text']
    print(f"Generating {p['id']} with Model 2...")
    image = pipe2(prompt_text, num_inference_steps=30).images[0]
    filename = f"{OUTPUT_DIR}/model2_openjourney_{p['id']}.png"
    image.save(filename)
    print(f"Saved {filename}")

print("\nGeneration Complete!")
