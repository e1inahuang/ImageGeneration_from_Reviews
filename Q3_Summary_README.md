# Q3 Image Generation Summary

## Overview
This section covers the "Creative Application" (Image Generation) requirement. We used two different Diffusion models to generate product visualizations based on prompts derived from Q1 (Description) and Q2 (Reviews).

## Files Generated
- **`Final_Project_Q3.docx`**: The final report containing prompts, generated images, and comparative analysis.
- **`generate_q3_images_local.py`**: The Python script used to generate images locally using Stable Diffusion.
- **`q3_generated_images/`**: Directory containing the 6 generated images.
    - `model1_sd15_*.png`: Images from Stable Diffusion v1.5.
    - `model2_openjourney_*.png`: Images from OpenJourney (Midjourney style).

## Models Used
1.  **Stable Diffusion v1.5**: Standard baseline model.
2.  **OpenJourney (prompthero/openjourney)**: Fine-tuned model for artistic/Midjourney-style outputs.

## Key Findings
- **Review-Based Prompts** (Q2) produced more "futuristic" and premium-looking results than description-based prompts.
- **OpenJourney** provided better lighting and dramatic composition compared to the flatter, more realistic look of **SD v1.5**.
- AI successfully visualized key features like "magnetic goggles" and "rear LED light" without having seen the specific product image during training (zero-shot visualization).
