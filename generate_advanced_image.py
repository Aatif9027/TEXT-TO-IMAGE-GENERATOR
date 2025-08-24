import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

def generate_image(text_prompt, output_path="generated_image.png"):
    # Load pretrained model and pipeline
    model_id = "runwayml/stable-diffusion-v1-5"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to(device)

    # Generate image
    image = pipe(text_prompt).images[0]

    # Save image
    image.save(output_path)
    print(f"Image saved at {output_path}")

if __name__ == "__main__":
    prompt = input("Enter your text prompt: ")
    generate_image(prompt)
