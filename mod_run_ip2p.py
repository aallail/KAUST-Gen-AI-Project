import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
from torchvision import transforms
import os
from diffusers import StableDiffusionInstructPix2PixPipeline
import PIL
from diffusers.loaders import TextualInversionLoaderMixin
import transformers
import argparse
def embed_prompt(prompt, pipe, device):
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        print("none of the above")
    prompt_embeds = None
    if prompt_embeds is None:
        # textual inversion: process multi-vector tokens if necessary
        if isinstance(pipe, TextualInversionLoaderMixin):
            prompt = pipe.maybe_convert_prompt(prompt, pipe.tokenizer)
        
        text_inputs = pipe.tokenizer(
            prompt,
            padding="max_length",
            max_length= pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        
        text_input_ids = text_inputs.input_ids
        untruncated_ids = pipe.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
        
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
            text_input_ids, untruncated_ids
        ):
            removed_text = pipe.tokenizer.batch_decode(
                untruncated_ids[:, pipe.tokenizer.model_max_length - 1 : -1]
            )
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {pipe.tokenizer.model_max_length} tokens: {removed_text}"
            )
        
        if hasattr(pipe.text_encoder.config, "use_attention_mask") and pipe.text_encoder.config.use_attention_mask:
            attention_mask = text_inputs.attention_mask.to(device)
        else:
            attention_mask = None

        
        prompt_embeds = pipe.text_encoder(text_input_ids.to(device), attention_mask=attention_mask)
        #prompt_embeds = pipe.text_encoder(text_input_ids.to(device))
        prompt_embeds = prompt_embeds[0]
        
        if pipe.text_encoder is not None:
            prompt_embeds_dtype = pipe.text_encoder.dtype
        else:
            prompt_embeds_dtype = pipe.unet.dtype
        
        prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)
        return prompt_embeds
        
def combine_embed(model, processor, image, prompt, pipe, device):
    embed = embed_prompt(prompt, pipe, device).to(device)

    inputs = processor(images=torch.stack(image).to(device), return_tensors="pt")
    outputs = model(**inputs)
    last_hidden_state = outputs.last_hidden_state
    pooled_output = outputs.pooler_output.to(device)
    print(pooled_output.shape)
    adjusted = pooled_output.unsqueeze(1)
    combined = adjusted + embed
    return combined
class NumpyImageDataset(Dataset):
    def __init__(self, json_file, base_dir, transform=None):
        """
        Args:
            json_file (str): Path to the JSON file containing metadata.
            base_dir (str): Base directory where the image files are stored.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        with open(json_file, 'r') as file:
            self.data = json.load(file)
        self.base_dir = base_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        # Construct full file path
        file_path = os.path.join(self.base_dir, item['file_name'])
        if not file_path.endswith('.npy'):
            file_path += '.npy'
        
        # Load image data from numpy file
        image_data = np.load(file_path).astype(np.float32)[0]

        
        strings = (item['emotion'], item['description'], item['response'],  item['file_name'])
        return image_data, strings



transform = transforms.Compose([
    transforms.Resize((512, 512)),           
])

parser = argparse.ArgumentParser(description="Choose Path.")

# Add boolean arguments
parser.add_argument('--basel', action='store_true',  # Uses store_true to save True if --dir1 is specified
                    help='Enable feature for directory 1')
parser.add_argument('--prompt', action='store_true',  # Uses store_true to save True if --dir2 is specified
                    help='Enable feature for directory 2')

args = parser.parse_args()
if args.basel:
    folder1 = "/bases/"
else:
    folder1 = "/mods/"
if args.prompt:
    folder2 = "/short_filtered"
else:
    folder2= "long_filtered"
json_file = "/ibex/user/saghieim/finalproject/data/data_structures/" + folder2 + ".json"
image_dir = "/ibex/user/saghieim/finalproject/data/flintstones/ibex/ai/home/shenx/story/data/flintstones/video_frames_sampled_4x"
output_folder="/ibex/user/saghieim/finalproject/output/i2p2p/" + folder1 + folder2
output_folder="/ibex/user/saghieim/finalproject/output/i2p2p/samples166/" + folder1

dataset = NumpyImageDataset(json_file, image_dir)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
     "timbrooks/instruct-pix2pix", torch_dtype=torch.float16, safety_checker=None
 )
pipe = pipe.to("cuda")
if not args.basel:
    model = transformers.CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = transformers.AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
device = torch.device("cuda:0")
for images, tuples in dataloader:
    emotions, _, prompts, file_names = tuples
    if not args.basel:
        prompts = combine_embed(model, processor, list(images), list(prompts), pipe, device)
    emotions = list(emotions)
    file_names = list(file_names)
    array = images
    #images = np.transpose(images, (0, 3, 1, 2))
    images = [PIL.Image.fromarray(image.cpu().detach().numpy().astype(np.uint8),"RGB") for image in images]
    if args.basel:
        results = pipe(prompt=list(prompts), image=images, guidance_scale=4.5, image_guidance_scale=2.0)  # Assume pipeline can handle batch processing
    else:
        results = pipe(prompt_embeds=prompts, image=images, guidance_scale=5.5, image_guidance_scale=2.0)  # Assume pipeline can handle batch processing
    # Handle results
    for emotion, file_name, result in zip(emotions, file_names, results.images):
        output_dir = os.path.join(output_folder, emotion)
        os.makedirs(output_dir, exist_ok=True)

        # Define the output path for the image
        output_path = os.path.join(output_dir, f"{file_name}-{emotion}.png")
        
        # Save each image to its respective folder based on emotion
        result.save(output_path)
