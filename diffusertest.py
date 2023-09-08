#### Modules 
## pip install diffusers
## pip install transformers accelerate

from diffusers import StableDiffusionPipeline
from PIL import Image
import torch

model_id = "dreamlike-art/dreamlike-diffusion-1.0"
 
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
 
pipe = pipe.to("cuda")

prompt = "a grungy woman with rainbow hair, travelling between dimensions, dynamic pose, happy, soft eyes and narrow chin, extreme bokeh, dainty figure, long hair straight down, torn kawaii shirt and baggy jeans, In style of by Jordan Grimmer and greg rutkowski, crisp lines and color, complex background, particles, lines, wind, concept art, sharp focus, vivid colors"
 
image = pipe(prompt).images[0]
 
image.show()
