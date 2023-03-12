import torch
from diffusers import StableDiffusionPipeline
from hf_token import hf_token

device = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

# if hf_token == "YOUR_HF_TOKEN":
#     # read users token from cli
#     hf_token = input("Please enter your HuggingFace token: ")

try:
    # grab the model from HF using your token
    model_id = "stabilityai/stable-diffusion-2-1-base"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, revision="fp16", torch_dtype=torch.float16).to(device)

    # save the model
    pipe.save_pretrained("model")

except OSError as e:
    print(e)
    print("Invalid Token, Please Try Again")

