# import fastapi
from fastapi import FastAPI, UploadFile, File
from starlette.responses import StreamingResponse
from diffusers import StableDiffusionPipeline
import base64
import uvicorn
import io
import torch
from io import BytesIO

# instantiate the app
app = FastAPI()

# cuda or cpu config
def get_device():
    if torch.cuda.is_available():
        print('cuda is available')
        return torch.device('cuda')
    else:
        print('cuda is not available')
        return torch.device('cpu')

# create a route
@app.get("/")
def index():
    return {"text" : "We're running!"}

# create a text2img route
@app.post("/text2img")
def text2img(model_inputs:dict):
    device = get_device()

    model = StableDiffusionPipeline.from_pretrained("../model")
    model.to(device)

    model.enable_attention_slicing()
    model.enable_vae_slicing()
                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
    # Run the pipeline, showing some of the available arguments
    pipe_output = model(
        prompt=model_inputs.get('prompt', None), # What to generate
        negative_prompt="Oversaturated, blurry, low quality", # What NOT to generate
        height=480, width=640,     # Specify the image size
        guidance_scale=12,          # How strongly to follow the prompt
        num_inference_steps=35,    # How many steps to take
        #generator=generator        
    )

    #del model
    # if torch.cuda.is_available():
    #     torch.cuda.empty_cache()

    image = pipe_output.images[0]
    buffered = BytesIO()
    image.save(buffered,format="JPEG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    # Return the results as a dictionary
    return {'image_base64': image_base64}

# run the app
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)





