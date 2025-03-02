# Import Libraries
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel
from diffusers import StableDiffusionPipeline
import torch
import uvicorn
import io
import base64
from PIL import Image

# Load the model
print("############# Load model #################")
model_name = "OFA-Sys/small-stable-diffusion-v0"
pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float32)
pipeline_t2i = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
#pipeline_t2i = pipe.to("cpu")

print("############# Initiate fastapi #################")
app = FastAPI(debug=True)

# Define request model
class PromptRequest(BaseModel):
    prompt: str

@app.post("/generate")
async def generate_image(data: PromptRequest):
    try:
        #print("Generating Images!")
        image = pipeline_t2i(prompt=data.prompt).images[0]

        # Convert image to base64
        img_io = io.BytesIO()
        image.save(img_io, format="PNG")
        img_io.seek(0)
        img_base64 = base64.b64encode(img_io.getvalue()).decode("utf-8")

        return {"image": img_base64}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.get("/")
async def main(request: Request):
    return HTMLResponse(
        """
    <html>
    <body>
    <h1>Image Generator</h1>
    <h2>Eg. A baby penguin watches the sunset</h2>
    <input type="text" id="prompt" placeholder="Enter a prompt">
    <button onclick="submitPrompt()">Generate Image</button>
    <script>
        async function submitPrompt() {
            let prompt = document.getElementById("prompt").value;
            let response = await fetch("/generate", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ prompt: prompt })
            });
            let data = await response.json();
            if (data.image) {
                let img = document.createElement("img");
                img.src = "data:image/png;base64," + data.image;
                document.body.appendChild(img);
            } else {
                alert("Error: " + data.detail);
            }
        }
    </script>
    </body>
    </html>
    """
        )

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)







