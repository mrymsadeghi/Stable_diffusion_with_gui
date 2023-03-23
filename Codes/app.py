import tkinter as tk
import customtkinter as ctk 
import os
from PIL import Image, ImageTk
from authtoken import auth_token

import torch
from torch import autocast
from diffusers import StableDiffusionPipeline 

# Create the app
app = tk.Tk()
app.geometry("1000x660")
app.title("Stable Diffusion guidance scale testing") 
ctk.set_appearance_mode("dark") 

G_SCALES = [7, 14, 20]
root_path = "C:\PyProjects\StableDiffusion\images"

img1 = Image.open(os.path.join(root_path, "robotic bird_7.png"))
img2 = Image.open(os.path.join(root_path, "robotic bird_16.png"))
img3 = Image.open(os.path.join(root_path, "robotic bird_20.png"))
h = 300
w1 = int((h/img1.size[1])*img1.size[0])
w2 = int((h/img2.size[1])*img2.size[0])
w3 = int((h/img3.size[1])*img3.size[0])
img1 = img1.resize((w1, h), Image.ANTIALIAS)
img2 = img2.resize((w2, h), Image.ANTIALIAS)
img3 = img3.resize((w3, h), Image.ANTIALIAS)
generated_images = [img1, img2, img3]
prompt = ctk.CTkEntry(height=40, width=512, text_font=("Arial", 20), text_color="black", fg_color="white") 
prompt.place(x=220, y=30)

lmain1 = ctk.CTkLabel(height=512, width=300)
lmain1.place(x=10, y=110)
lmain2 = ctk.CTkLabel(height=512, width=300)
lmain2.place(x=320, y=110)
lmain3 = ctk.CTkLabel(height=512, width=300)
lmain3.place(x=630, y=110)
label1 = ctk.CTkLabel(text=f"Guidance Scale: {str(G_SCALES[0])}", text_font=("Arial", 14), text_color="blue")
label1.place(x=30, y=540)
label2 = ctk.CTkLabel(text=f"Guidance Scale: {str(G_SCALES[1])}", text_font=("Arial", 14), text_color="blue")
label2.place(x=340, y=540)
label3 = ctk.CTkLabel(text=f"Guidance Scale: {str(G_SCALES[2])}", text_font=("Arial", 14), text_color="blue")
label3.place(x=650, y=540)

modelid = "CompVis/stable-diffusion-v1-4"
device = "cpu"
pipe = StableDiffusionPipeline.from_pretrained(modelid, revision="fp16", torch_dtype=torch.float16, use_auth_token=auth_token) 
pipe.to(device) 

def generate(): 
    with autocast(device): 
        g_prompt = prompt.get()
        generated_images = [pipe(g_prompt, guidance_scale=g_scale)["sample"][0] for g_scale in G_SCALES]
                
        photo_images = []
        for i, image in enumerate(generated_images):    
            photo_image = ImageTk.PhotoImage(image)
            photo_images.append(photo_image)
            if i == 0:
                lmain1.configure(image=photo_image)
            elif i == 1:
                lmain2.configure(image=photo_image)
            elif i == 2:
                lmain3.configure(image=photo_image)

trigger = ctk.CTkButton(height=40, width=120, text_font=("Arial", 20), text_color="white", fg_color="blue") 
trigger.configure(text="Generate") 
trigger.place(x=406, y=80) 

app.mainloop()