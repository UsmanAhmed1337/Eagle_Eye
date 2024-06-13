import gradio as gr
import numpy as np
from main import pose_predict

def flip(im):
    return np.flipud(im)

demo = gr.Interface(
    pose_predict, 
    gr.Image(sources=["webcam"], streaming=True), 
    "image",
    live=True
)
demo.launch(share=True)
    