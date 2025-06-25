import torch
import torchvision.transforms as v2
from PIL import Image
import matplotlib.pyplot as plt
from transformers import DeiTImageProcessor, RobertaTokenizer, TrOCRProcessor,VisionEncoderDecoderModel
from io import BytesIO
import cv2
import tempfile
import numpy as np
import streamlit as st


device = 'cuda' if torch.cuda.is_available() else 'cpu'

@st.cache_resource
def model_transform(image_processor, used_tokenizer, model_trained, device):
    image_processor=DeiTImageProcessor.from_pretrained(image_processor)
    tokenizer=RobertaTokenizer.from_pretrained(used_tokenizer)
    processor=TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)

    model = VisionEncoderDecoderModel.from_pretrained(model_trained).to(device)
    model.eval()
    return model, processor


def image_transform(img_):
    if isinstance(img_, np.ndarray):
        img_ = Image.fromarray(img_)
        
    transform = v2.Compose([
                v2.Resize((384,384)),
                v2.ToTensor(),
                v2.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ])
    img_t = transform(img_).unsqueeze(0).to(device)

    return img_t
    
def generate_text(transformed_img, model, processor, device):
    with torch.no_grad():  # Disable gradient computation for inference
        generated_ids = model.generate(transformed_img)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
    return generated_text

def save_text_as_file(text, filename="generated_text.txt"):
    """Save text as a downloadable file."""
    file = BytesIO()
    file.write(text.encode())  # Convert text to bytes
    file.seek(0)
    
    return file

def extract_words_from_image(image_path):
    
    # Load image
    image = cv2.imread(image_path)

    if image is None:
        st.error("Error: Could not read the image. Please try again.")
        return []
    
    # Grayscale image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding (convert to binary image)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    # Find contours of text regions
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    word_images = []
    
    for i, contour in enumerate(contours):
        # Get bounding box of the word
        x, y, w, h = cv2.boundingRect(contour)
        
        # Crop the word from original image
        word_img = image[y:y+h, x:x+w]

        # Save the word image
        word_images.append(word_img)
        # cv2.imwrite(f"{save_folder}/word_{i}.png", word_img)

    return word_images

def main():
    im_processor="facebook/deit-base-distilled-patch16-224"
    us_tokenizer="flax-community/roberta-hindi"
    trained_model="model"
    t_model, t_processor = model_transform(im_processor, us_tokenizer, trained_model, device)

    # Page Title 
    st.title("Handwritten Hindi Sentence Image To Text Generator")
    st.write("Upload an image to generate its text.")

    # Upload Image
    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])
    
    st.write(uploaded_file)
    if uploaded_file is not None:
    # Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
            temp_file.write(uploaded_file.getbuffer())
            temp_filepath = temp_file.name  # Get file path

        st.image(temp_filepath, caption="Uploaded Image")
        sent=""
        # Generate Text
        words=extract_words_from_image(temp_filepath)
        for img in words:
            # Transform Image
            transform_img=image_transform(img)
            text = generate_text(transform_img, t_model, t_processor, device)
            sent+=text+" "
        
        # Display Text:
        if sent:
            st.write(f"**Text generated:** {sent}")
            
            file = save_text_as_file(sent)

            st.download_button(label="Download Text File",
                                data=file,
                                file_name="generated_text.txt",
                                mime="text/plain"
                            )

if __name__ == "__main__":
    main()
