import torch
import torchvision.transforms as v2
from PIL import Image
from transformers import DeiTImageProcessor, RobertaTokenizer, TrOCRProcessor,VisionEncoderDecoderModel
from io import BytesIO
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
    transform = v2.Compose([
                v2.Resize((384,384)),
                v2.ToTensor(),
                v2.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ])
    img_t = transform(img_).unsqueeze(0).to(device)

    return img_t
    
def generate_text(img, model, processor, device):
    # Transform Image
    transform_img=image_transform(img)
    with torch.no_grad():  # Disable gradient computation for inference
        generated_ids = model.generate(transform_img)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
    return generated_text

def save_text_as_file(text, filename="generated_text.txt"):
    """Save text as a downloadable file."""
    file = BytesIO()
    file.write(text.encode())  # Convert text to bytes
    file.seek(0)
    
    return file


def main():
    im_processor="facebook/deit-base-distilled-patch16-224"
    us_tokenizer="flax-community/roberta-hindi"
    trained_model="model"
    t_model, t_processor = model_transform(im_processor, us_tokenizer, trained_model, device)

    # Page Title 
    st.title("Handwritten Hindi One Word Image To Text Generator")
    st.write("Upload an image to generate its text.")

    # Upload Image
    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Image")
        

        # Generate Text
        text = generate_text(image, t_model, t_processor, device)
        
        # Display Text:
        if text:
            st.write(f"**Text generated:** {text}")
            
            file = save_text_as_file(text)

            st.download_button(label="Download Text File",
                                data=file,
                                file_name="generated_text.txt",
                                mime="text/plain"
                                )

if __name__ == "__main__":
    main()
