import torch
import streamlit as st
from utils import transform_image
import yaml
from model import Classifier2, AgeClassifier
from PIL import Image

@st.cache
def load_model() -> list:
    with open("config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_mask = Classifier2('vit_small_patch16_224', 3).to(device)
    model_gender = Classifier2('vit_small_patch16_224', 2).to(device)
    model_age = AgeClassifier('resnet50', 1).to(device)

    state_dict = torch.load(config['model_path'], map_location=device)
    model_mask.load_state_dict(state_dict['model_mask_state_dict'])
    model_gender.load_state_dict(state_dict['model_gender_state_dict'])
    model_age.load_state_dict(state_dict['model_age_state_dict'])
    
    return [model_mask, model_gender, model_age]


def get_prediction(model:list, image: Image) -> list:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = transform_image(image).to(device)
    mask_pred = model[0](image)
    gender_pred = model[1](image)
    age_pred = model[2](image)
    pred = torch.max(mask_pred, 1)[1] * 6 + torch.max(gender_pred, 1)[1] * 3 + torch.bucketize(age_pred, torch.Tensor([30,60]).cuda(), right=True).squeeze()
    return [image, pred]
