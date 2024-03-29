import streamlit as st

import io
import os
import yaml

from PIL import Image

from predict import load_model, get_prediction

from confirm_button_hack import cache_on_button_press

# SETTING PAGE CONFIG TO WIDE MODE
st.set_page_config(layout="wide")


root_password = '19881207'


def main():
    st.title("Mask Classification Model")

    with open("config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    

    model_list = load_model()
    model_list[0].eval()
    model_list[1].eval()
    model_list[2].eval()

    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg","png"])
    print('type(uploaded_file):', type(uploaded_file))

    if uploaded_file:
        image_bytes = uploaded_file.getvalue()
        print('type(image_bytes):', type(image_bytes))
        image = Image.open(io.BytesIO(image_bytes))
        print('type(image):', type(image))

        st.image(image, caption='Uploaded image')
        _, y_hat = get_prediction(model_list, image)
        label = config['classes'][y_hat.item()]
        st.write(f'result: {label}')


@cache_on_button_press('Authenticate')
def authenticate(password) ->bool:
    print(type(password))
    return password == root_password


password = st.text_input('password', type="password")

if authenticate(password):
    st.success('You are authenticated!')
    main()
else:
    st.error('The password is invalid.')