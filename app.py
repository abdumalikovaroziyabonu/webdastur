#uzgarish 2
import streamlit as st
from fastai.vision.all import *
import plotly.express as px
import pathlib
from PIL import Image
import requests
from io import BytesIO

# PosixPath moslashuvi
pathlib.PosixPath = pathlib.Path

# Sarlavha
st.markdown("#:Green[Tasvirlarni aniqlash]")
st.write("Klasslar: Car, Airplane, Boat, Carnivore, Musical_instrument, Sports_equipment, Telephone, Office_supplies, Kitchen_utensil")

# Rasmni yuklash - fayl yoki link orqali
st.markdown("> :green[Rasmni ushbu qismga yuklang]")
file_upload = st.file_uploader("Rasm yuklash (avif, png, jpeg, gif, svg)", type=["avif", "png", "jpeg", "gif", "svg","jfif"])
url_input = st.text_input("Yoki rasmning URL manzilini kiriting")

# Modelni yuklash
try:
    model = load_learner('transport_model.pkl')
except Exception as e:
    st.error(f"Modelni yuklashda xatolik: {e}")
    model = None

# Rasm yuklash va ko'rsatish
if file_upload or url_input:
    try:
        if file_upload:
            img = PILImage.create(file_upload)
            st.image(file_upload, caption="Yuklangan rasm")
        else:
            response = requests.get(url_input)
            img = PILImage.create(BytesIO(response.content))
            st.image(img, caption="URL orqali yuklangan rasm")
        
        if isinstance(img, PILImage) and model is not None:
            pred, pred_id, probs = model.predict(img)
            st.success(f"Bashorat: {pred}")
            st.info(f"Ehtimollik: {probs[pred_id] * 100:.1f}%")
            
            # Diagramma chizish
            fig = px.bar(x=probs * 100, y=model.dls.vocab, labels={'x': "Ehtimollik (%)", 'y': "Klasslar"}, orientation='h')
            st.plotly_chart(fig)
        else:
            st.error("Tasvirni yoki modelni yuklashda muammo bor.")
    except Exception as e:
        st.error(f"Bashorat qilishda xatolik: {e}")


