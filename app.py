import streamlit as st
from fastai.vision.all import *
import plotly.express as px
import pathlib
from PIL import Image
import requests
from io import BytesIO

# PosixPath muvofiqligi
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# Streamlit sarlavhasi
st.title('Transportni klassifikatsiya qiluvchi model')

# Rasm yuklash
files = st.file_uploader("Rasm yuklash", type=["avif", "png", "jpeg", "gif", "svg"])
if files:
    # Rasmni ko‘rsatish
    st.image(files)
    img = PILImage.create(files.getvalue())

    try:
        # Modelni yuklash
        model = load_learner('transport_model.pkl')
    except Exception as e:
        st.error(f"Modelni yuklashda xatolik: {e}")
        model = None

    if model is not None:
        # Bashoratni topish
        try:
            pred, pred_id, probs = model.predict(img)
            st.success(f"Bashorat: {pred}")
            st.info(f"Ehtimollik: {probs[pred_id] * 100:.1f}%")

            # Diagrammani chizish
            fig = px.bar(x=probs * 100, y=model.dls.vocab, labels={'x': "Ehtimollik (%)", 'y': "Klasslar"}, orientation='h')
            st.plotly_chart(fig)
        except Exception as e:
            st.error(f"Bashoratda xatolik: {e}")

# Asl qiymatni tiklash
pathlib.PosixPath = temp


# import streamlit as st
# from fastai.vision.all import *
# import plotly.express as px
# import pathlib
# pathlib.PosixPath = pathlib.Path

# # title
# st.title('Transportni klassifikatsiya qiluvchi model')

# # Rasmni joylash
# files = st.file_uploader("Rasm yuklash", type=["avif", "png", "jpeg", "gif", "svg"])
# if files:
#     st.image(files)  # rasmni chiqarish
#     # PIL convert
#     img = PILImage.create(files)
    
#     # Modelni yuklash
#     model = load_learner('transport_model.pkl')

#     # Bashorat qiymatni topamiz
#     pred, pred_id, probs = model.predict(img)
#     st.success(f"Bashorat: {pred}")
#     st.info(f"Ehtimollik: {probs[pred_id] * 100:.1f}%")

#     # Plotting
#     fig = px.bar(x=probs * 100, y=model.dls.vocab)
#     st.plotly_chart(fig)