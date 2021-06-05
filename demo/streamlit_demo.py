import streamlit as st
from predictionOnImage import return_prediction
from PIL import Image
from matplotlib import pyplot as plt 
import time
st.title("Distracted Driver Detection")

fig = plt.figure()

def main():
    file_uploaded = st.file_uploader("Choose File", type=["png","jpg","jpeg"])
    class_btn = st.button("Classify")
    if file_uploaded is not None:    
        image = Image.open(file_uploaded)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
    if class_btn:
        if file_uploaded is None:
            st.write("Invalid command, please upload an image")
        else:
            with st.spinner('Model working....'):
                plt.imshow(image)
                plt.axis("off")
                predictions = return_prediction(image)
                time.sleep(1)
                st.success('Classified')
                st.write(predictions)
                # st.pyplot(fig)


if __name__=='__main__':
    main()