import streamlit as st
from clinic import clinic
from hospital import hospital
from pharmacy import pharmacy
from styles import overall_css

st.set_page_config(page_title="Rayan Hospital", page_icon="🤖")
# Apply the CSS styles
st.markdown(overall_css, unsafe_allow_html=True)

# Center the title
st.markdown("<h1 style='text-align: center;'>Rayan Hospital</h1>", unsafe_allow_html=True)

# Initialize session state for page navigation with "hospital" as the default
if "page" not in st.session_state:
    st.session_state.page = "hospital"

def navigate_to(page):
    st.session_state.page = page

# Create columns for navigation buttons
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Hospital"):
        navigate_to("hospital")

with col2:
    if st.button("Clinics"):
        navigate_to("clinic")

with col3:
    if st.button("Pharmacy"):
        navigate_to("pharmacy")

# Create 3 columns
col1, col2 = st.sidebar.columns(2)

with col1:
    st.image("cloudSolutions.jpg")

with col2:
    st.image("rayan.png")

# Display the appropriate page based on session state
if st.session_state.page == "hospital":
    hospital()
elif st.session_state.page == "clinic":
    clinic()
elif st.session_state.page == "pharmacy":
    pharmacy()
else:
    hospital()  # Default to hospital

def set_background_image(image_path):
    """
    Set a background image for the Streamlit app.
    :param image_path: str, path to the background image
    """
    # Read the image file
    with open(image_path, "rb") as image_file:
        image_bytes = image_file.read()
    
    # Encode the image in base64
    import base64
    encoded_image = base64.b64encode(image_bytes).decode()
    
    # Define the CSS to set the background image
    background_image_style = f"""
    <style>
    .stApp {{
        background: url(data:image/jpg;base64,{encoded_image});
        background-size: cover;
    }}
    </style>
    """
    
    # Add the CSS to the Streamlit app
    st.markdown(background_image_style, unsafe_allow_html=True)

# Path to the background image
image_path = "bgimage.png"

# Call the function to set the background image
set_background_image(image_path)