import os
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
from matplotlib.ticker import NullLocator
import matplotlib.pyplot as plt

import torch
from torch.autograd import Variable
import torchvision.transforms as transforms

from utils import util
from utils import datasets
import models

import detect

# Default Model parameters
model_def = './config/yolov3-solar.cfg'
class_path = './sample_data/classes.names'
conf_thres = 0.8
nms_thres = 0.2
batch_size = 1
n_cpu = 0
img_size = 416
path_output = 'output/'

# Model Files
WEB_PATH_MODEL = 'https://githubraw.com/ManishSahu53/solarHotspotAnalysis/releases/download/1.1/yolov3_ckpt_49.pth'
PATH_MODEL = './weights/yolov3_ckpt_49.pth'
PATH_README = 'https://githubraw.com/ManishSahu53/solarHotspotAnalysis/1.1/README.md'
PATH_DEMO_IMAGE = './sample_data/test/RJPG/'

# Constants for sidebar dropdown
SIDEBAR_OPTION_PROJECT_INFO = 'Project Info'
SIDEBAR_OPTION_DEMO_IMAGE = "Proceed a Demo Image"
SIDEBAR_OPTION_UPLOAD_IMAGE = "Upload an Image"
LOADING_TEXT = 'Please wait for model to detect hotspots. This can take few minutes. Hotspots can be from Solar or any other Image'

SIDEBAR_OPTIONS = [SIDEBAR_OPTION_DEMO_IMAGE,
                   SIDEBAR_OPTION_UPLOAD_IMAGE, SIDEBAR_OPTION_PROJECT_INFO]

st.set_page_config(page_title="Solar HotSpot Analysis Tool", page_icon=":beer:",
                   layout="wide", initial_sidebar_state="expanded",)
                
st.sidebar.warning('Upload an JPG or RJPG Image. For best results.')

util.check_dir('./weights')
f_checkpoint = Path(PATH_MODEL)
if not f_checkpoint.exists():
    with st.spinner("Downloading model weights... this may take up to a few minutes. (~250 MB) Please don't interrupt it."):
        util.download_file(url=WEB_PATH_MODEL, local_filename=PATH_MODEL)

st.sidebar.title("Automatic Hotspot Analysis Tool")
left_column, right_column = st.beta_columns(2)

app_mode = st.sidebar.selectbox(
    "Please select from the following", SIDEBAR_OPTIONS)


@st.cache
def streamlit_load_model(model_def, img_size, PATH_MODEL, device):
    model = detect.load_model(model_def, img_size, PATH_MODEL, device=device)
    return model

# @st.cache(allow_output_mutation=True)


def process(path):
    img = np.array(Image.open(path).convert('RGB'))
    img_tensor = transforms.ToTensor()(img)

    img_tensor, _ = datasets.pad_to_square(img_tensor, 0)
    img_tensor = datasets.resize(img_tensor, img_size)

    img_tensor = Variable(img_tensor.type(Tensor))
    img_tensor = torch.unsqueeze(img_tensor, 0)

    # Model Prediction
    with torch.no_grad():
        detections = model(img_tensor)
        detections = util.non_max_suppression(
            detections, conf_thres, nms_thres)

    # Creating two Columns in Display. Left side User image, right side is Processed
    fig2 = plt.figure()
    plt.axis('off')
    plt.title('Selected Image')
    plt.imshow(img)
    left_column.pyplot(fig2, caption="Selected Input")
    ploting, fig, mapping = detect.post_process_prediction(mapping={temp_path_image: []}, img=img, img_size=img_size, path=temp_path_image,
                                                           detections=detections[0], classes=classes,
                                                           is_temperature=radiometric)
    # Save generated image with detections
    ploting.axis("off")
    ploting.title('Processed Image')
    fig.gca().xaxis.set_major_locator(NullLocator())
    fig.gca().yaxis.set_major_locator(NullLocator())

    right_column.pyplot(fig, caption="Processed Input")

    # Printing JSON
    st.json(mapping)
    # st.balloons()


# Loading Model
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
device = "cpu"

model = detect.load_model(model_def, img_size, PATH_MODEL, device=device)
classes = detect.load_classes(class_path)  # Extracts class labels from file

if app_mode == SIDEBAR_OPTION_PROJECT_INFO:
    st.sidebar.write(" ------ ")
    st.sidebar.success("Project information showing on the right!")
    st.write(util.get_file_content_as_string(PATH_README))

elif app_mode == SIDEBAR_OPTION_DEMO_IMAGE:
    st.sidebar.write(" ------ ")
    st.write('Select DEMO image or upload your image to process from LEFT side panel')

    path_photos = util.list_list(PATH_DEMO_IMAGE, 'JPG')
    option = st.sidebar.selectbox('Please select a sample image, then click Get HotSpots button', [
                                  os.path.basename(i) for i in path_photos])
    radiometric = st.sidebar.checkbox('Image is RJPG or Radiometric JPG?', )
    pressed = st.sidebar.button('Get Hotspots')

    if pressed:
        with st.spinner(LOADING_TEXT):

            st.empty()
            temp_path_image = os.path.join(PATH_DEMO_IMAGE, option)
            # print(f'temp_path_image: {temp_path_image}')
            process(temp_path_image)


elif app_mode == SIDEBAR_OPTION_UPLOAD_IMAGE:
    #upload = st.empty()
    # with upload:
    st.sidebar.info('PRIVACY POLICY: uploaded images are never saved or stored. They are held entirely within memory for prediction \
        and discarded after the final results are displayed. ')
    f = st.sidebar.file_uploader(
        "Please Select to Upload an Image", type=['png', 'jpg', 'jpeg'])
    if f is not None:
        tfile = tempfile.NamedTemporaryFile(delete=True)
        tfile.write(f.read())

        # print(f'tfile: {tfile.name}')

        radiometric = st.sidebar.checkbox(
            'Image is RJPG or Radiometric JPG?', )
        pressed = st.sidebar.button('Get Hotspots')

        if pressed:
            with st.spinner(LOADING_TEXT):
                st.empty()
                temp_path_image = tfile.name
                # print(f'temp_path_image: {temp_path_image}')
                process(temp_path_image)

else:
    st.sidebar.write('Please select valid option')


st.sidebar.write(" ------ ")
st.sidebar.write("**:beer: Buy me a [beer]**")
expander = st.sidebar.beta_expander("This app is developed by Manish Sahu.")
expander.write(
    "Contact me on [Linkedin](https://www.linkedin.com/in/manishsahuiitbhu/)")
expander.write(
    "The source code is on [GitHub](https://github.com/ManishSahu53/solarHotspotAnalysis)")
