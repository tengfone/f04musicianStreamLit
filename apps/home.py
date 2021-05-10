import streamlit as st
from machine_learning.CNN_main import CNN_pred
import os
import base64

def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
    return href

def app():
    st.title('Home')

    st.markdown("""
    # F04 Music Transcription 🎹
    A multi-model audio analysis tool, done periodically over a span of music wav files. Utilizing this model would produce a visual transcription that is able to show what specific notes are being played in the audio files.   
    
    To navigate, click the header on the left.
    """)
    st.markdown("""
        Please note that due to the limitations of Heroku Free Tier, the better accuracy prediction module (CNN-LSTM) is unable to run on this demo. If required, please run the old UI [here](https://github.com/tengfone/F04Musician) where the parameters of the LSTM can be tuned. This
        demo is using *CNN only*.
    """)

    st.title('Demo')

    st.write("""
       Record a .wav sound of a piano like this:
       """)

    selectedViewAudio = st.selectbox("View Demo Sound", ["Fur Elise", "Astronomia"])

    if (selectedViewAudio == "Fur Elise"):
        audio_file = open('./wav/test_furelise.wav', 'rb')
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format='audio / wav')
    elif (selectedViewAudio == 'Astronomia'):
        audio_file = open('./wav/test_coffin.wav', 'rb')
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format='audio / wav')

    st.write("""
               # Uploading Custom Wav File

               There are pre-uploaded demo files in `Running the Model`.
               """)

    zip_file = st.file_uploader("Upload .wav file", type=['wav'])
    if zip_file is not None:
        st.success("File Uploaded!")
        uploadedFileName = zip_file.name
        uploadedFileName = "./wav/" + uploadedFileName
        with open(uploadedFileName, 'wb') as out:  ## Open temporary file as bytes
            out.write(zip_file.read())


    st.write("""
           # Running the Model
           """)

    allFilesInAudio = []
    for (dirpath, dirnames, filenames) in os.walk('./wav'):
        for i in filenames:
            eachFile = dirpath + '/' + i
            allFilesInAudio.append(eachFile)
        break
    allFileNames = []

    for i in allFilesInAudio:
        singleName = i.split('/')[-1]
        allFileNames.append(singleName)

    selectedVideo = st.selectbox("Uploaded .wav files",
                                   allFileNames)

    selectedVideo = "./wav/" + selectedVideo

    # predOutput = 5
    # st.success(f"{predOutput}")
    if st.button("Transcribe"):
        if selectedVideo:
            with st.spinner("Loading... Please wait, it is slow, I know..."):
                if CNN_pred(selectedVideo):
                    st.write("""
                           # Transcribed Music: (.midi Output)
                           """)
                    st.markdown(get_binary_file_downloader_html('outputtest.midi', 'Midi File'), unsafe_allow_html=True)
                else:
                    st.warning("Error Please Contact @tengfone on GitHub")
        else:
            st.error("Select A File")
