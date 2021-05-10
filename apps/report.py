import streamlit as st


def app():
    st.title('Report & Model Info')
    st.markdown("""
## Abstract  
A multi-model audio analysis tool, done periodically over a span of music wav files. Utilizing this model would produce a visual transcription that is able to show what specific notes are being played in the audio file.  
## Introduction
Automatic Music Transcription (AMT) is a process of converting an acoustic music into any form of musical notation, which allows the preservation of music
performances and the composers’ pieces over a long period of time. However, due to the complexity and the diversity of music nature and musical features,  
this process has been a challenging task, even for musicians with many years of formal training [1, 6, 10, 13]. Furthermore, in human music transcription, there  
exists some problems, such as, each musician may transcribe the same musical piece differently due to differences in musical perceptions, differences in viewing  
the importance of different aspects of a musical piece, and differences in transcription speed when given different musical genres to transcribe [10]. These  
introduce inconsistency in the human transcription process and other methods are required to retrieve the musical information and transcribe important  
musical features, such as pitch, tempo, loudness, and duration [6].  
In recent years, machine learning models have proved to bring many benefits  
in music research, from music generation [8] to music genre classification [11].  
Hence, many research has been conducted to apply machine learning models to  
the music transcription process [6, 16, 15], and it was identified that the two  
crucial stages in AMT is pitch detection and note tracking [14]. Pitch detection  
is to estimate the pitch that is played at each time frame, and note tracking is  
to be able to identify the onset and offset time of a pitch. Convolutional Neural Network (CNN) is one of the most popular models that is used in AMT, and  
is widely used in the two stages [1, 16]. This paper explores the use of CNN  
model in monophonic music transcription, focusing mainly on the note tracking  
stage, and compare it with a state-of-the-art [5] model - CREPE (Convolutional  
Representation for Pitch Estimation), which targets the pitch detection stage  
[9].  
  
 ## Motivation It was noticed that in the music industry, talent and perfect pitch are what themusician is born with. Although it might be possible to train the ear to detect  
certain notes and melodies, it is mostly people with innate talent that are able  
to transcribe intricate songs into notes or sheet music with ease. This project  
aims to remove this limitation using machine learning model to help people with  
a strong passion for music to be able to play any song they desire even when  
the sheet music is not available to them.  
  
 ## Project Overview This project utilizes the forms of machine learning algorithms, ConvolutionalNeural Network (CNN) and Long Short-Term Memory (LSTM), in an attempt  
to resolve the problems in AMT.  
  
 ## Dataset and Collection MIDI Aligned Piano Sounds (MAPS) dataset is used for training and testingthe two models [4]. The dataset consists of 29910 piano recordings in .wav file  
format, categorised into nine folders based on the different instruments used,  
and their recording conditions. Within each folder, the piano sound files are  
further categorised based on the audio characteristics into four sets – ISOL,  
RAND, UCHO and MUS set. The description of the four categories are shown  
in Table 1
    """)
    col1, col2, col3 = st.beta_columns([1, 6, 1])
    with col2:
        st.image('./images/table1.PNG')
    st.markdown("""
The MAPS recordings are sampled with a sampling frequency of 44kHz, and
each recording is paired with a .txt file, which records the onset and offset time
for all the notes that are played with its corresponding midi number. The .txt file
will be used as a ground truth to evaluate the performance of the models. In our
initial model - CNN only model, only the ISOL file is used for training, whereas
the entire dataset is used to train our final model - CNN + LSTM model. The
dataset is preprocessed first before training, which will be explained in the later
section.

    ## Data Pre-processing
    
    ### Librosa
The initial model involves using librosa [12]. Using Fast Fourier Transform
(FFT), the audio signal input was converted from time domain to frequency
domain, where the magnitude is the contribution of each frequency to the overall
sound. Subsequently, the magnitude was mapped to relative frequency using
linspace (function that gives a number of evenly spaced numbers in an interval.
(freq interval to be considered is 0 to sr, no of evenly spaced values = len(mag)).
The power spectrum obtained is shown in Figure 1.
    """)
    col1, col2, col3 = st.beta_columns([1, 6, 1])
    with col2:
        st.image('./images/fig1.PNG')
    st.markdown("""
To understand how the frequency contributed to the overall sound throughout time, the short time Fourier transform (STFT) is used with the following
parameter definitions:
#FFT window size
n_fft = 2048
#number of frames of audio between STFT column
hop_length = 512
#STFT
stft = librosa.core.stft(signal, hop_length=hop_length, n_fft=n_fft)
The output of the STFT can be visualized through a spectrogram, a heatmap
with frequency on the y-axis, time on the x-axis and colour of the heatmap as
the amplitude of the frequency, as shown in Figure 2.
spectrogram = np.abs(stft)
    """)
    col1, col2, col3 = st.beta_columns([1, 6, 1])
    with col2:
        st.image('./images/fig2.PNG')
    st.markdown("""
To linearize the STFT, logarithm is applied to the spectrogram and the output
is shown in Figure 3.
    """)
    col1, col2, col3 = st.beta_columns([1, 6, 1])
    with col2:
        st.image('./images/fig3.PNG')
    st.markdown("""
Another method explored to classify the training .wav files to their respective
label notes is chroma feature. While the pitch can be obtained easily using
chroma feature, information about the octave is not captured. Figure 4 shows
some sample outputs from librosa’s chroma feature.
With reference to Figure 5, M65 has a predicted note of F, and M63 has a
predicted note of D, which are both correct. However, the tool is less accurate
for notes towards the ends of the spectrum. Hence, a model is required to obtain
higher accuracies.
    """)
    col1, col2, col3 = st.beta_columns([1, 6, 1])
    with col2:
        st.image('./images/fig4.PNG')
    col1, col2, col3 = st.beta_columns([1, 6, 1])
    with col2:
        st.image('./images/fig5.PNG')
    st.markdown("""
        ### nnAudio
        
To achieve better accuracies, preprocessing and normalisation on the spectrogram were done to provide a better input representation for the CNN model [2].
The Mel spectrogram was chosen as the preprocessing step because it is proven
to provide a high accuracy using a compact spectrogram representation, thus
reducing the computational speed [2].
nnAudio [3], an audio processing toolbox which provides on-the-fly audio to
spectrogram conversion using the GPU, and consists of the Mel spectrogram
function, is used in our model as it is able to generate the spectrogram representations around 100 times faster than traditional libraries [2].
The setup for the Mel spectrogram is as follows:
self.spec_layer = Spectrogram.MelSpectrogram(
sr=44100,
n_mels=156,
hop_length=441,
pad_mode=’constant’,
fmin=20, fmax=4200,
trainable_mel=False,
trainable_STFT=False)
.to(device)
The sampling rate was kept at 44100 Hz in accordance to the MAPS dataset. A
hop length of 441 meant that a timestep of 10ms can easily be obtained. fmin
and fmax relates to the lowest and highest frequency of the keys of a piano
respectively.

## CNN Architecture

The preliminary model was first trained using a CNN architecture. The MelSpectrogram layer is first obtained from the library nnAudio. The output from
the Mel-Spectrogram layer is passed through a log function for the purpose of
making training more efficient. After which, 2D Convolution is applied across
both frequency and time domains with an accompanying ReLU Activation function. The output is subsequently passed through a Linear layer. The output of
this fully connected layer is finally passed through a sigmoid activation function
for classification.
        """)
    col1, col2, col3 = st.beta_columns([1, 6, 1])
    with col2:
        st.image('./images/fig6.PNG')
    st.markdown("""
        ## CNN-LSTM Architecture
        
Building on the CNN only architecture, the convolution across time domain
is replaced with LSTM. In training the model, a batch size of 1024 is passed
through the spectrogram as input through a 2D Convolution layer across frequency domain which does feature extraction and downsampling of the Mel
spectrogram. After reshaping the output of the CNN blocks of size 3008, it is
passed through a LSTM layer with hidden size of 1024, and this output is passed
through a dense layer with an output size of 88. The final output is obtained
by passing through a sigmoid activation function.
        """)
    col1, col2, col3 = st.beta_columns([1, 6, 1])
    with col2:
        st.image('./images/fig7.PNG')
    st.markdown("""
## Model Adjustments
        
To further improve the model accuracy, some ablation studies were conducted
for the model.

### Adjusting Batch Size and Window Size
        
Comparing Figure 8 and Figure 9, while larger batch sizes are known to have
greater generalization errors, the increase in window size has a greater effect on
the accuracy of the output. The model with a longer window size has longer
duration on the notes as well as more defined probability values, with red as the
highest probability.
However, the longer the window sizes, the slower the training speed as more data
will be trained on. With the understanding that larger batch sizes also have
faster computational speeds, the final batch size and window size used for training the final model are shown in Table 2 along with the other hyperparameters
of the final model.
        """)
    col1, col2, col3 = st.beta_columns([1, 6, 1])
    with col2:
        st.image('./images/fig8.PNG')
    col1, col2, col3 = st.beta_columns([1, 6, 1])
    with col2:
        st.image('./images/fig9.PNG')
    col1, col2, col3 = st.beta_columns([1, 6, 1])
    with col2:
        st.image('./images/tab2.PNG')
    st.markdown("""
## Output Processing
        
### Adjusting Window Size
        
By adjusting the window size of the test data, while the window size of the
training data is fixed, it is also observed that this affects the output from the
test data
        """)
    col1, col2, col3 = st.beta_columns([1, 6, 1])
    with col2:
        st.image('./images/fig10.PNG')
    st.markdown("""
As seen from Figure 10 above, the results are generated from the exact same
model, with the top row having a test window size of 10, bottom row having a
test window size of 5, and with Fur Elise (left) and lunatic princess (right) as
the inputs. It is observed that in the images of the top row, the notes are more
joined together. However, overall, there is also more loss of notes, especially the
shorter ones, in the top row.

## Combining Window Sizes
        
With the above observation, an attempt to combine the outputs of different
window sizes, where the maximum value at each timestep and Musical Instrument Digital Interface (MIDI) note across the windows sizes is taken, was made.
The rationale behind is to get the best of both worlds where larger window sizes
help to keep longer notes joined together and small window sizes help to detect
the shorter notes.
However, this resulted in some unwanted joining of notes and additional noise,
as seen from Figure 11.""")
    col1, col2, col3 = st.beta_columns([1, 6, 1])
    with col2:
        st.image('./images/fig11.PNG')
    st.markdown("""
Despite that, with the fact that a note that has a high probability at a certain
timestep across different windows means that the model has a high confidence
for that prediction, by increasing the activation, a final output that is better
than the raw output of the model can be obtained, where the incorrectly joined
notes are separated again, and the additional noise are removed, as seen from
Figure 12.
        """)
    col1, col2, col3 = st.beta_columns([1, 6, 1])
    with col2:
        st.image('./images/fig12.PNG')
    st.markdown("""
## Front-end User Interface
    
The model is pre-loaded using PyTorch .pt model. The web application can either be hosted locally or on an AWS EC2 XLarge 32GB Disk Storage Instance. The front-end code can be found at https://github.com/tengfone/F04Musician""")
    col1, col2, col3 = st.beta_columns([1, 6, 1])
    with col2:
        st.image('./images/fig13.PNG')
    st.markdown("""
## Evaluation
    
### CREPE Model for Evaluation
    
The two models are compared against the current state of the art in monophonic
pitch estimation - Convolutional Representation for Pitch Estimation (CREPE)
model to provide a rough estimation of the performance of the two models.
The CREPE pitch estimation model is able to generate predictions with timedomain audio signal as its inputs. For each prediction that is made on the
user-selected timestep, the algorithm also outputs the confidence value, which
measures the probability of detecting any audio signal during that time frame.
To make better comparison between our models and the CREPE model, the
timestep selected for the CREPE model was also set to be 10ms for each prediction. Different confidence intervals were tested on the training dataset, as
shown in Figure 12.1, and the confidence level of 0.85 was chosen after comparing it with audio sound with different characteristics, as shown in Table 3.
Simple post processing process was done on the CREPE prediction output, the
single stray notes were removed by setting its frequency to be the same as the
frequency of the previous time frame.""")
    col1, col2, col3 = st.beta_columns([1, 6, 1])
    with col2:
        st.image('./images/tab3.PNG')
    col1, col2, col3 = st.beta_columns([1, 6, 1])
    with col2:
        st.image('./images/fig14.PNG')
    st.markdown("""
### Evaluation Metric
        
As a huge portion of the MAPS recordings contains long waiting time between the start of the recording to the onset time of the first note appears, and between the offset time of the last note to the end of the recordings, weights were added to the accuracy score to highlight the correct prediction made on the non-zero midi notes (true positive, TP), and not neglecting the correct prediction made on the zero midi notes (true negative, TN). The weighted accuracy score was calculated using the formula below:
""")
    col1, col2, col3 = st.beta_columns([1, 6, 1])
    with col2:
        st.image('./images/form.PNG')
    st.markdown("""
α score of 1 and β score of 0.5 were chosen after multiple trials with different α and β combinations. The same testing dataset was used to get the average weighted accuracy score of the three model, which the scores are shown in Table 4.
    """)
    col1, col2, col3 = st.beta_columns([1, 6, 1])
    with col2:
        st.image('./images/tab4.PNG')
    col1, col2, col3 = st.beta_columns([1, 6, 1])
    with col2:
        st.image('./images/tab5.PNG')
    st.markdown("""
## Discussion
    
From Table 5, it is observed that the 2nd model - CNN + LSTM model has a
significant improvement in performance when it is compared against its previous
model - CNN model. It shows that by adding a LSTM layer, the training model
is able to perform better as audio itself can be seen as a sequential data where
learning front and back while retaining information from the previous layer helps
with identifying the correct pitch and furthermore, LSTM takes into account
on temporal dependence.
However, our two models are trained and tested with MAPS dataset, it introduces some biases when comparing the weighted accuracy score with the
CREPE model, which is not trained using the MAPS dataset. Hence, the average weighted accuracy score when comparing against the CREPE model can
only serve as a relative measurement of improvement between our 1st and 2nd
model.

## Future works and improvements
    
### Polyphonic music transcription
    
Currently the model only works on monophonic music with some promising entities for polyphonic music transcription. This can be achieved by training it on MAESTRO data set, a data set that consist of polyphonic music data [7].
    
### Transcription
    
The output for the prediction is currently a MIDI file and is presented as a piano roll. There can be an option to convert the MIDI file output into a proper piano sheet.
    
### Improve Accuracy
    
It is possible to do ensemble model by training more models (3D Convolutional Layer) which can be then averaged out to obtain the best possible results.
    
## References
    
[1] M. Bereket. An ai approach to automatic natural music transcription. 2017.

[2] K. W. Cheuk, K. Agres, and D. Herremans. The impact of audio input

representations on neural network based music transcription. In 2020 International Joint Conference on Neural Networks (IJCNN), pages 1–6, 2020.

[3] K. W. Cheuk, K. Agres, and D. Herremans. The impact of audio input
representations on neural network based music transcription. In 2020 International Joint Conference on Neural Networks (IJCNN), pages 1–6, 2020.

[4] V. Emiya, N. Bertin, B. David, and R. Badeau. Maps - a piano database
for multipitch estimation and automatic transcription of music. page 11,
07 2010.

[5] B. Gfeller, C. Frank, D. Roblek, M. Sharifi, M. Tagliasacchi, and M. Velimirovic. Spice: Self-supervised pitch estimation. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 28:1118–1128, 2020.

[6] C. Hawthorne, A. Stasyuk, A. Roberts, I. Simon, C.-Z. A. Huang, S. Dieleman, E. Elsen, J. Engel, and D. Eck. Enabling factorized piano music
modeling and generation with the MAESTRO dataset. In International
Conference on Learning Representations, 2019.

[7] C. Hawthorne, A. Stasyuk, A. Roberts, I. Simon, C.-Z. A. Huang, S. Dieleman, E. Elsen, J. Engel, and D. Eck. Enabling factorized piano music
modeling and generation with the MAESTRO dataset. In International
Conference on Learning Representations, 2019.

[8] N. Hewahi, S. AlSaigal, and S. AlJanahi. Generation of music pieces using
machine learning: long short-term memory neural networks approach. Arab
Journal of Basic and Applied Sciences, 26(1):397–413, 2019.

[9] J. W. Kim, J. Salamon, P. Li, and J. P. Bello. Crepe: A convolutional
representation for pitch estimation, 2018.

[10] A. Klapuri. Introduction to Music Transcription, pages 3–20. 01 2006.

[11] B. Lansdown. Machine Learning for Music Genre Classification. PhD
thesis, 09 2019.

[12] B. McFee, C. Raffel, D. Liang, D. Ellis, M. McVicar, E. Battenberg, and
O. Nieto. librosa: Audio and music signal analysis in python. 2015.

[13] C. Roads. Research in music and artificial intelligence. ACM Comput.
Surv., 17(2):163–190, June 1985.

[14] M. Rom´an, A. Pertusa, and J. Calvo-Zaragoza. An end-to-end framework
for audio-to-score music transcription on monophonic excerpts. 09 2018.

[15] V. Sarnatskyi, V. Ovcharenko, M. Tkachenko, S. Stirenko, Y. Gordienko,
and A. Rojbi. Music transcription by deep learning with data and ”artificial
semantic” augmentation. CoRR, abs/1712.03228, 2017.

[16] B. Sturm, J. Santos, O. Ben-Tal, and I. Korshunova. Music transcription
modelling and composition using deep learning. 04 2016.""")

