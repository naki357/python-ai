# I want to create a simple agentic system that will connect to this Google Drive folder: https://drive.google.com/drive/u/0/folders/1FTL2iDts8iD2PsRczCwUMTQ0n0NY8GDm
# Analyze the files in the folder and summarize their contents.  It should also rename all .mp3 files to accurately refelect their content.

from pydub import AudioSegment
import speech_recognition as sr
import os   
from transformers import pipeline, Wav2Vec2ForCTC, Wav2Vec2Processor
import tempfile
import shutil
import re
import pandas as pd 
import io
import mimetypes
import librosa 
import numpy as np 
import torch 
import torchaudio
import matplotlib.pyplot as plt
from google import genai

# region Google Drive file processing
# from pydrive.auth import GoogleAuth
# from pydrive.drive import GoogleDrive
# GoogleAuth.DEFAULT_SETTINGS['client_config_file'] = "googleauth/credentials2.json"
# drive = GoogleDrive(GoogleAuth())
# folder_id = '1zqwQZTskfus34lWXujYEDpU59H3yaH86' 
# file_list = drive.ListFile({'q': f"'{folder_id}' in parents and trashed=false"}).GetList()
# # Loop through all files in the list
# for file_obj in file_list:
#     file_title = file_obj['title']
#     mime_type = file_obj['mimeType']
#     print(f"Processing file: {file_title}, MimeType: {mime_type}")

#     # Process files based on their type
#     if 'audio' in mime_type:
#         # Code for analyzing audio files goes here
#         try:
#             print(f"Downloading audio file: {file_title}")
#             # Save audio content to a temporary file
#             with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio_file:
#                 file_obj.GetContentFile(temp_audio_file.name)
#                 temp_filename = temp_audio_file.name
            
#             # Load and analyze audio with pydub
#             audio = AudioSegment.from_file(temp_filename)
#             print("Audio analysis:")
#             print(f"  Duration: {audio.duration_seconds} seconds")
#             print(f"  Channels: {audio.channels}")
#             print(f"  Frame Rate: {audio.frame_rate} Hz")
#             print(f"  Average loudness: {audio.dBFS} dBFS")
#             print("-" * 20)

#             # Clean up
#             os.remove(temp_filename)
#         except Exception as e: 
#             print(f"Error processing audio file: {file_title}: {e}")

#     elif 'spreadsheet' in mime_type or file_title.lower().endswith(('.xlsx', '.csv')):
#         try:
#             # Code for analyzing spreadsheet files goes here
#             print(f"Downloading spreadsheet: {file_title}")
#             # Download the file content
#             content = file_obj.GetContentString()
#             data = io.StringIO(content)
            
#             # Read the file into a pandas DataFrame based on file type
#             if file_title.lower().endswith('.csv'):
#                 df = pd.read_csv(data)
#             else: # Assumes Excel for other spreadsheet types
#                 df = pd.read_excel(data)

#             # --- Your Spreadsheet Analysis Code Here ---
#             print("Spreadsheet analysis:")
#             print(df.head()) # Print the first few rows
#             print(df.describe()) # Print summary statistics
#             print("-" * 20)
#         except Exception as e:
#             print(f"Error processing spreadsheet {file_title}: {e}")
# endregion

# region Local file processing 
def analyze_folder_content(folder_path, use_google_genai):
    """
    Inspects a local folder, categorizes files, and performs basic analysis.
    For audio files, it extracts key features like duration and dominant frequency.
    """
    if not os.path.isdir(folder_path):
        print(f"Error: '{folder_path}' is not a valid directory.")
        return {}

    file_analysis_results = {}

    for root, _, files in os.walk(folder_path):
        for file_name in files:
            my_file_path = os.path.join(root, file_name)
            mime_type, _ = mimetypes.guess_type(my_file_path)
            file_type = mime_type.split('/')[0] if mime_type else "unknown"
            file_extension = os.path.splitext(file_name)[1].lower()

            file_info = {
                "path": my_file_path,
                "size_bytes": os.path.getsize(my_file_path),
                "type": file_type,
                "extension": file_extension,
            }

            # region Audio file processing
            if file_type == "audio":
                try:
                    y, sr = librosa.load(my_file_path)
                    duration = librosa.get_duration(y=y, sr=sr)
                    # Extract dominant frequency using spectral centroid
                    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
                    dominant_frequency = np.mean(spectral_centroid)
                    file_info.update({
                        "audio_duration_seconds": duration,
                        "audio_dominant_frequency_hz": dominant_frequency,
                    })

                    waveform, sample_rate = torchaudio.load(my_file_path)
                    print(f"Analyzing: {file_name}" )
                    # Plot the waveform with the help of MatPlot Lib.
                    plot_waveform(waveform=waveform, sample_rate=sample_rate, filename=file_name)
                    
                    # Example: Calculate the root mean square (RMS) energy
                    rms = torch.sqrt(torch.mean(waveform**2))
                    # print(f"  RMS Energy: {rms.item():.4f}")
                    file_info.update({
                        "sample_rate": str(sample_rate) + " Hz", 
                        "num_channels": waveform.shape[0], 
                        "num_samples": waveform.shape[1], 
                        # "duration(s)": waveform.shape[1]/sample_rate:.2f, 
                        # "rms_energy": rms.item():.4f, 
                    })

                    # Example: Calculate the Mel-frequency cepstral coefficients (MFCCs)
                    # Ensure the audio is mono for MFCC calculation if it's stereo
                    if waveform.shape[0] > 1:
                        waveform_mono = torch.mean(waveform, dim=0, keepdim=True)
                    else:
                        waveform_mono = waveform

                    # Define MFCC transform
                    mfcc_transform = torchaudio.transforms.MFCC(
                        sample_rate=sample_rate,
                        n_mfcc=13, # Number of MFCCs to compute
                    )
                    mfccs = mfcc_transform(waveform_mono)
                    file_info.update({
                        "mfccs_shape": mfccs.shape
                    })

                    # We decide here if we'll use Google Gemini or local torch/torchaudio processing 
                    if use_google_genai:
                        client = genai.Client()
                        client.max_output_tokens = 100

                        myfile = client.files.upload(file=my_file_path)
                        prompt = 'Generate a transcript of the speech.'

                        response = client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=[prompt, myfile]
                        )
                        file_info.update({
                            "transcription": response.text
                        })
                    else:
                        # Now process speech recognition using Wav2Vec2ForCTC 
                        # Load the pre-trained model
                        model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
                        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

                        # Load audio data
                        new_sample_freq = 16000
                        waveform, sample_rate = torchaudio.load(my_file_path)
                        waveform_resampled = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=new_sample_freq)(waveform)                 

                        # Perform inference
                        with torch.no_grad():
                            logits = model(waveform_resampled).logits

                        # Decode logits to text
                        predicted_ids = torch.argmax(logits, dim=-1)
                        transcription = processor.batch_decode(predicted_ids)

                        file_info.update({
                            "transcription": transcription
                        })

                except Exception as e:
                    file_info["audio_analysis_error"] = str(e)
            #end region 
            elif file_type != "unknown":
                if use_google_genai:
                    client = genai.Client()
                    client.max_output_tokens = 100

                    myfile = client.files.upload(file=my_file_path)
                    prompt = 'Describe what this file is in general. Also tell me which song entries and artists appear the most.'

                    response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=[prompt, myfile]
                    )
                    file_info.update({
                        "longdescription": response.text
                    })
                else:
                    try:
                        with open(my_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read(500) # Read first 500 characters for preview
                            file_info["text_preview"] = content
                            file_info["line_count"] = len(f.readlines()) # Reopen to count lines
                    except Exception as e:
                        file_info["text_analysis_error"] = str(e)

            file_analysis_results[my_file_path] = file_info

    return file_analysis_results

def plot_waveform(waveform, sample_rate, filename):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
    figure.suptitle(filename)
    plt.show(block=False)

playlist_folder_path = "./data/playlists/"
audio_folder_path = "./data/audio/"
use_genai = True
audio_analysis = analyze_folder_content(playlist_folder_path, use_genai)
print("Completed processing.")
# print(analyze_folder_content(playlist_folder_path))
# endregion 