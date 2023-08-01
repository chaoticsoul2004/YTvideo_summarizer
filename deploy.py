import streamlit as st
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from youtube_transcript_api import YouTubeTranscriptApi


def video_id(link):
    video_id = link.split('=')[1]
    return video_id


def transcript(id):
    transcript_json = YouTubeTranscriptApi.get_transcript(id)
    result = ""
    for i in transcript_json:
        result += ' ' + i['text']

    return result


@st.cache_data()
def get_model():
    tokenizer = AutoTokenizer.from_pretrained('t5-small')
    model = AutoModelForSeq2SeqLM.from_pretrained(
        'Mayank1309/YTvideoSummarizer')
    return tokenizer, model


tokenizer, model = get_model()

user_input = st.text_area('enter the link for the video you want to summarize')
button = st.button("summarize")

if user_input and button:
    id = video_id(user_input)
    transcript = transcript(id)
    st.write(transcript)
    device = model.device

    tokenized = tokenizer(transcript, truncation=True,
                          padding='longest', return_tensors='pt')
    tokenized = {k: v.to(device) for k, v in tokenized.items()}
    tokenized_result = model.generate(**tokenized, max_length=250)
    tokenized_result = tokenized_result.to('cpu')
    predicted_summary = tokenizer.decode(tokenized_result[0])
    st.write(predicted_summary)
