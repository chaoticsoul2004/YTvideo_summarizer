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


def get_model_trans():
    tokenizer = AutoTokenizer.from_pretrained('t5-small')
    model = AutoModelForSeq2SeqLM.from_pretrained(
        'Mayank1309/my_model')
    return tokenizer, model


tokenizer, model = get_model()
tokenizer, translator = get_model_trans()


user_input = st.text_area('enter the link for the video you want to summarize')
button = st.button("summarize")


# button2 = st.button("press to translate to french")

def translate(document):
    device = model.device
#   document = document.replace("</s>", "").replace("</pad>", "")
#   special_tokens_dict = {'additional_special_tokens': ['</s>', '</pad>']}

#   tokenizer.add_special_tokens(special_tokens_dict)
    document = document.replace("</s>", "").replace("<pad>", "")
    tokenized = tokenizer([document], truncation=True,
                          padding='longest', return_tensors='pt')
    tokenized = {k: v.to(device) for k, v in tokenized.items()}
    tokenized_result = translator.generate(**tokenized, max_length=128)
    tokenized_result = tokenized_result.to('cpu')
    translated = tokenizer.decode(tokenized_result[0])
    return translated


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
    st.write("\n")
    st.write("french: ")
    st.write(translate(predicted_summary))
