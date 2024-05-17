import streamlit as st
import pandas as pd
import time
from transformers import pipeline

st.success('Gratulacje! Z powodzeniem uruchomiłeś aplikację')

st.title('Aplikacja s20527: Wydźwięk tekstu i tłumaczenie tekstu')
st.header('Wprowadzenie do zajęć')

st.subheader('Streamlit')
st.write('Przykładowa aplikacja z wykorzystaniem Streamlit')
st.write('Streamlit jest biblioteką pozwalającą na uruchomienie modeli uczenia maszynowego.')

st.subheader('Informacje o aplikacji')
st.write('Ta aplikacja została stworzona w celu demonstracji możliwości biblioteki Streamlit oraz modeli z Hugging Face.')
st.write('Aplikacja umożliwia analizę wydźwięku emocjonalnego tekstu w języku angielskim oraz tłumaczenie tekstu z języka angielskiego na język niemiecki.')

st.image('slownik.jpg', caption='Słownik', use_column_width=True)

st.subheader('Instrukcja użytkowania')
st.write('Ta aplikacja pozwala na analizę wydźwięku tekstu w języku angielskim oraz tłumaczenie tekstu z języka angielskiego na język niemiecki.')
st.write('Wybierz odpowiednią opcję z listy poniżej, a następnie wprowadź tekst do analizowania lub tłumaczenia.')

# Wybór opcji
st.header('Przetwarzanie języka naturalnego')
option = st.selectbox(
    "Opcje",
    [
        "Wydźwięk emocjonalny tekstu (eng)",
        "Tłumaczenie tekstu (eng -> de)"
    ],
)

# Wydźwięk emocjonalny tekstu
if option == "Wydźwięk emocjonalny tekstu (eng)":
    text = st.text_area(label="Wpisz tekst")
    if text:
        with st.spinner('Analizuję wydźwięk emocjonalny...'):
            classifier = pipeline("sentiment-analysis")
            answer = classifier(text)
            st.success('Analiza zakończona!')
            st.write(answer)

# Tłumaczenie tekstu
if option == "Tłumaczenie tekstu (eng -> de)":
    text = st.text_area(label="Wpisz tekst")
    if text:
        with st.spinner('Tłumaczę tekst...'):
            translator = pipeline("translation_en_to_de")
            answer = translator(text, max_length=512)
            st.success('Tłumaczenie zakończone!')
            st.balloons()  # Efekt balonów po zakończeniu tłumaczenia
            st.write(answer[0]['translation_text'])

st.subheader('Załadowanie przykładowego zbioru danych')
df = pd.read_csv("DSP_4.csv", sep=';')
st.dataframe(df)

# Dodanie numeru indeksu
st.write('Numer indeksu: s20527')