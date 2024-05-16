import streamlit as st
import pandas as pd
import time
import os
from transformers import pipeline

# Informacja o uruchomieniu aplikacji
st.success('Gratulacje! Z powodzeniem uruchomiłeś aplikację')

# Tytuł aplikacji
st.title('Aplikacja s20527: Wydźwięk tekstu i tłumaczenie tekstu')

# Podtytuły
st.header('Wprowadzenie do zajęć')
st.subheader('Streamlit')
st.text('Przykładowa aplikacja z wykorzystaniem Streamlit')
st.write('Streamlit jest biblioteką pozwalającą na uruchomienie modeli uczenia maszynowego.')

# Instrukcja użytkowania
st.subheader('Instrukcja użytkowania')
st.write('Ta aplikacja pozwala na analizę wydźwięku tekstu w języku angielskim oraz tłumaczenie tekstu z języka angielskiego na język niemiecki.')
st.write('Wybierz odpowiednią opcję z listy poniżej, a następnie wprowadź tekst do analizowania lub tłumaczenia.')

# Wczytanie pliku CSV i wyświetlenie danych
df = pd.read_csv("DSP_4.csv", sep=';')
st.dataframe(df)

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
            st.write(answer[0]['translation_text'])

# Zadanie do wykonania
st.subheader('Zadanie do wykonania')
st.write('Wykorzystaj Huggin Face do stworzenia swojej własnej aplikacji tłumaczącej tekst z języka angielskiego na język niemiecki. Zmodyfikuj powyższy kod dodając do niego kolejną opcję, tj. tłumaczenie tekstu. Informacje potrzebne do zmodyfikowania kodu znajdziesz na stronie Huggin Face - https://huggingface.co/docs/transformers/index')
st.write('🐞 Dodaj właściwy tytuł do swojej aplikacji, może jakieś grafiki?')
st.write('🐞 Dodaj krótką instrukcję i napisz do czego służy aplikacja')
st.write('🐞 Wpłyń na user experience, dodaj informacje o ładowaniu, sukcesie, błędzie, itd.')
st.write('🐞 Na końcu umieść swój numer indeksu')
st.write('🐞 Stwórz nowe repozytorium na GitHub, dodaj do niego swoją aplikację, plik z wymaganiami (requirements.txt)')
st.write('🐞 Udostępnij stworzoną przez siebie aplikację (https://share.streamlit.io) a link prześlij do prowadzącego')

# Dodanie numeru indeksu
st.write('Numer indeksu: s20527')