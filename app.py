import streamlit as st
from derevo_new2 import derevo_new2  # Прогнозирование дерево со стандартизацией и поиском лучших параметров

# Настройка темы
st.set_page_config(
    layout="wide",
    initial_sidebar_state="collapsed", 
    page_title="Информационная система для прогнозирования решения по выдаче кредита",
    page_icon="📈",
)

def app(): 
    # Отображение страницы
    derevo_new2()

if __name__ == "__main__":
    app()
