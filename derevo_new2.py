import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import pickle
import streamlit as st
import sys
import io


data = pd.read_csv('loan_new.csv')

def derevo_new2(): 
    st.title('Информационная система для прогнозирования решения по выдаче кредита') 
    global data
    buffer = io.StringIO()
    # Сохраняем текущий поток stdout
    old_stdout = sys.stdout
# Перенаправляем stdout в наш буфер
    sys.stdout = buffer
# Возвращаем stdout в его обычное состояние
    sys.stdout = old_stdout
# Получаем вывод из буфера
    info_text = buffer.getvalue()

# Загрузка данных
    data = pd.read_csv('loan_new.csv')
# Выбор числовых и категориальных переменных
    numerical_features = ['ApplicantIncome', 'LoanAmount', 'Dependents', 'Property_Area', 'AlimonyObligations', 'CreditObligations', 'HasAssets','Credit_History']
# Стандартизация числовых переменных
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(data[numerical_features])
# Объединение числовых и категориальных переменных
    X = np.hstack([scaled_features])
    y = data['Loan_Status'].values
# Разделение данных на обучающий и тестовый наборы
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=102)

# Оптимизация гиперпараметров
    parameters = {
  'n_estimators': [50, 100, 200, 500],  'max_depth': [3, 5, 7, 10],
  'min_samples_split': [2, 4, 8, 16],  'min_samples_leaf': [1, 2, 4, 8]
}
# RandomizedSearchCV
    random_search = RandomizedSearchCV(RandomForestClassifier(), parameters, cv=StratifiedKFold(n_splits=5), scoring='accuracy', n_iter=100)
    random_search.fit(X_train, y_train)
    best_params = random_search.best_params_
# Обучение модели с оптимальными гиперпараметрами
    rfmodel = RandomForestClassifier(**best_params)
    rfmodel.fit(X_train, y_train)
# Оценка производительности модели
    pred_l = rfmodel.predict(X_test)
    acc_l = accuracy_score(y_test, pred_l)*100

# Сравнение реальных и прогнозируемых значений
    comparison_df = pd.DataFrame({'Actual': y_test, 'Predicted': pred_l})
 

   # сохраняем обученную модель в файл 'best_model_gb.pkl'
    with open('best_model2_gb.pkl', 'wb') as file:
        pickle.dump(rfmodel, file)



    def predict(Credit_History, ApplicantIncome, LoanAmount, Dependents, Property_Area, AlimonyObligations, CreditObligations, HasAssets):
        with open('best_model2_gb.pkl', 'rb') as file:
            best_model = pickle.load(file)
        return best_model.predict([[Credit_History, ApplicantIncome, LoanAmount, Dependents, Property_Area, AlimonyObligations, CreditObligations, HasAssets]])

        st.title('Предсказание состояния займа')



    credit_history_options = {"Проблемы с кредитной историей": 0, "Хорошая кредитная история": 1}
    Credit_History = st.selectbox('История кредитов', options=list(credit_history_options.keys()), key='ch_key')
    Credit_History_encoded = credit_history_options[Credit_History]

    area_options = {"Городская зона": 0, "Сельская зона": 1, "Пригородная зона": 1 } 
    Property_Area = st.selectbox('Тип местности недвижимости', options=list(area_options.keys()), key='pa_key')
    Property_Area_encoded = area_options[Property_Area]

    alimony_options = {"Не имеются алиментные обязательства": 0, "Имеются алиментные обязательства": 1} 
    AlimonyObligations = st.selectbox('Алиментные обязательства', options=list(alimony_options.keys()), key='ao_key')
    AlimonyObligations_encoded = alimony_options[AlimonyObligations]

    hasAssets_options = {"Не имеется собственность": 0, "Имеется собственность": 1} 
    HasAssets = st.selectbox('Имущество в собственности', options=list(hasAssets_options.keys()), key='ha_key')
    HasAssets_encoded = hasAssets_options[HasAssets]

    credit_options = {"Не имеются": 0, "Имеются": 1} 
    CreditObligations = st.selectbox('Другие кредитные обязательства', options=list(credit_options.keys()), key='co_key')
    CreditObligations_encoded = credit_options[CreditObligations]

# Ввод данных
    Credit_History_encoded = credit_history_options[Credit_History]
    ApplicantIncome = st.slider('Доход заявителя', min_value=0, max_value=300000, value=50000, key='ai_slider')
    Loan_Amount = st.slider('Сумма кредита', min_value=9000, max_value=700000, value=1000, key='la_slider')
    Dependents = st.slider('Иждивенцы', min_value=0, max_value=3, value=1, key='d_slider')
    Property_Area_encoded = area_options[Property_Area]





    def predict_probability(Credit_History_encoded, ApplicantIncome, Loan_Amount, Dependents, Property_Area_encoded, AlimonyObligations_encoded, HasAssets_encoded, CreditObligations_encoded):
        data_point = np.array([(Credit_History_encoded, ApplicantIncome, Loan_Amount, Dependents, Property_Area_encoded, AlimonyObligations_encoded, HasAssets_encoded, CreditObligations_encoded)]).reshape(1, -1)
    # Предсказание вероятности
        probability = rfmodel.predict_proba(data_point)[0][1]
    # Отображение результата
        st.write(f'Вероятность одобрения кредита: {probability:.2f}')
    # Интерпретация результата
        if probability > 0.8:
            st.success('Высокая вероятность одобрения!')
        else:
            st.warning('Низкая вероятность одобрения.')

    if st.button('Посмотреть шансы получения кредита'):
            predict_probability(Credit_History_encoded, ApplicantIncome, Loan_Amount, Dependents, Property_Area_encoded, AlimonyObligations_encoded, HasAssets_encoded, CreditObligations_encoded)

    # Предсказание статуса
    if st.button('Получить ответ'):
        data_point = np.array([(Credit_History_encoded, ApplicantIncome, Loan_Amount, Dependents, Property_Area_encoded, AlimonyObligations_encoded, HasAssets_encoded, CreditObligations_encoded)]).reshape(1, -1)
        prediction = rfmodel.predict(data_point)

        if prediction[0] == 1:
            st.success('Заявка на кредит одобрена!')
        else:
            st.warning('Заявка на кредит отклонена.')

