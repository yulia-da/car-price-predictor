import streamlit as st
import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, OneHotEncoder

st.markdown('## Предсказание стоимости автомобиля ##')
st.markdown('### Загрузите файл с обучающими данными или введите их вручную: ###' )
data = st.file_uploader("Загрузите ваш файл", type=["csv"])
if data is not None:
    df = pd.read_csv(data)
    if 'selling_price' in df.columns:
       df = df.drop('selling_price', axis=1)
    st.dataframe(df)
    st.session_state.data = df.copy()
else:
    features_ = {
    'name': 'categorical',
    'year': 'numeric',
    'km_driven': 'numeric',
    'fuel': 'categorical',
    'seller_type': 'categorical',
    'transmission': 'categorical',
    'owner': 'categorical',
    'mileage': 'categorical',
    'engine': 'categorical',
    'max_power': 'categorical',
    'torque': 'categorical',
    'seats': 'numeric'
}

    features = list(features_.keys())
    feature_types = features_

    if 'data' not in st.session_state:
        st.session_state.data = pd.DataFrame(columns=features)

    
    col1, col2, col3 = st.columns(3)

    current_sample = {}
    for idx, feature in enumerate(features):
        with [col1, col2, col3][idx % 3]:
            if feature_types[feature] == 'numeric':
                current_sample[feature] = st.number_input(feature, value=0.0, key=f"input_{feature}")
            else:
                current_sample[feature] = st.text_input(feature, key=f"input_{feature}")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Добавить к данным", type="primary", use_container_width=True):
            missing_fields = []
            for feature in features:
                if feature_types[feature] == 'categorical':
                    if not current_sample[feature] or current_sample[feature] == "":
                        missing_fields.append(feature)
            if missing_fields:
                st.warning(f"Пожалуйста, заполните поля: {', '.join(missing_fields)}")
            else:
                new_row = {}
                for feature in features:
                    new_row[feature] = current_sample[feature]
            
                new_df = pd.DataFrame([new_row])
                st.session_state.data = pd.concat([st.session_state.data, new_df],ignore_index=True)
                st.success(f"Запись добавлена! Всего записей: {len(st.session_state.data)}")
            
                st.rerun()
    with col2:
        if st.button("Готово", use_container_width=True):
            if len(st.session_state.data) > 0:
                st.success(f"Собрано {len(st.session_state.data)} записей")
            else:
                st.warning("Нет добавленных записей")

    with col3:
        if st.button("Очистить все данные", use_container_width=True):
            st.session_state.data = pd.DataFrame(columns=features)
            st.success("Все данные очищены!")
            st.rerun()

    st.subheader(f"Добавлено ({len(st.session_state.data)} записей)")

    st.dataframe(st.session_state.data)
df = st.session_state.data.copy()

st.write(f"Размер данных: {df.shape[0]} строк, {df.shape[1]} столбцов")
to_clean = ['mileage', 'engine', 'max_power']
for col in to_clean:
    df[col] = df[col].str.extract(r'(\d+\.?\d*)').astype(float)

def rpm(value):
  s = str(value).lower()
  match = re.search(r'(\d+(?:,\d+)?(?:-\d+(?:,\d+)?)?)\s*rpm', s)
  if match:
    return match.group(1).replace(',', '')
  match = re.search(r'@\s*(\d+(?:,\d+)?(?:-\d+(?:,\d+)?)?)', s)
  if match:
    return match.group(1).replace(',', '').replace('.', '')
  match = re.search(r'/\s*(\d+(?:\.\d+)?)$', s)
  if match:
    return match.group(1)
  if 'rpm' not in s:
    numbers = re.findall(r'\d+(?:[.,]\d+)?', s)
    if len(numbers) == 1:
      return np.nan
  return value

def to_nm(value):
    s = str(value).lower()
    nm_match = re.search(r'(\d+(?:\.\d+)?).*?nm', s)
    if nm_match:
        return nm_match.group(1)
    kgm_match = re.search(r'(\d+(?:\.\d+)?).*?kgm', s)
    if kgm_match:
        nm_value = float(kgm_match.group(1)) * 9.80665
        return f"{nm_value:.1f}"
    at_match = re.search(r'(\d+(?:\.\d+)?).*?@', s)
    if at_match:
        return at_match.group(1)
    match = re.search(r'(\d+(?:\.\d+)?)', s)
    return match.group(1) if match else value

df['max_torque_rpm'] = df['torque'].apply(rpm)
df['torque'] = df['torque'].apply(to_nm)

def avg_torque_rpm(value):
    s = str(value)
    if '-' in s:
      numbers = re.findall(r'(\d+(?:\.\d+)?)', s)
      if len(numbers) == 2:
        avg = (float(numbers[0]) + float(numbers[1])) / 2
        return avg
    return float(s)

df['max_torque_rpm'] = df['max_torque_rpm'].apply(avg_torque_rpm)
df['torque'] = df['torque'].astype(float)

numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

for col in numeric_cols:
    median_val = df[col].median()
    df[col] = df[col].fillna(median_val)

df['engine'] = df['engine'].astype(int)
df['seats'] = df['seats'].astype(int)

st.subheader("Числовые признаки")
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
selected_num = st.selectbox("Выберите признак", num_cols)
if selected_num:
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(df[selected_num], kde=True, color='blue', ax=ax)
    ax.set_title(f'Распределение {selected_num}')
    ax.set_xlabel(col)
    st.pyplot(fig)

cat_cols = df.select_dtypes(include=['object']).columns.tolist()
if cat_cols:
    st.subheader("Категориальные признаки")
    selected_cat = st.selectbox("Выберите признак", cat_cols)
    if selected_cat:
        fig, ax = plt.subplots(figsize=(10, 5))
        cnt = df[selected_cat].value_counts().head(10).reset_index()
        cnt.columns = [selected_cat, 'count']
        sns.barplot(data=cnt, x=selected_cat, y='count', ax=ax, color='purple')
        ax.set_title(f'Распределение {selected_cat}')
        ax.set_xlabel(selected_cat)
        ax.set_ylabel('Количество')
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig)

st.write("Тепловая карта корреляции:")
df.corr(numeric_only=True)
fig , ax = plt.subplots(figsize = (10,10))

sns.heatmap(df.corr(numeric_only=True),
            linewidths=0.5, annot=True,cmap='viridis',
            linecolor="white", annot_kws = {'size':12})
st.pyplot(fig)

df['power_engine'] = df['max_power']/df['engine']
df['km_per_year'] = df['km_driven'] / (2026 - df['year'])
df['is_old_car'] = (df['year'] < 2016).astype(int)
df['high_km'] = (df['km_driven'] > 100000).astype(int)

def load_model():
    with open('model-4.pickle', 'rb') as file:
        data = pickle.load(file)
    
    return data['model'], data['encoder'], data['scaler'], data.get('feature_names', None), data['cols']

model, encoder, scaler, feature_names, cols = load_model()

def predict(data, feature_names):
    df = data.copy()
    categorical_cols = ['name','fuel','seller_type','transmission','owner','seats'] 
    if 'selling_price' in df.columns:
        df = df.drop('selling_price', axis=1)
    if 'seats' in df.columns:
        df['seats'] = df['seats'].astype(str)
    
    if 'name' in df.columns:
        df['name'] = df['name'].str.split().str[0]
    encoded_cats = encoder.transform(df[categorical_cols])
 
    encoded_cols = encoder.get_feature_names_out(categorical_cols)
    encoded_df = pd.DataFrame(encoded_cats, columns=encoded_cols, index=df.index)

    numerical_df = df.drop(columns=categorical_cols).copy()
    x_prepared = pd.concat([numerical_df, encoded_df], axis=1)
    print(x_prepared.columns)
    x_scaled = scaler.transform(x_prepared[feature_names])
    y_pred = model.predict(x_scaled)

    return y_pred

if st.button("Предсказать цену"):
        predictions = predict(df, feature_names)
        df['selling_price'] = predictions
        st.write("Результаты:", df)
        df_columns = df.drop('selling_price', axis=1).columns
        feature_list = list(feature_names)
        for col in df_columns:
          if col in feature_list:
            idx = feature_list.index(col)
            st.write(f"{col}: {model.coef_[idx]:.6f}")
    