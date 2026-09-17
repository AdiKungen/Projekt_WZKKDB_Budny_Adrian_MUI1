import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, TomekLinks

st.set_page_config(page_title="Adrian Budny Gr. 1 - Projekt WZKKDB", layout="wide")

st.title("Adrian Budny Gr. 1 - Projekt WZKKDB")

# PB: Wczytywanie danych
st.sidebar.header("1. Wczytywanie danych")

df = None
target_col = None

uploaded_file = st.sidebar.file_uploader("Wybierz plik CSV", type=["csv"])
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success("Plik wczytany pomyślnie")
        
        df = df.dropna(axis=1, how='all')
        
        all_cols = df.columns.tolist()
        target_col = st.sidebar.selectbox("Wybierz klasę decyzyjną:", all_cols, index=1)
    except Exception as e:
        st.error(f"Błąd wczytywania pliku: {e}")
        st.stop()
else:
    st.info("Oczekiwanie na plik CSV...")
    st.stop()

# PG: Przeglądanie danych surowych
with st.expander("Podgląd danych surowych", expanded=False):
    st.dataframe(df.head())
    st.write(f"Wymiary danych: {df.shape}")
    st.write("Statystyki opisowe:")
    st.dataframe(df.describe())

# PB: Wybieranie cech
feature_cols = [c for c in df.columns if c != target_col]
selected_features = st.sidebar.multiselect("Wybierz cechy do analizy:", feature_cols, default=feature_cols)

if not selected_features:
    st.warning("Musisz wybrać przynajmniej jedną cechę do analizy!")
    st.stop()

X_raw = df[selected_features]
y = df[target_col]

# PB: Uzupełnianie brakujących danych
st.sidebar.header("2. Imputacja")
missing_count = X_raw.isnull().sum().sum()

imputation_method = st.sidebar.selectbox("Metoda uzupełniania braków:", ["Brak", "Usuń wiersze", "Średnia", "Mediana", "Najczęstsza"])

X_imputed = X_raw.copy()

if missing_count > 0:
    if imputation_method == "Brak":
        X_imputed = X_raw.copy()
        missing_count = X_imputed.isnull().sum().sum()
    elif imputation_method == "Usuń wiersze":
        data_temp = pd.concat([X_raw, y], axis=1).dropna()
        X_imputed = data_temp[selected_features]
        y = data_temp[target_col]
        missing_count = X_imputed.isnull().sum().sum()
    else:
        strategy_map = {
            "Średnia": "mean",
            "Mediana": "median",
            "Najczęstsza": "most_frequent"
        }
        imputer = SimpleImputer(strategy=strategy_map[imputation_method])
        X_imputed = pd.DataFrame(imputer.fit_transform(X_raw), columns=selected_features)
        missing_count = X_imputed.isnull().sum().sum()

if missing_count > 0:
    st.sidebar.warning(f"Wykryto {missing_count} brakujących wartości!")
else:
    st.sidebar.success("Brak pustych wartości w wybranych kolumnach")

# PG: Przeglądanie danych po uzupełnieniu braków
with st.expander(f"Podgląd danych po Imputacji: {imputation_method}", expanded=False):
    st.dataframe(X_imputed.head())
    st.write(f"Czy są jeszcze braki? {'Tak' if X_imputed.isnull().sum().sum() > 0 else 'Nie'}")
    st.write(f"Wymiary: {X_imputed.shape}")
    st.write("Statystyki opisowe:")
    st.dataframe(X_imputed.describe())

# PB: Skalowanie
st.sidebar.header("3. Skalowanie")
scaler_option = st.sidebar.selectbox("Metoda skalowania:", ["Brak", "StandardScaler", "MinMaxScaler", "RobustScaler"])

X_processed = X_imputed.copy()

if scaler_option != "Brak":
    try:
        if scaler_option == "StandardScaler":
            scaler = StandardScaler()
        elif scaler_option == "MinMaxScaler":
            scaler = MinMaxScaler()
        elif scaler_option == "RobustScaler":
            scaler = RobustScaler()
        
        numeric_cols = X_processed.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            X_processed[numeric_cols] = scaler.fit_transform(X_processed[numeric_cols])
        st.sidebar.success(f"Zastosowano {scaler_option}")
    except Exception as e:
        st.sidebar.error(f"Błąd balansowania: {e}")

# PG: Przeglądanie danych po skalowaniu
with st.expander(f"Podgląd danych po Skalowaniu: {scaler_option}", expanded=False):
    st.dataframe(X_processed.head())
    st.write("Statystyki opisowe:")
    st.dataframe(X_processed.describe())

# PB: Balansowanie
st.sidebar.header("4. Balansowanie")
balance_type = st.sidebar.radio("Technika balansowania:", ["Brak", "Oversampling", "Undersampling"])

X_final, y_final = X_processed.copy(), y.copy()
balance_algo_name = "Brak"
nan_flag_bal = 0

if balance_type == "Oversampling":
    algo = st.sidebar.selectbox("Algorytm:", ["SMOTE", "ADASYN"])
    balance_algo_name = algo
    if algo == "SMOTE":
        sampler = SMOTE(random_state=42)
        nan_flag_bal = 1
    else:
        sampler = ADASYN(random_state=42)
        nan_flag_bal = 1
elif balance_type == "Undersampling":
    algo = st.sidebar.selectbox("Algorytm:", ["RandomUnderSampler", "TomekLinks"])
    balance_algo_name = algo
    if algo == "RandomUnderSampler":
        sampler = RandomUnderSampler(random_state=42)
    else:
        sampler = TomekLinks()
        nan_flag_bal = 1

if balance_type != "Brak":
    if X_processed.isnull().values.any() and nan_flag_bal == 1:
        st.sidebar.error("Wykryto brakujące dane (NaN) w zbiorze wejściowym!")
    else:
        try:
            X_final, y_final = sampler.fit_resample(X_processed, y)
            st.sidebar.success(f"Zastosowano {balance_algo_name}")
        except Exception as e:
            st.sidebar.error(f"Błąd balansowania: {e}")

# PG: Przeglądanie danych po balansowaniu
with st.expander(f"Podgląd rozkładu klas po Balansowaniu: {balance_algo_name}", expanded=False):
    col1, col2 = st.columns(2)
    
    counts_before = y.value_counts().reset_index()
    counts_before.columns = ['Klasa', 'Liczność']
    counts_before['Stan'] = 'Przed'
    
    counts_after = y_final.value_counts().reset_index()
    counts_after.columns = ['Klasa', 'Liczność']
    counts_after['Stan'] = 'Po'
    
    df_counts = pd.concat([counts_before, counts_after])
    fig_bal = px.bar(df_counts, x='Klasa', y='Liczność', color='Stan', barmode='group',
                     title=f"Efekt techniki: {balance_type} ({balance_algo_name})")
    
    st.plotly_chart(fig_bal, use_container_width=True)
    st.write(f"Liczba próbek przed: {len(X_processed)} | Po: {len(X_final)}")

# PG: Dolne zakładki
st.divider()
tab1, tab2, tab3 = st.tabs(["Redukcja wymiarowości", "Klasyfikacja pojedyncza", "Klasyfikacja wspólna"])

# PG: Zakładka - Redukcja wymiarowości
with tab1:
    st.subheader("Redukcja wymiarowości (PCA)")
    X_numeric = X_final.select_dtypes(include=[np.number])

    if X_final.isnull().values.any():
        st.error("Wykryto brakujące dane (NaN) w zbiorze wejściowym!")
    elif X_numeric.shape[1] > 1:
        pca_dim = st.radio("Wymiarowość:", ["2D", "3D"], horizontal=True)
        
        n_comp = 3 if pca_dim == "3D" else 2
        pca = PCA(n_components=n_comp)
        components = pca.fit_transform(X_numeric)
        
        pca_df = pd.DataFrame(components, columns=[f"PC{i+1}" for i in range(n_comp)])
        pca_df['Target'] = y_final.values if isinstance(y_final, pd.Series) else y_final
        
        pca_df['Target'] = pca_df['Target'].astype(str)
        
        if pca_dim == "2D":
            fig = px.scatter(pca_df, x='PC1', y='PC2', color='Target', title="PCA Wykres 2D")
        else:
            fig = px.scatter_3d(pca_df, x='PC1', y='PC2', z='PC3', color='Target', title="PCA Wykres 3D")
            
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Za mało danych numerycznych do wykonania PCA")

# PG: Zakładka - Klasyfikacja pojedyncza
with tab2:
    st.subheader("Klasyfikacja pojedyncza")
    
    X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.3, random_state=42)
    
    model_choice = st.selectbox("Algorytm:", ["Random Forest", "k-NN", "SVM"])
    nan_flag_train = 0

    model = RandomForestClassifier(random_state=42)
    if model_choice == "k-NN":
        model = KNeighborsClassifier()
        nan_flag_train = 1
    elif model_choice == "SVM":
        model = SVC()
        nan_flag_train = 1
    
    if X_final.isnull().values.any() and nan_flag_train == 1:
        st.error("Wykryto brakujące dane (NaN) w zbiorze wejściowym!")
    else:
        if st.button("Trenuj"):
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            
            acc = accuracy_score(y_test, preds)
            st.metric("Dokładność (Accuracy)", f"{acc:.2%}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.text("Raport klasyfikacji:")
                st.code(classification_report(y_test, preds))
            with col2:
                cm = confusion_matrix(y_test, preds)
                fig_cm = px.imshow(cm, text_auto=True, title="Macierz pomyłek")
                st.plotly_chart(fig_cm, use_container_width=True)

# PG: Zakładka - Klasyfikacja wspólna
with tab3:
    st.subheader("Klasyfikacja wspólna")
    if X_final.isnull().values.any():
        st.error("Wykryto brakujące dane (NaN) w zbiorze wejściowym!")
    else:
        if st.button("Porównaj 3 modele"):
            X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_final, y_final, test_size=0.3, random_state=42)
            
            results = []
            models = {"Random Forest": RandomForestClassifier(random_state=42), "k-NN": KNeighborsClassifier(), "SVM": SVC()}
            
            progress_bar = st.progress(0)
            idx = 0
            for name, m in models.items():
                m.fit(X_train_c, y_train_c)
                acc = m.score(X_test_c, y_test_c)
                results.append({"Model": name, "Accuracy": acc})
                idx += 1
                progress_bar.progress(idx / len(models))
                
            res_df = pd.DataFrame(results).sort_values("Accuracy", ascending=False)
            
            col1, col2 = st.columns([1, 2])
            with col1:
                st.dataframe(res_df)
            with col2:
                fig_bar = px.bar(res_df, x="Model", y="Accuracy", color="Accuracy", title="Porównanie skuteczności")
                st.plotly_chart(fig_bar, use_container_width=True)