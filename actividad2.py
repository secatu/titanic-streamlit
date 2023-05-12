import streamlit as st
import pandas as pd
import numpy as np
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.preprocessing import StandardScaler
from keras.models import load_model

def load_dataset(rename=True):
    df = pd.read_csv('assets/dataset.csv', sep=";")
    df = df.dropna(how='any',axis=0)
    
    df.rename(columns={'Age': 'Edad', 'Cabin': 'Cabina', 'Embarked': 'Embarque', 'Fare': 'Tarifa', 
                       'Sex':'Genero', 'Name':'Nombre', 'Survived':'Sobrevivio', 'Pclass':'Clase'}, inplace=True)
    df = df.replace({'Genero' : { 'male' : 'hombre', 'female' : 'mujer'}})
    df = df.replace({'Embarque' : { 'C' : 'Cherbourg', 'Q' : 'Queenstown', 'S': 'Southampton'}})
    df = df.replace({'Clase' : { 1 : 'Primera', 2 : 'Segunda', 3: 'Tercera'}})
    df = df.replace({'Sobrevivio' : { 1 : 'Si', 0 : 'No'}})

    df.reset_index(drop=True, inplace=True)

    return df

def download_file(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="dataset.csv">Descargar archivo csv</a>'
    return href

def dataset_info(df):
    st.subheader("Información sobre el dataset actual:")
    st.write('Número de filas: ' + str(df.shape[0]))
    st.write('Número de columnas: ' + str(df.shape[1]))

    st.write("Información sobre variables numéricas:")
    st.dataframe(df.describe(), width=700)

    st.write("Información sobre variables categóricas:")

    col1, col2 = st.columns(2)
    col1.write("Género")
    col1.dataframe(pd.DataFrame(df['Genero'].value_counts()))

    col2.write("Embarcación")
    col2.dataframe(pd.DataFrame(df['Embarque'].value_counts()))


def print_plots(df):
    col1, col2 = st.columns(2)

    with col1:
        st.write("Número de supervivientes y fallecidos")
        fig1 = plt.figure(figsize=(5, 4))
        g1 = sns.countplot(x='Sobrevivio', data=df);
        g1.set(ylabel="Número de personas", xlabel="")
        st.pyplot(fig1)

    with col2:
        st.write("Relación clase y fallecidos")
        fig2 = plt.figure(figsize=(5, 4))
        g2 = sns.countplot(x='Sobrevivio', data=df, hue = df['Clase'])
        plt.legend(loc='upper right')
        g2.set(ylabel="Número de personas", xlabel="")
        st.pyplot(fig2)
        
    col3, col4 = st.columns(2)

    with col3:
        st.write("Boxplot con relación entre edad y Género")
        fig3 = plt.figure(figsize=(5, 4))
        g3 = sns.boxplot(data=df, y='Edad', x='Genero')
        st.pyplot(fig3)

    with col4:
        st.write("Boxplot con relación entre tarifa y Género")
        fig4 = plt.figure(figsize=(5, 4))
        g4 = sns.boxplot(data=df, y='Tarifa', x='Genero')
        st.pyplot(fig4)

    col5, col6 = st.columns(2)

    with col5:
        st.write("Relación embarque y fallecidos")
        fig2 = plt.figure(figsize=(5, 4))
        g2 = sns.countplot(x='Sobrevivio', data=df, hue = df['Embarque'])
        plt.legend(loc='upper right')
        g2.set(ylabel="Número de personas", xlabel="")
        st.pyplot(fig2)

    with col6:
        st.write("Relación embarque y género")
        fig2 = plt.figure(figsize=(5, 4))
        g2 = sns.countplot(x='Embarque', data=df, hue = df['Genero'])
        plt.legend(loc='upper right')
        g2.set(ylabel="Número de personas", xlabel="")
        st.pyplot(fig2)

    col7, col8 = st.columns(2)

    with col7:
        st.write("Relación número hijos/padres y fallecidos")
        fig2 = plt.figure(figsize=(5, 4))
        g2 = sns.countplot(x='Sobrevivio', data=df, hue = df['Parch'])
        plt.legend(loc='upper right')
        g2.set(ylabel="Número de personas", xlabel="")
        st.pyplot(fig2)

    with col8:
        st.write("Relación número hermanos/conyugues y fallecidos")
        fig2 = plt.figure(figsize=(5, 4))
        g2 = sns.countplot(x='Sobrevivio', data=df, hue = df['SibSp'])
        plt.legend(loc='upper right')
        g2.set(ylabel="Número de personas", xlabel="")
        st.pyplot(fig2)

    col9, col10 = st.columns(2)

    with col9:

        st.write("Relación edad y tarifa")
        fig2 = plt.figure(figsize=(5, 4))
        g2 = plt.scatter(x=df["Edad"], y=df['Tarifa'], alpha=0.1)
        plt.xlabel("Edad")
        plt.ylabel('Tarifa')
        st.pyplot(fig2)

    with col10:
        
        st.write("Relación tarifa y clase")
        fig2 = plt.figure(figsize=(5, 4))
        g2 = plt.scatter(x=df["Clase"], y=df['Tarifa'], alpha=0.1)
        plt.xlabel("Clase")
        plt.ylabel('Tarifa')
        st.pyplot(fig2)


def main():
    st.title('Titanic dataset')

    #Cargar modelo
    with open('assets/modelo.pickle', 'rb') as file:
        model = pickle.load(file)

    #Cargar dataset
    df = load_dataset()

    # Sidebar - filtros
    st.sidebar.header('Filtros')
    
    # Filtro clase
    pclass = ['Primera', 'Segunda', 'Tercera']
    selected_class = st.sidebar.multiselect('Clase', pclass, pclass)
    if(len(selected_class) == 0):
        selected_class = pclass

    # Sidebar - Filtro genero
    genre = ['mujer', 'hombre']
    selected_genre = st.sidebar.multiselect('Género', genre, genre)

    if(len(selected_genre) == 0):
        selected_genre = ['mujer', 'hombre']

    # Sidebar - Filtro edad
    age_values = st.sidebar.slider('Edad', 0, 100, (0, 100))

    # Sidebar - Filtro Tarifa
    tarifa_values = st.sidebar.slider('Tarifa', 0, 600, (0, 600))

    # Sidebar - Filtro embarcacion
    embarked_list = ['Cherbourg', 'Queenstown', 'Southampton']
    selected_embarked = st.sidebar.multiselect('Lugar de embarcacion', embarked_list, embarked_list)
    if(len(selected_embarked) == 0):
        selected_embarked = embarked_list

    # Sidebar - Filtro sobrevivio
    sobrevivio_list = ['Si', 'No']
    selected_sobrevivio = st.sidebar.multiselect('¿Sobrevivió?', sobrevivio_list, sobrevivio_list)
    if(len(selected_sobrevivio) == 0):
        selected_sobrevivio = sobrevivio_list

    df = df[(df.Clase.isin(selected_class))]
    df = df[(df.Embarque.isin(selected_embarked))]
    df = df[(df.Sobrevivio.isin(selected_sobrevivio))]
    df = df[(df.Edad > age_values[0]) & (df.Edad < age_values[1])]
    df = df[(df.Tarifa > tarifa_values[0]) & (df.Tarifa < tarifa_values[1])]
    df = df[(df.Genero.isin(selected_genre))]

    df.reset_index(drop=True, inplace=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Pasajeros totales", df.shape[0])
    col2.metric("Sobrevivieron", df[(df.Sobrevivio == 'Si')].shape[0])
    col3.metric("Fallecieron", df[(df.Sobrevivio == 'No')].shape[0])


    st.dataframe(df, height=315)
    #Descargar dataset
    st.markdown(download_file(df), unsafe_allow_html=True)


    #Información sobre el dataset
    dataset_info(df)

  
    #Graficas
    print_plots(df)
    

    
    # Matriz de correlación
    st.write("Matriz de correlación")
    #Transformar variables categoricas en numericas
    df2 = load_dataset()
    stacked = df2[['Embarque', 'Genero', 'Clase', 'Sobrevivio']].stack()
    df2[['Embarque', 'Genero', 'Clase', 'Sobrevivio']] = pd.Series(stacked.factorize()[0], index=stacked.index).unstack()
    #Elimino las variables Cabin, Name, Ticket y PassengerId ya que no son útiles para la creación de la red neuronal
    df2.drop(['Cabina', 'Nombre', 'Ticket', 'PassengerId'], axis=1, inplace=True)

    corr = df2.corr()
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style("white"):
        f, ax = plt.subplots(figsize=(5,3))
        ax = sns.heatmap(df2.corr(), annot=True, fmt='.2f', linewidths=2)
    st.pyplot(f)
    

    #PREDICCIONES
    st.sidebar.subheader('')
    st.sidebar.subheader('Realizar predicción')
    prediction_result = st.sidebar.header('')
    
    age_prediction = st.sidebar.number_input('Edad', step=1, min_value = 0)

    embarked_list = st.sidebar.selectbox('Lugar de embarcación',('Cherbourg', 'Queenstown', 'Southampton'))
    embarked_dict = {"Cherbourg": 0, "Queenstown": 4, "Southampton": 2}
    embarked_prediction = embarked_dict[embarked_list]

    fare_prediction = st.sidebar.number_input('Tarifa del pasajero')

    parch_prediction = st.sidebar.number_input('Número de padres / hijos a bordo', step=1, min_value=0)

    class_list = st.sidebar.selectbox('Clase',('Primera', 'Segunda', 'Tercera'))
    class_dict = {"Primera": 1, "Segunda": 2, "Tercera": 3}
    class_prediction = class_dict[class_list]

    genre_list = st.sidebar.selectbox('Género',('Hombre', 'Mujer'))
    genre_dict = {"Hombre": 3, "Mujer": 1}
    genre_prediction = genre_dict[genre_list]

    sibsp_prediction = st.sidebar.number_input('Número de hermanos / cónyuges a bordo', step=1, min_value=0)

    if st.sidebar.button('Realizar predicción'):
        row = [[age_prediction, embarked_prediction, fare_prediction, parch_prediction, class_prediction, genre_prediction,sibsp_prediction]]
        prueba = pd.DataFrame(row, columns=['Edad', 'Embarque', 'Tarifa', 'Parch', 'Clase', 'Genero', 'SibSp']) 

        continuous = ['Edad', 'Tarifa', 'Parch', 'Clase', 'SibSp']
        scaler = StandardScaler()

        for var in continuous:
            prueba[var] = prueba[var].astype('float64')
            prueba[var] = scaler.fit_transform(prueba[var].values.reshape(-1, 1))
    
        predicted = model.predict(prueba)
        predicted = round((predicted[0][0] * 100), 2)

        if(predicted > 50):
            cadena = "Sobrevivió con una probabilidad de " + str(predicted) + " %"
        else:
            cadena = "Falleció con una probabilidad de " + str(predicted) + " %"

        prediction_result.header(cadena)
    

main()

