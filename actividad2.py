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
    #Elimino valores nulos
    df = df.dropna(how='any',axis=0)
    if(rename):
        #Cambio nombre de las columnas a español para los gráficos
        df.rename(columns={'Age': 'Edad', 'Embarked': 'Embarque', 'Fare': 'Tarifa', 
                       'Sex':'Genero', 'Survived':'Sobrevivio', 'Pclass':'Clase'}, inplace=True)
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
        g1 = sns.countplot(df['Sobrevivio'])
        g1.set(ylabel="Número de personas", xlabel="", xticklabels=["Fallecidos", "Supervivientes"]) # "0=Died", "1=Sobrevivio"
        st.pyplot(fig1)

    with col2:
        st.write("Relación clase y fallecidos")
        fig2 = plt.figure(figsize=(5, 4))
        g2 = sns.countplot(x='Sobrevivio', data=df, hue = df['Clase'])
        plt.legend(loc='upper right')
        g2.set(ylabel="Número de personas", xlabel="", xticklabels=["Fallecidos", "Supervivientes"])
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


def main():
    st.title('Titanic dataset')

    #Cargar modelo
    model = load_model('assets/ultimo.h5')

    #Cargar dataset
    df = load_dataset()
    

    # Sidebar - filtros
    st.sidebar.header('Filtros')
    
    # Filtro clase
    pclass = ['Primera', 'Segunda', 'Tercera']
    selected_class = st.sidebar.multiselect('Clase', pclass, pclass)
    classes = {"Primera": 1, "Segunda": 2, "Tercera": 3}

    final_selection = []
    for i in selected_class:
        final_selection.append(classes[i])

    if(len(final_selection) == 0):
        final_selection = [1,2,3]

    # Sidebar - Filtro genero
    genre = ['female', 'male']
    selected_genre = st.sidebar.multiselect('Género', genre, genre)

    if(len(selected_genre) == 0):
        selected_genre = ['female', 'male']

    # Sidebar - Filtro edad
    values = st.sidebar.slider('Edad', 0, 100, (0, 100))

    # Sidebar - Filtro embarcacion
    embarked_list = ['Cherbourg', 'Queenstown', 'Southampton']
    selected_embarked = st.sidebar.multiselect('Lugar de embarcacion', embarked_list, embarked_list)
    embarked_dict = {"Cherbourg": 'C', "Queenstown": 'Q', "Southampton": 'S'}

    final_embarked = []
    for i in selected_embarked:
        final_embarked.append(embarked_dict[i])

    if(len(final_embarked) == 0):
        final_embarked = ['C', 'Q', 'S']


    df = df[(df.Clase.isin(final_selection))]
    df = df[(df.Embarque.isin(final_embarked))]
    df = df[(df.Edad > values[0]) & (df.Edad < values[1])]
    df = df[(df.Genero.isin(selected_genre))]

    df.reset_index(drop=True, inplace=True)


    col1, col2, col3 = st.columns(3)
    col1.metric("Pasajeros totales", df.shape[0])
    col2.metric("Sobrevivieron", df[(df.Sobrevivio == 1)].shape[0])
    col3.metric("Fallecieron", df[(df.Sobrevivio == 0)].shape[0])


    st.dataframe(df, height=315)
    #Descargar dataset
    st.markdown(download_file(df), unsafe_allow_html=True)


    #Información sobre el dataset
    dataset_info(df)

  
    #Graficas
    print_plots(df)
    

    # Matriz de correlación
    #Transformar variables categoricas en numericas
    df2 = load_dataset(rename=False)
    stacked = df2[['Embarked','Sex']].stack()
    df2[['Embarked','Sex']] = pd.Series(stacked.factorize()[0], index=stacked.index).unstack()
    #Elimino las variables Cabin, Name, Ticket y PassengerId ya que no son útiles para la creación de la red neuronal
    df2.drop(['Cabin', 'Name', 'Ticket', 'PassengerId'], axis=1, inplace=True)

    corr = df2.corr()
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style("white"):
        f, ax = plt.subplots(figsize=(5,3))
        ax = sns.heatmap(df2.corr(), annot=True, fmt='.2f', linewidths=2)
    st.pyplot(f)
    

    #PREDICCIONES
    st.sidebar.subheader('Realizar predicción')
    hola = st.sidebar.header('')
    
    age_prediction = st.sidebar.number_input('Edad', step=1, min_value = 0)

    embarked_list = st.sidebar.selectbox('Lugar de embarcación',('Cherbourg', 'Queenstown', 'Southampton'))
    embarked_dict = {"Cherbourg": 0, "Queenstown": 4, "Southampton": 2}
    embarked_prediction = embarked_dict[embarked_list]

    fare_prediction = st.sidebar.number_input('Tarifa del pasajero')

    parch_prediction = st.sidebar.number_input('Número de padres/hijos a bordo', step=1, min_value=0)

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
        st.dataframe(prueba)

        continuous = ['Edad', 'Tarifa', 'Parch', 'Clase', 'SibSp']
        scaler = StandardScaler()

        for var in continuous:
            prueba[var] = prueba[var].astype('float64')
            prueba[var] = scaler.fit_transform(prueba[var].values.reshape(-1, 1))
    
        predicted = model.predict(prueba)
        predicted_formated = round(float(predicted[0]),4) * 100

        if(predicted[0] > 0.5):
            cadena = "Sobrevivió con una probabilidad de " + str(predicted_formated) + " %"
        else:
            cadena = "Falleció con una probabilidad de " + str(predicted_formated) + " %"

        hola.header(cadena)
        st.sidebar.title(cadena)
    

    


main()

