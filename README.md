# Sistema de Recomendación
## Panel público interactivo para un sistema de recomendación de películas con interfaz gradio.

![Banner](docs/assets/images/movie_banner.png)

El script utiliza filtrado colaborativo elemento a elemento con similitud de coseno centrada.
Procesa los datos de calificación para tener en cuenta los sesgos de calificación de los usuarios. Utiliza un cálculo eficiente de similitud con Scikit-learn. Utiliza una interfaz desplegable interactiva con Gradio. Y gestiona datos dispersos mediante el centrado de la media.

### Código Python:

### 1. Importar bibliotecas
```
import pandas as pd
import gradio as gr
from sklearn.metrics.pairwise import cosine_similarity
```
### 2. Cargar y preprocesar los datos
```
movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")
```
### 3. Crear una matriz de similaridad item-item usando filtrado colaborativo
```
merged_data = ratings.merge(movies, on='movieId')
```
### 4. Crear una matriz usuario-item con ratings centrados en la media
```
mean_ratings = merged_data.groupby('movieId')['rating'].mean()
merged_data['centered_rating'] = merged_data.apply(
    lambda row: row['rating'] - mean_ratings[row['movieId']], axis=1
)
user_item_matrix = merged_data.pivot_table(
    index='movieId', 
    columns='userId', 
    values='centered_rating', 
    fill_value=0
)
```
### 5. Calcular la similitud del coseno entre películas
```
cosine_sim = cosine_similarity(user_item_matrix)
cosine_sim_df = pd.DataFrame(
    cosine_sim,
    index=user_item_matrix.index,
    columns=user_item_matrix.index
)
```
### 6. Función de recomendación
```
def recommend_movies(movie_title):
    try:
        movie_id = movies.loc[movies['title'] == movie_title, 'movieId'].iloc[0]
        sim_scores = cosine_sim_df[movie_id].sort_values(ascending=False).head(11)
        sim_scores = sim_scores.drop(movie_id).head(10)
        recommended_movies = movies.set_index('movieId').loc[sim_scores.index]['title']
        return "\n".join(f"{i+1}. {movie}" for i, movie in enumerate(recommended_movies))
    except:
        return "Película que no encontrada en la base de datos"
```
### 7. Crear interfaz Gradio
```
interface = gr.Interface(
    fn=recommend_movies,
    inputs=gr.Dropdown(
        choices=movies['title'].unique().tolist(),
        label="Seleccione una película"),
    outputs=gr.Textbox(label="Películas Recomendadas"),
    title="Sistema de Recomendación de Películas",
    description="Seleccione una película para obtener recomendaciones similares basadas en Filtrado Colaborativo"
)

interface.launch()
```
## RESULTADO:

![Interfaz](docs/assets/images/sistema_de_recomendacion.png)

