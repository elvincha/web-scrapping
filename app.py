import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import folium
from streamlit_folium import folium_static

# Configurer la page
st.set_page_config(page_title="Recommandation de Restaurants", layout="wide")

# Titre de l'application
st.title("Application de Recommandation de Restaurants")

@st.cache_data
def load_data():
    df = pd.read_csv('restaurant_final.csv')
    df['cuisine'] = df['cuisine'].fillna('Inconnu')
    df['caracteristiques'] = df['type'] + ' ' + df['cuisine'] + ' ' + df['address'] + ' ' + df['vegan'].astype(str) + ' ' + df['vegetarian'].astype(str)
    return df

df = load_data()

# Prétraitement pour la similarité cosinus
@st.cache_data
def compute_cosine_similarity(df):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['caracteristiques'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim

cosine_sim = compute_cosine_similarity(df)

# Fonction de recommandation par similarité
def recommander_restaurant(nom_restaurant, df, cosine_sim):
    indices = pd.Series(df.index, index=df['restaurant_name'].str.lower()).drop_duplicates()
    idx = indices.get(nom_restaurant.lower())
    if idx is None:
        st.error("Restaurant non trouvé. Veuillez vérifier le nom.")
        return None, None
    
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:4]  # Top 3 similaires
    
    restaurant_indices = [i[0] for i in sim_scores]
    return df.iloc[idx], df.iloc[restaurant_indices]

# Fonction de recommandation par coordonnées
def recommander_par_coordonnees(coord_adresse, df, n_recommandations=5, note_minimale=0):
    df_filtre = df[df['reviews_average'] >= note_minimale].copy()
    df_filtre['distance'] = df_filtre.apply(
        lambda row: geodesic(coord_adresse, (row['meta_geo_point_lat'], row['meta_geo_point_lon'])).kilometers, axis=1)
    df_recommandations = df_filtre.sort_values(by='distance').head(n_recommandations)
    return df_recommandations

# Sidebar pour la navigation
st.sidebar.title("Options de Recommandation")
option = st.sidebar.selectbox("Choisissez une option", ["Par Similarité de Restaurant", "Par Localisation et Note"])

# Recommandation par Similarité de Restaurant
if option == "Par Similarité de Restaurant":
    st.header("Recommandation Basée sur la Similarité de Restaurant")
    restaurant_nom = st.selectbox("Sélectionnez un restaurant", df['restaurant_name'].str.capitalize())
    
    if st.button("Recommander"):
        restaurant_initial, recommendations = recommander_restaurant(restaurant_nom, df, cosine_sim)
        if restaurant_initial is not None and recommendations is not None:
            # Afficher les détails du restaurant initial
            st.subheader(f"Détails de {restaurant_initial['restaurant_name']}")
            st.write(f"**Adresse:** {restaurant_initial['address']}")
            st.write(f"**Type:** {restaurant_initial['type']}")
            st.write(f"**Cuisine:** {restaurant_initial['cuisine']}")
            st.write(f"**Note Moyenne:** {restaurant_initial['reviews_average']} / 5")
            st.write(f"**Nombre d'Avis:** {restaurant_initial['reviews_count']}")
    
            # Afficher les recommandations
            st.subheader("Restaurants Recommandés")
            st.table(recommendations[['restaurant_name', 'address', 'type', 'cuisine', 'reviews_average', 'reviews_count']])
    
            # Créer une carte
            map_center = [restaurant_initial['meta_geo_point_lat'], restaurant_initial['meta_geo_point_lon']]
            carte = folium.Map(location=map_center, zoom_start=14)
    
            # Ajouter le restaurant initial
            folium.Marker(
                location=map_center,
                popup=f"{restaurant_initial['restaurant_name']} - {restaurant_initial['address']}",
                icon=folium.Icon(color='red')
            ).add_to(carte)
    
            # Ajouter les recommandations
            for _, row in recommendations.iterrows():
                folium.Marker(
                    location=[row['meta_geo_point_lat'], row['meta_geo_point_lon']],
                    popup=f"{row['restaurant_name']} \n\n {row['address']}",
                    icon=folium.Icon(color='blue')
                ).add_to(carte)
            
            # Afficher la carte
            folium_static(carte, width=700, height=500)

# Recommandation par Localisation et Note
elif option == "Par Localisation et Note":
    st.header("Recommandation Basée sur la Localisation et la Note")
    geolocator = Nominatim(user_agent="my_geocoder")
    
    adresse_utilisateur = st.text_input("Entrez votre adresse", "131 Boulevard Exelmans, 75016 Paris")
    note_minimale = st.slider("Note minimale", 0.0, 5.0, 4.0, step=0.1)
    n_recommandations = st.number_input("Nombre de recommandations", min_value=1, max_value=20, value=5)
    
    if st.button("Trouver des Restaurants"):
        location = geolocator.geocode(adresse_utilisateur)
        
        if location:
            coord_exemple = (location.latitude, location.longitude)
            recommendations = recommander_par_coordonnees(coord_exemple, df, n_recommandations, note_minimale)
            
            st.subheader(f"Restaurants près de {adresse_utilisateur}")
            st.table(recommendations[['restaurant_name', 'address', 'type', 'cuisine', 'reviews_average', 'reviews_count', 'distance']].round(2))
            
            # Créer une carte
            carte = folium.Map(location=coord_exemple, zoom_start=14)
            
            # Ajouter le point de référence
            folium.Marker(
                location=coord_exemple,
                popup=f"Votre emplacement ({adresse_utilisateur})",
                icon=folium.Icon(color='green')
            ).add_to(carte)
            
            # Ajouter les restaurants recommandés
            for _, row in recommendations.iterrows():
                popup_info = f"""
                <strong>{row['restaurant_name']}</strong><br>
                Adresse: {row['address']}<br>
                Note moyenne: {row['reviews_average']} / 5<br>
                Nombre d'avis: {row['reviews_count']}<br>
                Distance: {row['distance']:.2f} km
                """
                folium.Marker(
                    location=[row['meta_geo_point_lat'], row['meta_geo_point_lon']],
                    popup=popup_info,
                    icon=folium.Icon(color='blue')
                ).add_to(carte)
            
            # Afficher la carte
            folium_static(carte, width=700, height=500)
        else:
            st.error("Adresse non trouvée. Veuillez vérifier l'adresse saisie.")
