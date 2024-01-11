# Import des bibliothèques
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from io import StringIO
from unidecode import unidecode
from fuzzywuzzy import fuzz, process
from streamlit_folium import folium_static

##**Mini-projet data - Lieux culturels :**

"""## **Mini-projet data:  Lieux culturels**

"""

# Définition des liens vers les fichiers CSV
url_horaires = "https://raw.githubusercontent.com/WildCodeSchool/data-training-resources/main/quests/bi_case/horaires.csv"
url_lieux = "https://gist.githubusercontent.com/ThomasG77/963f269756daf33c4c1b5e786d5914a4/raw/f3b878b771bacea1471d78290703d5d614b442f0/lieux_culture_nantes.csv"
url_frequentation = "https://raw.githubusercontent.com/WildCodeSchool/data-training-resources/main/quests/bi_case/frequentation.csv"

# Chargement des fichiers CSV dans des DataFrames
horaires_df = pd.read_csv(url_horaires, delimiter=';')
lieux_df = pd.read_csv(url_lieux)
freq_df = pd.read_csv(url_frequentation, sep=";")


# Séparer la colonne "localisation" en deux colonnes "Latitude" et "Longitude" dans freq_df
freq_df[['Latitude', 'Longitude']] = freq_df['localisation'].str.split(', ', expand=True)

# Supprimer la colonne 'localisation' de freq_df
freq_df = freq_df.drop('localisation', axis=1)

# Supprimer les accents, mettre en minuscule et supprimer les espaces en plus pour certaines colonnes dans horaires_df
columns_to_clean_horaires = ['Equipement', 'Période', 'Jour de la semaine']
horaires_df[columns_to_clean_horaires] = horaires_df[columns_to_clean_horaires].applymap(lambda x: unidecode(str(x)).lower().strip())

# Supprimer les accents, mettre en minuscule et supprimer les espaces en plus pour certaines colonnes dans lieux_df
columns_to_clean_lieux = ['nom_comple', 'libtheme', 'libcategor', 'libtype', 'commune', 'adresse']
lieux_df[columns_to_clean_lieux] = lieux_df[columns_to_clean_lieux].applymap(lambda x: unidecode(str(x)).lower().strip())

# Supprimer les accents, mettre en minuscule et supprimer les espaces en plus pour certaines colonnes dans freq_df
columns_to_clean_freq = ['Nom', 'Commune', 'Type']
freq_df[columns_to_clean_freq] = freq_df[columns_to_clean_freq].applymap(lambda x: unidecode(str(x)).lower().strip())



# Colonne sur laquelle effectuer la jointure
colonne_jointure_1 = 'Equipement'
colonne_jointure_2 = 'nom_comple'

# Seuil de correspondance floue (ajustez selon vous besoin)
seuil_correspondance = 65

# Fonction pour trouver la meilleure correspondance pour chaque élément dans horaires_df
def trouver_meilleure_correspondance(row):
    correspondances = process.extractOne(row[colonne_jointure_1], lieux_df[colonne_jointure_2], scorer=fuzz.ratio)
    if correspondances[1] >= seuil_correspondance:
        return lieux_df.loc[correspondances[2]]['nom_comple']
    else:
        return None

# Appliquer la fonction de correspondance pour créer une nouvelle colonne dans horaires_df
horaires_df['correspondance_nom'] = horaires_df.apply(trouver_meilleure_correspondance, axis=1)

# Effectuer la jointure sur la nouvelle colonne
up_horaires_df = pd.merge(horaires_df, lieux_df, how='left', left_on='correspondance_nom', right_on='nom_comple')



# Les données géographiques
donnees_geographiques_csv = """
Equipment,X,Y
bibliotheque de l'ecole des beaux-arts de nantes saint-nazaire,47.206591,-1.559844
le chronographe,47.191571,-1.566421
le miroir d'eau,47.216379,-1.550475
memorial de l'abolition de l'esclavage,47.20935518190213,-1.5646432008346078
"""

# Lire les données dans un DataFrame
donnees_geographiques_df = pd.read_csv(StringIO(donnees_geographiques_csv))

# Fusionner les DataFrames sur la colonne 'Equipement'
merge_result = pd.merge(up_horaires_df, donnees_geographiques_df[['Equipment', 'X', 'Y']], how='left', left_on='Equipement', right_on='Equipment')

# Mettre à jour les valeurs manquantes de 'X' et 'Y' dans up_horaires_df avec les valeurs correspondantes de merge_result
up_horaires_df['X'].update(merge_result['Y_y'])
up_horaires_df['Y'].update(merge_result['X_y'])


# Assurez-vous que les colonnes de date sont au format datetime
up_horaires_df['Date de début'] = pd.to_datetime(up_horaires_df['Date de début'], errors='coerce')
up_horaires_df['Date de fin'] = pd.to_datetime(up_horaires_df['Date de fin'], errors='coerce')
up_horaires_df['Date de l\'exception'] = pd.to_datetime(up_horaires_df['Date de l\'exception'], errors='coerce')

# Remplacer la valeur spécifique dans la colonne "Période"
up_horaires_df['Période'] = up_horaires_df['Période'].replace("vac scol fev 2020", "vac scol fev 2022")



# Liste des colonnes à supprimer
colonnes_a_supprimer = [
    'gid', '___idobj',
    'theme', 'libtheme', 'categorie', 'type',
    'statut', 'adresse', 'telephone', 'web', 'code_posta'
]

# Supprimer les colonnes spécifiées
up_horaires_df = up_horaires_df.drop(colonnes_a_supprimer, axis=1)

# Afficher les premières lignes du DataFrame après la suppression



# Utiliser str.contains pour rechercher des mots spécifiques dans la colonne "Equipement"
biblio_keywords = ['bibliotheque', 'mediatheque', 'ludotheque']
up_horaires_df['Catégorie'] = up_horaires_df['Equipement'].str.lower().apply(lambda x: any(keyword in x for keyword in biblio_keywords))

# Mapper les résultats booléens à "Bibliothèque" ou "Musée"
up_horaires_df['Catégorie'] = up_horaires_df['Catégorie'].map({True: 'Bibliothèque', False: 'Musée'})

# Renommer les colonnes "X" et "Y" en "Latitude" et "Longitude"
up_horaires_df = up_horaires_df.rename(columns={'Y': 'Latitude', 'X': 'Longitude'})

# Filtrer les lignes où au moins une colonne de date est supérieure ou égale à 2022
date_columns = ['Date de début', 'Date de fin', 'Date de l\'exception']
up_horaires_df = up_horaires_df[(up_horaires_df[date_columns].apply(lambda x: x.dt.year >= 2022)).any(axis=1) & (up_horaires_df["Type d'horaire"] == "Ouverture")]


#**********************************************************************

def create_filtered_map(category='Tous', day='Tous', equipment='Tous'):
    # Filtrer le DataFrame par catégorie si ce n'est pas 'Toutes'
    filtered_df = up_horaires_df if category == 'Tous' else up_horaires_df[up_horaires_df['Catégorie'] == category]

    # Filtrer par jour de la semaine si spécifié
    if day != 'Tous':
        filtered_df = filtered_df[filtered_df['Jour de la semaine'] == day]

    # Filtrer par équipement si spécifié
    if equipment != 'Tous':
        filtered_df = filtered_df[filtered_df['Equipement'] == equipment]

    # Filtrer les horaires normaux
    filtered_df = filtered_df[filtered_df["Type d'horaire"] == 'Ouverture']

    # Vérifier si le DataFrame filtré n'est pas vide
    if not filtered_df.empty:
        # Créer une carte centrée sur la première entrée des coordonnées du DataFrame filtré
        carte = folium.Map(location=[filtered_df['Latitude'].iloc[0], filtered_df['Longitude'].iloc[0]], zoom_start=14)

        # Ajouter des marqueurs pour chaque entrée du DataFrame filtré
        for index, row in filtered_df.iterrows():
            # Créer la chaîne de texte pour les horaires
            horaires_text = f"Heures d'ouverture : {row['Heure de début']} - {row['Heure de fin']}"

            # Ajouter le marqueur avec popup
            folium.Marker(
                location=[row['Latitude'], row['Longitude']],
                popup=f"{row['Equipement']}<br>{horaires_text}"
            ).add_to(carte)

        # Afficher la carte dans Streamlit
        st.title(f"Carte filtrée par catégorie : {category}, jour : {day}, équipement : {equipment}")
        folium_static(carte)
    else:
        st.warning("Aucune donnée disponible avec les filtres sélectionnés.")



#-------------------------------------

# Filtres interactifs
st.sidebar.title("Filtres pour les horaires")

categorie_selectionnee = st.sidebar.selectbox('Sélectionner une catégorie:', ['Tous'] + up_horaires_df['Catégorie'].unique().tolist(), key='categorie_selector_' + str(hash(up_horaires_df['Catégorie'].to_string())))
equipement_selectionne = st.sidebar.selectbox('Sélectionner un équipement:', ['Tous'] + up_horaires_df['Equipement'].unique().tolist(), key='equipement_selector_' + str(hash(up_horaires_df['Equipement'].to_string())))
jour_selectionne = st.sidebar.selectbox('Sélectionner un jour de la semaine:', ['Tous'] + up_horaires_df['Jour de la semaine'].unique().tolist(), key='jour_selector_' + str(hash(up_horaires_df['Jour de la semaine'].to_string())))

# Filtrer les données en fonction des sélections
filtered_df = up_horaires_df
if categorie_selectionnee != 'Tous':
    filtered_df = filtered_df[filtered_df['Catégorie'] == categorie_selectionnee]
if jour_selectionne != 'Tous':
    filtered_df = filtered_df[filtered_df['Jour de la semaine'] == jour_selectionne]
if equipement_selectionne != 'Tous':
    filtered_df = filtered_df[filtered_df['Equipement'] == equipement_selectionne]


# Afficher la carte filtrée en fonction des filtres sélectionnés
create_filtered_map(category=categorie_selectionnee, day=jour_selectionne,equipment=equipement_selectionne)

# Afficher les horaires d'ouverture filtrés
st.title("Horaires d'ouverture des équipements culturels sélectionnés")
st.dataframe(filtered_df[['Equipement', 'Jour de la semaine', 'Heure de début', 'Heure de fin']])

#########################################################################################################################

# Supprimer les accents et caractères spéciaux des noms de colonnes
freq_df.columns = [unidecode(col) for col in freq_df.columns]

# Afficher les noms de colonnes modifiés
#print(freq_df.columns)

import re

# Définir une fonction pour supprimer les caractères spéciaux
def remove_special_characters(column_name):
    return re.sub(r"[^a-zA-Z0-9_]", "", column_name)

# Appliquer la fonction à chaque nom de colonne
freq_df.columns = [remove_special_characters(col) for col in freq_df.columns]


#########################################################################################################################
#*************************************************

nantais_df = freq_df[freq_df['Commune'] == 'nantes']

st.sidebar.title("Filtres Pour les Nombre d'entrer")
# Filtrer par année
annee_option = ['Toutes'] + nantais_df['Annee'].unique().tolist()
annee_selectionnee = st.sidebar.selectbox('Sélectionner une année:', annee_option)

if annee_selectionnee != 'Toutes':
    nantais_df = nantais_df[nantais_df['Annee'] == annee_selectionnee]

# Visualisation de l'évolution de la fréquentation au fil des années
st.title(f'Évolution de la fréquentation des lieux nantais en {annee_selectionnee}')
fig_evolution = plt.figure(figsize=(12, 6))
sns.lineplot(x= 'Annee' , y="Nombredentrees", hue='Nom', data= nantais_df)
st.pyplot(fig_evolution)

# Distribution des entrées par type de lieu
st.title(f'Distribution des entrées par type de lieu à Nantes en {annee_selectionnee}')
fig_distribution = plt.figure(figsize=(10, 6))
sns.boxplot(x='Type', y='Nombredentrees', data=nantais_df)
plt.xticks(rotation=45, ha='right')
st.pyplot(fig_distribution)

# Carte interactive
st.title(f'Carte interactive des dréquantation des  lieux culuturelle nantais en {annee_selectionnee}')
folium_map = folium.Map(location=[47.2184, -1.5536], zoom_start=13)

for index, row in nantais_df.iterrows():
    popup_text = f"{row['Nom']}<br>Année: {row['Annee']}<br>Entrées: {row['Nombredentrees']}"
    folium.Marker(
        location=[float(row['Latitude']), float(row['Longitude'])],
        popup=folium.Popup(popup_text, max_width=300),
        icon=folium.Icon(color='blue')
    ).add_to(folium_map)

folium_static(folium_map)

st.dataframe(nantais_df[['Nom','Annee', 'Commune', 'Type', 'Nombredentrees']])
