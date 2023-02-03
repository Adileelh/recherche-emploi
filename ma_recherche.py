import joblib
import dill
from sklearn import set_config
from sklearn.metrics import r2_score
from sklearn.compose import TransformedTargetRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, RobustScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import re
import warnings
import seaborn as sns
warnings.simplefilter('ignore')


fn_data = 'data.json'

df = pd.read_json(fn_data, encoding='utf8')

# Conversion du .json en csv
df.to_csv("data.csv", index=False)

# ---------------------------------------------------------------------------------------------------------
# NETTOYAGE DE LA COLONNE LIEU
# ----------------------------------------------------------------------------------------------------------

# extraction des lieux et salaires min et max a partir de la colonne lieu
split_df = pd.DataFrame(df['lieu'].tolist(), columns=['location', 'salaire'])

# Ajout des nouvelles colonne au df
df = pd.concat([df, split_df], axis=1)
df = df.drop(['lieu'], axis=1)
df['lieu'] = df['location']
df.drop(['location'], axis=1, inplace=True)

# lieu
df['lieu'] = df['lieu'].apply(lambda x: ''.join(
    map(str, x)).replace("\n", "").lower())
df['lieu']
# lieu modification

# Lieu
s = [item for item in df['lieu']]
pat = re.compile(r'^ile.*')
pat2 = re.compile(r'^pari.*')
pat3 = re.compile(r'^france - an.*')
pat4 = re.compile(r'^guyancourt.*')
pat5 = re.compile(r'^île.*')
pat6 = re.compile(r'^la défense,*')
pat7 = re.compile(r'^neuilly sur seine*')
pat8 = re.compile(r'^paray vieille poste*')
pat9 = re.compile(r'^rueil *')
list_lieu = []
for item in s:
    # recherche dans la list les lines lieu qui commence par pat
    test = bool(pat.search(item))
    test2 = bool(pat2.search(item))
    test3 = bool(pat3.search(item))
    test4 = bool(pat4.search(item))
    test5 = bool(pat5.search(item))
    test6 = bool(pat6.search(item))
    test7 = bool(pat7.search(item))
    test8 = bool(pat8.search(item))
    test9 = bool(pat9.search(item))
   # remplacer chaque line de lieu non hemogenic par une seul meme valeur
    if test == True:
        list_lieu.append('île-de-france')
    elif test2 == True:
        list_lieu.append('paris')
    elif test3 == True:
        list_lieu.append('antony')
    elif test4 == True:
        list_lieu.append('guyancourt')
    elif test6 == True:
        list_lieu.append('la defense')
    elif test7 == True:
        list_lieu.append('neuilly-sur-seine')
    elif test8 == True:
        list_lieu.append('paray-vieille-poste')
    elif test9 == True:
        list_lieu.append('rueil-malmaison')

    else:
        list_lieu.append(item)
df['lieu'] = list_lieu
# print(df['lieu'].head(230).sort_values(ascending = True))

# Suppression des caracteres non désirées dans la colonne compétence et rajout d'une virgule apres chaque élements "\n"
df['competences'] = df['competences'].apply(lambda x: ','.join(map(str, x))
                                            .replace("\n", " ")
                                            .replace(" ", "")
                                            .replace(",", ", "))
# df['competences']


# on retire tout ce qui n'est pas un chiffre pour pouvoir convertir en float
df['salaire'] = df['salaire'].apply(lambda x: ''.join(map(str, x))
                                    .replace("\n", "").replace(" / an", "")
                                    .replace("/an selon profil", "").replace(" €", "")
                                    .replace('.000,00', '000.00')
                                    if type(x) == str else np.nan)

df[['salaire_min', 'salaire_max']] = df.salaire.str.split(" - ", expand=True)

# on convertit en float
df['salaire_min'] = df['salaire_min'].apply(lambda x: float(x))
df['salaire_max'] = df['salaire_max'].apply(lambda x: float(x))

# colonnes entreprise et type de contrat

# séparation de la liste dans la colonne "Type de poste" en plusieurs colonnes
# identification des colonnes utiles
split_df = pd.DataFrame(df['Type de poste'].tolist(), columns=[
                        'col1', 'col2', 'entreprise', 'type_poste_1', 'type_client', 'type_poste_2', 'col7', 'type_contrat'])
split_df.isnull().sum()

# supression des caractères inutiles comme \n
# on saute les valeurs None qu'on remplace par Nan
split_df = split_df[['entreprise', 'type_client',
                     'type_poste_1', 'type_poste_2', 'type_contrat']]
split_df['type_poste_1'] = split_df['type_poste_1'].apply(
    lambda x: ''.join(map(str, x)).replace("\n", ""))
split_df['type_poste_2'] = split_df['type_poste_2'].apply(
    lambda x: ''.join(map(str, x)).replace("\n", "") if type(x) == str else np.nan)
split_df['type_contrat'] = split_df['type_contrat'].apply(
    lambda x: ''.join(map(str, x)).replace("\n", "") if type(x) == str else np.nan)
split_df['entreprise'] = split_df['entreprise'].apply(lambda x: ''.join(
    map(str, x)).replace("\n", "") if type(x) == str else np.nan)

# on prend la colonne type de poste la plus remplie et on complète les Nan à l'aide des autres colonnes
split_df['type_contrat'] = split_df['type_contrat'].fillna(
    split_df['type_poste_2'])
split_df['type_contrat'] = split_df['type_contrat'].fillna(
    split_df['type_poste_1'])

# on supprime les colonnes qui ont servi à remplir la colonne principale
split_df.drop(['type_poste_1', 'type_poste_2'], axis=1, inplace=True)

# on supprime la mention 'temps plein'
split_df['type_contrat'] = split_df['type_contrat'].str.replace(
    ' - temps plein', '')

# on s'est rendus compte que pour les index 110 et 121, il n'y avait pas de nom d'entreprise
# donc on remplace par Nan
split_df['entreprise'] = split_df['entreprise'].replace(
    r'^\s*$', np.nan, regex=True)
# split_df.isna().sum()

# on s'est rendus compte que les noms d'ENGIE et Société Générale était écrit de 2 manières différentes
# donc on homogénise
split_df['entreprise'] = split_df['entreprise'].str.replace(
    'Groupe ENGIE', 'ENGIE').replace('Societe Generale', 'Société générale')
# split_df['entreprise'].unique()

# on traite la colonne type_client
split_df['type_client'] = split_df['type_client'].replace(
    r'^\s*$', np.nan, regex=True)

# on rajoute au df principal
df = pd.concat([df, split_df], axis=1)

# et on supprime l'ancienne colonne type de poste
df.drop(['Type de poste'], axis=1, inplace=True)
# df


# extraction des caracteres necessaires dans la colonne date de publication
df['Date de publication'] = df['Date de publication'].apply(
    lambda x: x.replace("\n", "").replace("postée il y a", ""))
# df['Date de publication']
# extraction de la chaine de caractere apres le nombre de jour et mettre dans colonne "duree"
df["duree"] = df["Date de publication"].str.extract(r'(\d+)(\D+)')[1]
# print(df['duree'])

# extraction de la chaine de caractere numerique et mettre dans colonne "nb_jour"
df['nb_jour'] = df['Date de publication'].str.extract(
    r'(\d+)').fillna(1).astype(int)
# print(df['duree'])

# ------------------------------------------------------------------------------------------------------------------------------
# recuperer une valeur dans une colonne et appliquer une operation sur lautre en fonction de la valeur de la premiere colonne
# ------------------------------------------------------------------------------------------------------------------------------


# Transformer les valeurs de la colonne duree en liste
col_duree = df['duree'].tolist()

# Transformer les valeurs de la colonne nb_jour en liste
col_nb_jour = df['nb_jour'].tolist()

# Itérer simultanément sur les deux listes  avc la methode zip()
combined_list = list(zip(col_duree, col_nb_jour))

# creation d'une liste vide
modified_combined_list = []

# iterartion sur les element combiner de la list
for element in combined_list:
    if element[0] == ' mois':
        modified_combined_list.append(('mois', element[1] * 31))
    elif element[0] == ' heures':
        modified_combined_list.append(('heures', 0))
    else:
        modified_combined_list.append(element)

# mise du resultat dans liste de tuples
combined_list = modified_combined_list

# extraire la deuxieme partie des tuples car zip() retourne une liste de tuple
col_duree_final = [element[1] for element in combined_list]

# df['nb_jour_final'] = pd.DataFrame(modified_combined_list[1])
df['nb_jour_final'] = pd.DataFrame(col_duree_final)


# extraire la valeur de chaque ligne de nb_jour_final et la mettre dans parametre days="" de time delta
def process_date_list(days_list, reference_date):

    new_date_list = []
    # iteartion sur la liste de jours que l'on passe en parametre
    for days in days_list:
        # date  obtenu apres la soustraction du nombre de jour a la date de ref
        new_date = reference_date - timedelta(days=days)
        # mise au format souhaité de la date
        new_date = new_date.strftime("%Y/%m/%d")
        # integration de la date dans une nouvelle liste
        new_date_list.append(new_date)
    return new_date_list


# parametre a donner a la fonction "le dataframe qui contient tout les jours"
days_list = df['nb_jour_final']


# date de reference
datetime_str = '15/01/2023'
reference_date = datetime.strptime(datetime_str, '%d/%m/%Y')

# fonction jouée avec la liste des jours et la date de reference
result = process_date_list(days_list, reference_date)

# creation nouvelle colonne date
df["date"] = pd.DataFrame(result)


df = df[['Intitulé du poste', 'date', 'competences', 'entreprise',
         'type_client', 'type_contrat', 'lieu', 'salaire_min', 'salaire_max']]
# df

# -------------------------------------------------------------------------------------------------
# NETTOYAGE DE LA COLONNE INTITULE DU POSTE
# -------------------------------------------------------------------------------------------------
# Suppression des caracteres "\n"
df['Intitulé du poste'] = df['Intitulé du poste'].apply(
    lambda x: ''.join(map(str, x)).replace("\n", ""))
# df['Intitulé du poste']
# Appliquer la fonction `str.lower()` sur chaque ligne
df['Intitulé du poste'] = df['Intitulé du poste'].apply(lambda x: x.lower()
                                                        .replace('(h/f)', '')
                                                        .replace("h/f", "")
                                                        .replace("f/h", "")
                                                        .replace("-", "")
                                                        .replace("/", " - ")
                                                        .replace("#3", "")
                                                        .replace("data ingenieur", "data engineer")
                                                        .replace("alternant", "alternance")
                                                        .replace("datascientist", "data scientist")
                                                        )

# -------------------------------------------------------------------------------------------------
# NETTOYAGE DES INTITULES EN GENERALISANT LES METIER RECHERCHES SANS ETRE PRECIS
# -------------------------------------------------------------------------------------------------

job_list = ['data analyst', 'data scientist', 'business analyst', 'data engineer',
            'senior data engineer', 'data quality manager', 'stage', 'alternance']


def replace_job_with_job_list(df, column, job_list):
    for i, job in enumerate(job_list):
        df.loc[df[column].str.contains(job, na=False), column] = job_list[i]
    return df


replace_job_with_job_list(df, 'Intitulé du poste', job_list)

# -------------------------------------------------------------------------------------------------
# SUPPRESSION DES STOPS WORDS
# -------------------------------------------------------------------------------------------------

# import nltk
# from nltk.corpus import stopwords
# stop = stopwords.words(['french','english'])


# df['Intitulé du poste'] = df['Intitulé du poste'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

# df


# recherche de mot avec un tiret
pattern = "(?u)\\b[\\w-]+\\b"
vectorizer = CountVectorizer(token_pattern=pattern)

tokens = vectorizer.fit_transform(df['competences'])
vectorizer.get_feature_names_out()
tokens.toarray()

# créer un nouveau DataFrame à partir de X.toarray()
df_tokens = pd.DataFrame(
    tokens.toarray(), columns=vectorizer.get_feature_names_out())

# Ajouter les colonnes extraites à un dataframe existant
df_full = pd.concat([df, df_tokens], axis=1)

# df_full


# compétences les plus recherchées
dfcomp = df_tokens.sum(axis=0).sort_values(ascending=False).head(20)
dfcomp

dfcomp = pd.DataFrame(dfcomp)

# définir les séries de données
x = dfcomp[0].index
y = dfcomp[0].values

# créer un objet figure et un objet Axes
fig, ax = plt.subplots()

# utiliser la méthode bar pour tracer le graphique
ax.bar(x, y)

# ajouter un titre et des étiquettes d'axe
ax.set_title("Liste des compétences les plus recherchées")
ax.set_xlabel("compétences")
ax.set_ylabel("nombre d'occurences")

# faire pivoter les étiquettes d'axe X de 45°
for tick in ax.get_xticklabels():
    tick.set_rotation(45)

plt.rcParams["figure.figsize"] = (15, 10)
# afficher le résultat
plt.show()

# entreprises qui recrutent le plus
dfent = df_full['entreprise'].value_counts(
).sort_values(ascending=False).head(10)
dfent


dfent = pd.DataFrame(dfent)


# définir les séries de données
x = dfent['entreprise'].index
y = dfent['entreprise'].values

# créer un objet figure et un objet Axes
fig, ax = plt.subplots()

# utiliser la méthode bar pour tracer le graphique
ax.bar(x, y)

# ajouter un titre et des étiquettes d'axe
ax.set_title("entreprises qui recrutent le plus")
ax.set_xlabel("entreprises")
ax.set_ylabel("nombre d'occurences")

# faire pivoter les étiquettes d'axe X de 45°
for tick in ax.get_xticklabels():
    tick.set_rotation(45)

plt.rcParams["figure.figsize"] = (15, 10)
# afficher le résultat
plt.show()

# types de contrat les plus fréquents
dfcontrat = df_full['type_contrat'].value_counts().sort_values(ascending=False)
dfcontrat

dfcontrat = pd.DataFrame(dfcontrat)

# définir les séries de données
x = dfcontrat['type_contrat'].index
y = dfcontrat['type_contrat'].values

# créer un objet figure et un objet Axes
fig, ax = plt.subplots()

# utiliser la méthode bar pour tracer le graphique
ax.bar(x, y)

# ajouter un titre et des étiquettes d'axe
ax.set_title("Types de contrat les plus fréquents")
ax.set_xlabel("types de contrat")
ax.set_ylabel("nombre d'occurences")

# faire pivoter les étiquettes d'axe X de 45°
for tick in ax.get_xticklabels():
    tick.set_rotation(45)

# afficher le résultat
plt.show()


# types de client les plus fréquents
dfclient = df_full['type_client'].value_counts().sort_values(ascending=False)
dfclient


# où sont situés la plupart des postes
dflieu = df_full['lieu'].value_counts().sort_values(ascending=False)
dflieu.head(15)


dfclient = pd.DataFrame(dfclient)

# définir les séries de données
x = dfclient['type_client'].index
y = dfclient['type_client'].values

# créer un objet figure et un objet Axes
fig, ax = plt.subplots()

# utiliser la méthode bar pour tracer le graphique
ax.bar(x, y)

# ajouter un titre et des étiquettes d'axe
ax.set_title("Types de clients les plus fréquents")
ax.set_xlabel("types de client")
ax.set_ylabel("nombre d'occurences")

# faire pivoter les étiquettes d'axe X de 45°
for tick in ax.get_xticklabels():
    tick.set_rotation(45)

# afficher le résultat
plt.show()


# quelle entreprise recrute pour quel poste et quel salaire
df_table = df.drop(['type_client'], axis=1)
df_table.dropna(inplace=True)
df_table.dropna(inplace=True)
table = pd.pivot_table(df_table, index=['entreprise', 'Intitulé du poste'])
table

dfnan = df_full.dropna()

# types de clients qui rémunèrent le mieux
df_client = dfnan.groupby(['type_client']).median()
df_client['nombre_observations'] = df_full['type_client'].value_counts()
df_client[['salaire_min', 'salaire_max', 'nombre_observations']
          ].sort_values('salaire_min', ascending=False)
# données présentes uniquements pour 3 types de client

plt.figure(figsize=(13, 4))
plt.subplot(1, 2, 1)
meanprops = {'marker': 'o', 'markeredgecolor': 'black',
             'markerfacecolor': 'firebrick'}
sns.boxplot('type_client', 'salaire_min', data=dfnan, showmeans=True,
            meanprops=meanprops).set(title='Salaires min par type de client')
plt.subplot(1, 2, 2)
sns.boxplot('type_client', 'salaire_max', data=dfnan, showmeans=True,
            meanprops=meanprops).set(title='Salaires max par type de client')
plt.show()

# postes les mieux rémunérés
df_poste = df_full.dropna().groupby(['Intitulé du poste']).median()
df_poste['nombre_observations'] = df_full['Intitulé du poste'].value_counts()
postes_mieux_remuneres = df_poste[['salaire_min', 'salaire_max', 'nombre_observations']
                                  ].loc[df_poste['nombre_observations'] >= 2].sort_values('salaire_min', ascending=False)
postes_mieux_remuneres
# besoin de généraliser davantage les intitulés de poste

# make data
np.random.seed(3)
x = postes_mieux_remuneres.index.to_numpy()


ya = postes_mieux_remuneres[['salaire_max']].to_numpy()
yb = postes_mieux_remuneres[['salaire_min']].to_numpy()

# plot
fig, ax = plt.subplots()

# plt.stem(x, ya, 'b', markerfmt='o', label='salaire max')
# plt.stem(x, yb, 'g', markerfmt='o', label='salaire min')

markerline, stemlines, baseline = plt.stem(
    x, ya, markerfmt='o', label='salaire max')
plt.setp(stemlines, 'color', plt.getp(markerline, 'color'))
plt.setp(stemlines, 'linestyle', 'solid')

markerline, stemlines, baseline = plt.stem(
    x, yb, markerfmt='go', label='salaire min')
plt.setp(stemlines, 'color', plt.getp(markerline, 'color'))
plt.setp(stemlines, 'linestyle', 'solid')
ax.set_title("Ecart entre salaire min et salaire max médians par poste")
plt.xticks(rotation=45)

plt.legend()
plt.show()

# compétences recherchées pour les postes les mieux rémunérés
df_poste_comp = df[~df['Intitulé du poste'].isin(
    postes_mieux_remuneres.columns)]
df_poste_comp = df_full.dropna().groupby(['Intitulé du poste']).sum()
df_poste_comp.drop(['salaire_min', 'salaire_max'], axis=1, inplace=True)

df_poste_comp['nombre_competences'] = df_poste_comp.astype(bool).sum(axis=1)
df_poste_comp['nombre_observations'] = df_full['Intitulé du poste'].value_counts()

df_poste_comp = df_poste_comp.sort_values(
    "nombre_observations", ascending=False)
df_poste_comp = df_poste_comp.drop(df_poste_comp.index[5:])
df_poste_comp = df_poste_comp.loc[:, (df_poste_comp != 0).any(axis=0)]

df_poste_comp  # .style.background_gradient(cmap='Greens')


df_poste_comp.drop(
    ['nombre_competences', 'nombre_observations'], axis=1, inplace=True)
fig, ax = plt.subplots(figsize=(30, 5))
sns.heatmap(df_poste_comp, annot=True)

# compétences les plus demandées pour les salaires les plus élevés

df_salaire_comp = df_full.dropna().groupby(['salaire_max']).sum()
df_salaire_comp.drop(['salaire_min'], axis=1, inplace=True)
# comme il y a beaucoup de colonnes compétences, on cherche à savoir combien de compétences sont présentes dans chaque ligne
# attention en cas de disparité entre les annonces, cette valeur peut être fortement influencée par le nombres d'observations
df_salaire_comp['nombre_competences'] = df_salaire_comp.astype(
    bool).sum(axis=1)
df_salaire_comp['nombre_observations'] = df_full['salaire_max'].value_counts()
df_salaire_comp = df_salaire_comp.loc[:, (df_salaire_comp != 0).any(
    axis=0)].sort_values('salaire_max', ascending=False)
# .style.background_gradient(cmap='Blues')
df_salaire_comp.loc[df_salaire_comp['nombre_observations'] >= 2]


df_salaire_comp.drop(
    ['nombre_competences', 'nombre_observations'], axis=1, inplace=True)
fig, ax = plt.subplots(figsize=(30, 5))
sns.heatmap(df_salaire_comp, annot=True)

# lien entre compétences et type de contrat
df_contrat = df_full.dropna().groupby(['type_contrat']).sum()
df_contrat.drop(['salaire_min', 'salaire_max'], axis=1, inplace=True)
# comme il y a beaucoup de colonnes compétences, on cherche à savoir combien de compétences sont présentes dans chaque ligne
# attention en cas de disparité entre les annonces, cette valeur peut être fortement influencée par le nombres d'observations
df_contrat['nombre_competences'] = df_contrat.astype(bool).sum(axis=1)
df_contrat['nombre_observations'] = df_full['type_contrat'].value_counts()

# .sort_values('salaire_max', ascending=False)
df_contrat = df_contrat.loc[:, (df_contrat != 0).any(axis=0)]
# .style.background_gradient(cmap='Reds')
df_contrat.loc[df_contrat['nombre_observations'] >= 2]


df_contrat.drop(['nombre_competences', 'nombre_observations'],
                axis=1, inplace=True)
fig, ax = plt.subplots(figsize=(40, 2))
sns.heatmap(df_contrat, annot=True)

# compétences les plus recherchées par entreprise
df_salaire_entreprise = df_full.dropna().groupby(['entreprise']).sum()
df_salaire_entreprise.drop(
    ['salaire_min', 'salaire_max'], axis=1, inplace=True)
# comme il y a beaucoup de colonnes compétences, on cherche à savoir combien de compétences sont présentes dans chaque ligne
# attention en cas de disparité entre les annonces, cette valeur peut être fortement influencée par le nombres d'observations
df_salaire_entreprise['nombre_competences'] = df_salaire_entreprise.astype(
    bool).sum(axis=1)
df_salaire_entreprise['nombre_observations'] = df_full['entreprise'].value_counts()
df_salaire_entreprise = df_salaire_entreprise.loc[:, (df_salaire_entreprise != 0).any(
    axis=0)]  # .sort_values('salaire_max', ascending=False)
# .style.background_gradient(cmap='Purples')
df_salaire_entreprise.loc[df_salaire_entreprise['nombre_observations'] >= 2]


df_salaire_entreprise.drop(
    ['nombre_competences', 'nombre_observations'], axis=1, inplace=True)
fig, ax = plt.subplots(figsize=(30, 5))
sns.heatmap(df_salaire_entreprise, annot=True)


# Preprocessing


# Pipeline and model


df_model = df.drop(['type_client'], axis=1).dropna()

y = df_model[["salaire_min", "salaire_max"]]
# y = df_model["salaire_min"]
X = df_model.drop(["salaire_min", "salaire_max", "date"], axis=1)

df_model.shape


# create the pipeline
pipeline = Pipeline([
    ("regressor", TransformedTargetRegressor(regressor=RandomForestRegressor()))
])


column_cat = ['Intitulé du poste', 'entreprise', 'type_contrat', 'lieu']
transfo_cat = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Transformation of textual variables
column_comp = 'competences'
transfo_comp = Pipeline(
    steps=[
        ('bow', CountVectorizer())
    ])


# Class ColumnTransformer : apply alls steps on the whole dataset
preparation = ColumnTransformer(
    transformers=[
        ('data_cat', transfo_cat, column_cat),
        ('data_comp', transfo_comp, column_comp)

    ])


model = RandomForestRegressor()

multi_reg = MultiOutputRegressor(model)

pipe_model = Pipeline(steps=[('preparation', preparation),
                             ('model', multi_reg)
                             ])
pipe_model

set_config(display='diagram')
pipe_model


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=10)

# pipe_model = MultiOutputRegressor(model)

pipe_model.fit(X_train, y_train)

y_pred = pipe_model.predict(X_test)

score = r2_score(y_test, y_pred)
score


new_data = {
    'Intitulé du poste': 'data analyst',
    'competences': 'python, sql, agile, cloud',
    'entreprise': '',
    'type_contrat': 'CDD',
    'lieu': 'paris'}

df_pred = pd.DataFrame(new_data, index=[0])
df_pred


pred_salaire = pipe_model.predict(df_pred)
pred_salaire[0]


joblib.dump(pipe_model, 'random_forest_regressor')
