# -*- coding: utf-8 -*-

import pickle
import os
import textwrap

import streamlit as st
import numpy as np
import pandas as pd
import requests
import base64
import shap
import plotly.graph_objects as go
import plotly.express as px

###Initialisation###
loanColumn = 'SK_ID_CURR'
target = 'TARGET'
colorLabel = 'Demande de prêt:'
colW = 350
colH = 500
#url="https://flask-oc-p7-test-3.herokuapp.com/"
#url="https://heroku-test-fay1.herokuapp.com/"
url="https://heroku-test-fay2.herokuapp.com/"
#url = "C:/Users/archi/PycharmProjects/Flask/venv/"
minRating = -1
maxRating = 1
localThreshold = 0


###Les requètes API###
def convToB64(data):
    '''
    Cette fonction permet d'avoir:
    En entrée : les données de n'importe quel type.
    La fonction convertit les données en base-64 puis la chaîne résultante est encodée en UTF-8.
    Sortie : le résultat obtenu.
    '''
    return base64.b64encode(pickle.dumps(data)).decode('utf-8')


def restoreFromB64Str(data_b64_str):
    '''
    Cette fonction permet d'avoir:
    Entrée : les données sont converties en Base-64 puis encodées en UTF-8.
          Idéalement, les données de la fonction convToB64.
    La fonction restaure les données encodées dans leur format d'origine.
    Sortie : les données restaurées
    '''
    return pickle.loads(base64.b64decode(data_b64_str.encode()))


def askAPI(apiName, url=url, params=None):
    '''
    Cette fonction permet d'interroger une API.
    Elle a été conçu pour interroger une API s'exécutant sur un serveur FLASK.
    Les données reçues de l'API doivent être au format base-64 encodé en UFT-8.
    Saisir:
        <apiName> : Nom de l'API à interroger, c'est le nom trouvé à la fin
                   de l'URL avant la liste des paramètres.
        <url> : url où est stockée l'API à interroger.
        <params> : à 'None' par défaut si aucun paramètre à envoyer.
                  Les paramètres envoyés doivent être au format dictionnaire {'Parameter Name' : 'Parameter Value'}
    Sortie : la réponse fournie par l'API, décodée via la fonction restoreFromB64Str.
    '''
    url = url + str(apiName)
    resp = requests.post(url=url, params=params).text
    return restoreFromB64Str(resp)


def splitAndAskAPI(data):
    '''
    Cette fonction permet de contourner une limitation du serveur Heroku lors de l'utilisation du package gratuit
    il peut être nécessaire de fractionner les données à envoyer à l'API afin que les données
    puissent être reconstituées côté serveur.
    Pour être utilisée, l'API distante doit être conçue pour gérer ce type de données.
    '''
    ###Fractionner et envoyer les données à l'API###
    print('splitAndAskAPI')
    print(askAPI(apiName='initSplit'))
    if askAPI(apiName='initSplit'):
        print('OK')
        for j, i in enumerate(splitString(data, 5)):
            #         time.sleep(1)
            print(f'len du split n°{j}: {len(i)}')
            if askAPI(apiName='merge', params=dict(txtSplit=i, numSplit=j)):
                print('OK... pour le moment...')
            else:
                print('Aie Aie Aie...')
        resp = askAPI(apiName='endSplit')
        print(resp)
        return resp


@st.cache(suppress_st_warning=True)
def apiModelPrediction(data, loanNumber, columnName='SK_ID_CURR', url=url):
    '''
    Cette fonction permet de faire une prédiction en interrogeant le modèle distant via une API.
    La fonction prend en charge la limitation de Heroku sur la taille des données pouvant être envoyées dans une requête 'POST'.
    En entrée :
        <data>: les données au format dataframe de pandas compatibles avec le modèle interrogé.
        <loanNumber>: le numéro de prêt du client à interroger
        <columnName>: Le nom de la colonne contenant les numéros de prêt
        <url>: URL de l'API à interroger'
    En sortie : La prédiction du modèle selon le format (Prédiction exacte (0 ou 1), prédiction probabiliste [0;1])
    '''
    print('apiModelPrediction')
    # Préparation de l'information qui sera envoyée
    # Récupération de l'index
    idx = getTheIDX(data, loanNumber, columnName)
    print(f'idx={idx}')
    #Création d'une liste avec Pandas contenant les informations des clients
    dataOneCustomer = data.iloc[[idx]].values
    #Encodage des données en base 64 puis en charactères sous format utf-8
    dataOneCustomerB64Txt = convToB64(dataOneCustomer)
    #Les données sont séparées en cinq partie dù au la limitation de volume des données par Heroku
    dictResp = splitAndAskAPI(data=dataOneCustomerB64Txt)

    return dictResp['predExact'], dictResp['predProba']


###Importer les données###
@st.cache(suppress_st_warning=True)
def loadData():
    '''
    Cette fonction retourne les données au format pickle qui sont contenues dans le dossier "pickle"
    Les données retournées sont :
        'dataRef.pkl': contient les données de la base client qui a servi à former
        le modèle et qui servira à la réalisation des différents graphismes du tableau de bord.
        dataCustomer': contient une liste de clients qui peuvent être interrogés pour connaître
        si leur demande de prêt est accordée ou non.
    '''
    return pickle.load(open(os.getcwd() + '/pickle/dataRef.pkl', 'rb')), pickle.load(open(os.getcwd() + '/pickle/dataCustomer.pkl', 'rb'))
#    return pickle.load(open(os.path.dirname(__file__) + '/pickle/dataRef.pkl', 'rb')), pickle.load(
#        open(os.path.dirname(__file__) + '/pickle/dataCustomer.pkl', 'rb'))

@st.cache(suppress_st_warning=True)
def loadModel(modelName='model'):
    '''
    Cette fonction interroge et retourne le modèle stocké sur le serveur distant.
    '''
    return askAPI(apiName=modelName)


@st.cache(suppress_st_warning=True)
def loadThreshold():
    '''
    Cette fonction interroge et renvoie la valeur
    seuil (issue du modèle LGBMClassifier) stockée sur le serveur distant.
    '''
    return askAPI(apiName='threshold')


@st.cache(suppress_st_warning=True)
def loadRatingSystem():
    '''
    Cette fonction interroge et renvoie la valeur
    du système de notation (issue du modèle LGBMClassifier) stockée sur le serveur distant.
    En entrée : rien
    En sortie : score minimum, score maximum, valeur seuil.
    '''
    return askAPI(apiName='ratingSystem')


###Obtention des données###
@st.cache(suppress_st_warning=True)
def getDFLocalFeaturesImportance(model, X, loanNumber, nbFeatures=12, inv=False):
    '''
    Cette fonction retourne un dataframe Pandas qui est utilisée
    pour réaliser des graphes de 'features importance' avec la bibliothèque 'SHAP'.
    Cela permet de réaliser un graphe en utilisant une autre librairie graphique
    que celle utilisée par défaut avec 'SHAP'.
    '''
    idx = getTheIDX(data=X, columnName=loanColumn, value=loanNumber)
    shap_values = shap.TreeExplainer(model).shap_values(X.iloc[[idx]])[0]

    if inv:
        shap_values *= -1

    dfShap = pd.DataFrame(shap_values, columns=X.columns.values)
    serieSignPositive = dfShap.iloc[0, :].apply(lambda col: True if col >= 0 else False)

    serieValues = dfShap.iloc[0, :]
    serieAbsValues = abs(serieValues)
    return pd.DataFrame(
        {
            'values': serieValues,
            'absValues': serieAbsValues,
            'positive': serieSignPositive,
            'color': map(lambda x: 'red' if x else 'blue', serieSignPositive)
        }
    ).sort_values(
        by='absValues',
        ascending=False
    ).iloc[:nbFeatures, :].drop('absValues', axis=1)


def getTheIDX(data, value, columnName='SK_ID_CURR'):
    '''
    Renvoie l'indice correspondant à la 1ère valeur contenue
    dans la valeur contenue dans la colonne 'columnName' du Dataframe.
    '''
    return data[data[columnName] == value].index[0]


def splitString(t, nbSplit):
    '''
    Cette fonction divise une chaîne en morceaux <nbSplit>.
    Si possible, tous les morceaux de texte ont le même nombre de caractères.
    Renvoie la chaîne fractionnée au format liste.
    '''
    import textwrap
    from math import ceil
    return textwrap.wrap(t, ceil(len(t) / nbSplit))


@st.cache(suppress_st_warning=True)
def get_df_global_shap_importance(model, X):
    '''
    Cette fonction renvoie un datatrame utilisé pour créer le graphique de 'features importance' globales.
    Cela permet de recréer le graphe initialement fourni par SHAP avec une autre librairie graphique.
    '''
    #Explication des prédictions du modèle à l'aide de la bibliothèque shap
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)[0]
    return pd.DataFrame(
        zip(
            X.columns[np.argsort(np.abs(shap_values).mean(0))][::-1],
            np.sort(np.abs(shap_values).mean(0))[::-1]
        ),
        columns=['feature', 'importance']
    )


@st.cache(suppress_st_warning=True)
def convertUpperAndLowerBoundAndThreshoold(value, oldMin, oldMax, oldThreshold, newMin, newMax, newThreshold):
    '''
    Convertir une note avec définition d'une limite inférieure et d'une limite supérieure en une autre note
    qui peuvent avoir des limites inférieures et supérieures différentes.
    Il est possible de définir des seuils qui définissent la valeur limite de
    ce qui pourrait être approuvé/non approuvé.
    Par exemple:
    Un modèle renvoie un score compris entre 0 et 1 avec un seuil à 0,8.
    En dessous de 0,8, le résultat est non approuvé, au-dessus de 0,8, le résultat est approuvé.
    On veut afficher un score compris entre -1 et 1 avec un seuil à 0,
    indépendamment du seuil défini auparavant pour des soucis de lisibilité
    si le score était par exemple présenté à un client qui
    ne connaisse pas cette notion de seuil et préféreraient, intuitivement imaginer
    un score strictement négatif comme non approuvé et un score positif comme approuvé.
    En entrée :
    - value : le score à convertir,
    - oldMin : limite inférieure dans le système de notation du score à convertir;
    - oldMax : limite supérieur dans le système de notation du score à convertir;
    - oldThreshold : seuil dans le système de notation du score à convertir;
    - newMin : limite inférieure dans le système de notation du score qui sera converti;
    - newMax : limite supérieure dans le système de notation du score converti;
    - newThreshold : seuil dans le système de notation du score converti.
    En sortie :
    - le score dans le nouveau système de notation.
    '''

    if value < oldThreshold:
        oldMax = oldThreshold
        newMax = newThreshold
    else:
        oldMin = oldThreshold
        newMin = newThreshold

    return ((value - oldMin) * ((newMax - newMin) / (oldMax - oldMin))) + newMin


###Graphes###
@st.cache(suppress_st_warning=True)
def gauge_chart(score, minScore, maxScore, threshold):
    '''
    Renvoie une jauge à chiffres avec un certain nombre de paramètres prédéfinis.
    Il suffit de renseigner le <score> ainsi que le <threshold>.
    '''

    convertedScore = convertUpperAndLowerBoundAndThreshoold(value=score,
                                                            oldMin=minScore,
                                                            oldMax=maxScore,
                                                            oldThreshold=threshold,
                                                            newMin=minRating,
                                                            newMax=maxRating,
                                                            newThreshold=localThreshold)

    color = "RebeccaPurple"
    if convertedScore < localThreshold:
        color = "darkred"
    else:
        color = "green"
    fig = go.Figure(
        go.Indicator(
            domain={
                'x': [0, 0.9],
                'y': [0, 0.9]
            },
            value=convertedScore,
            mode="gauge+number+delta",
            title={
                'text': "Score"
            },
            gauge={
                'axis': {
                    'range': [-1, 1]
                },
                'bar': {
                    'color': color
                },
                'steps': [
                    {
                        'range': [-1, -0.8],
                        'color': "#ff0000"
                    },
                    {
                        'range': [-0.8, -0.6],
                        'color': "#ff4d00"
                    },
                    {
                        'range': [-0.6, -0.4],
                        'color': "#ff7400"
                    },
                    {
                        'range': [-0.4, -0.2],
                        'color': "#ff9a00"
                    },
                    {
                        'range': [-0.2, 0],
                        'color': "#ffc100"
                    },
                    {
                        'range': [0, 0.2],
                        'color': "#c5ff89"
                    },
                    {
                        'range': [0.2, 0.4],
                        'color': "#b4ff66"
                    },
                    {
                        'range': [0.4, 0.6],
                        'color': "#a3ff42"
                    },
                    {
                        'range': [0.6, 0.8],
                        'color': "#91ff1e"
                    },
                    {
                        'range': [0.8, 1],
                        'color': "#80f900"
                    }
                ],
                'threshold': {
                    'line': {
                        'color': color,
                        'width': 8
                    },
                    'thickness': 0.75,
                    'value': convertedScore
                }
            },
            delta={'reference': 0.5, 'increasing': {'color': "RebeccaPurple"}}
        ))
    return fig


@st.cache(suppress_st_warning=True)
def plotGlobalFeaturesImportance(model, X, nbFeatures=10):
    '''
    Retourne un chiffre permettant l'affichage de la feature importance globale.
    Le calcul est fait par la librairie 'SHAP' et l'affichage du graphique avec 'plotly'.
    '''
    #Suppression de la colonne <target> si elle existe.
    X = X.drop(target, axis=1, errors='ignore')

    data = get_df_global_shap_importance(model, X)
    y = data.head(nbFeatures)['importance']
    x = data.head(nbFeatures)['feature']

    fig = go.Figure(
        data=[go.Bar(x=x, y=y, marker=dict(color=y, colorscale='viridis'))],
        layout=go.Layout(
            title=go.layout.Title(text="Features importance Globales:")
        )
    )

    return fig


@st.cache(suppress_st_warning=True)
def plotLocalFeaturesImportance(model, X, loanNumber, nbFeatures=12):
    '''
    Retourne une valeur permettant l'affichage de la 'feature importance' locale.
    Le calcul est fait par la librairie 'SHAP' et l'affichage du graphique avec la librairie 'plotly'.
    '''
    dfValuesSign = getDFLocalFeaturesImportance(
        model=model,
        X=X,
        loanNumber=loanNumber,
        nbFeatures=nbFeatures,
        inv=False
    )
    i = dfValuesSign.index
    fig = px.bar(dfValuesSign,
                 x='values',
                 y=i,
                 color='color',
                 orientation='h',
                 category_orders=dict(index=list(i)))
    fig.update_layout(
        title="Features importance du client",
        yaxis={'title': None},
        xaxis={'title': None},
        showlegend=False
    )
    return fig


def adaptTargetValuesAndTitle(data):
    data = data.copy()
    data[target] = data[target].map({0: 'Accepté', 1: 'Refusé'})
    return data.rename(columns={target: 'Demande de prêt:'})


@st.cache(suppress_st_warning=True)
def plotDistOneFeature(dataRef, feature, valCust):
    dataRef = adaptTargetValuesAndTitle(dataRef)
    fig = px.histogram(dataRef,
                       x=feature,
                       color=colorLabel,
                       marginal="box",
                       histnorm='probability')  #Peut prendre plusieurs forme tel que, box, violin...
    fig.add_vline(x=valCust, line_width=3, line_dash="dash", line_color="red")
    return fig


@st.cache(suppress_st_warning=True)
def plotScatter2D(dataRef, listValCust):
    '''
    Cette fonction retourne un chiffre généré par plotly express.
    La figure est un nuage de points en deux dimensions représentant tous les clients.
    Seront également affichées deux lignes rouges (une verticale et l'autre horizontale)
    dont l'intersection représente la localisation du client observé.
    '''
    dataRef = adaptTargetValuesAndTitle(dataRef)

    fig = px.scatter(
        dataRef,
        x=listValCust[0][0],
        y=listValCust[1][0],
        color=colorLabel,
        color_discrete_sequence=px.colors.qualitative.Plotly
    )

    fig.add_vline(x=listValCust[0][1], line_width=1, line_dash="solid", line_color="red")
    fig.add_hline(y=listValCust[1][1], line_width=1, line_dash="solid", line_color="red")

    fig.update_layout(showlegend=True)

    return fig


@st.cache(suppress_st_warning=True)
def plotScatter3D(dataRef, listValCust):
    '''
    Cette fonction retourne une valeur générée par plotly express.
    La figure est un nuage de points en trois dimensions représentant tous les clients.
    La couleur des points est réalisée selon la caractéristique <feature>
    Le client observé est représenté par un point d'une couleur différente de tous les autres clients.
    '''

    dataRef = adaptTargetValuesAndTitle(dataRef)

    fig = px.scatter_3d(
        dataRef,
        x=listValCust[0][0],
        y=listValCust[1][0],
        z=listValCust[2][0],
        color=colorLabel,
        color_discrete_sequence=px.colors.qualitative.Plotly
    )
    fig.update_layout(showlegend=True)

    fig.add_scatter3d(
        x=[listValCust[0][1]],
        y=[listValCust[1][1]],
        z=[listValCust[2][1]],
        name='Client'
    )
    return fig