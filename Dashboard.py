# -*- coding: utf-8 -*-

import streamlit as st
import utils

def main():
    ###Initialisation###
    st.set_page_config(
        layout='wide',
        initial_sidebar_state="collapsed"
    )
    loanColumn = 'SK_ID_CURR'
    dataRef, dataCustomer = utils.loadData()
    model = utils.loadModel()
    print(model)
    minScore, maxScore, threshold = utils.loadRatingSystem()

    ###Top frame###
    col1, col2 = st.columns((1, 3))
    with col1:
        st.image('img/logo.png', width=300)
    with col2:
        st.title('Simulation pour un accord de prêt')
        st.header("Obtention d'une réponse instantanée")
        ###Entrée###
        user_input = st.selectbox('Veuillez renseigner le numéro de demande du prêt: ', dataCustomer[loanColumn].tolist())
        idxCustomer = utils.getTheIDX(dataCustomer, user_input, loanColumn)
        'Vous avez selectionné le prêt n°: ', user_input, ' correspondant au client numéro', idxCustomer
        ###Dataframe des features importances locales###
        df = utils.getDFLocalFeaturesImportance(model=model,
                                                X=dataCustomer,
                                                loanNumber=int(user_input),
                                                nbFeatures=12)

    ###Prediction du modèle API###
    predExact, predProba = utils.apiModelPrediction(data=dataCustomer, loanNumber=int(user_input))

    ###Validation du prêt###
    st.markdown("# Validation du prêt")
    loanResult = 'Status du prêt: '
    if predExact:
        loanResult += "Validé !"
        st.success(loanResult)
    else:
        loanResult += "Refusé..."
        st.error(loanResult)

    ###Centre###
    col1, col15, col2 = st.columns((2, 1, 2))
    with col1:
        ### Gauge Score
        fig = utils.gauge_chart(predProba, minScore, maxScore, threshold)
        st.write(fig)
    with col15:
        # Empty column to center the elements
        st.write("")
    with col2:
        ### Img OK/NOK
        if predExact:
            st.image('img/ok.png', width=400)
        else:
            st.image('img/NOT_OK.png', width=400)

    ###Features importance globales et locales###
    col1, col2 = st.columns((2))
    ###Features importance globales, col1/2###
    with col1:
        fig = utils.plotGlobalFeaturesImportance(model, dataRef, 10)
        st.write(fig)
    ###Features importance locales, col2/2###
    with col2:
        fig = utils.plotLocalFeaturesImportance(
            model=model,
            X=dataCustomer,
            loanNumber=int(user_input)
        )
        st.write(fig)

    ###Analyse mono et bidimensionnelle###
    ###Graphes illustrant la distribution###
    col1, col2 = st.columns((2))

    with col1:
        feature1 = st.selectbox('Choix de la première caratéristique: ', df.index, index=0)
        valueCustomer1 = dataCustomer.loc[dataCustomer[loanColumn] == user_input, feature1].values[0]
        fig = utils.plotDistOneFeature(dataRef, feature1, valueCustomer1)
        st.write(fig)

    with col2:
        feature2 = st.selectbox('Choix de la seconde caractéristique: ', df.index, index=1)
        valueCustomer2 = dataCustomer.loc[dataCustomer[loanColumn] == user_input, feature2].values[0]
        fig = utils.plotDistOneFeature(dataRef, feature2, valueCustomer2)
        st.write(fig)

    ###Scatter plot###
    col1, col2 = st.columns(2)
    ###Scatter plot 2D###
    with col1:
        listValueCustomer = [[feature1, valueCustomer1], [feature2, valueCustomer2]]
        fig = utils.plotScatter2D(dataRef, listValueCustomer)
        st.markdown('### ↓ Positionnement du client en fonction des deux premières caractéristiques séléctionnées')
        st.markdown(
            '### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Positionnement du client en fonction des trois caractéristiques sélectionnées par le client 🡮')
        st.write(fig)
    ###Scatter plot 3D###
    with col2:
        feature3 = st.selectbox('Choix de la troisième caractéristique:', df.index, index=2)
        listValueCustomer.append(
            [feature3, dataCustomer.loc[dataCustomer[loanColumn] == user_input, feature3].values[0]])
        fig = utils.plotScatter3D(dataRef, listValueCustomer)
        st.write(fig)


if __name__ == "__main__":
    main()