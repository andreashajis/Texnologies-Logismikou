import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from umap import UMAP
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# Page configuration
st.set_page_config(layout="wide")

# File uploader
st.sidebar.title("Upload Data")
uploaded_file = st.sidebar.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])

# Read data
if uploaded_file is not None:
    if uploaded_file.name.endswith("csv"):
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_excel(uploaded_file)
    
    st.write("Data Overview")
    st.dataframe(data.head())

    # Tabs
    tabs = st.sidebar.radio("Select Tab", ["Visualization", "Feature Selection", "Classification", "Info"])

    if tabs == "Visualization":
        st.header("Visualization Tab")

        # Data visualization
        if st.checkbox("Show Pairplot"):
            fig = sns.pairplot(data, hue= data.columns[-1])
            st.pyplot(fig)

       # Perform PCA
        if st.checkbox("Perform PCA"):
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(data.iloc[:, :-1])
    
            # PCA results
            pca_df = pd.DataFrame(data=pca_result, columns=["PC1", "PC2"])
            pca_df[data.columns[-1]] = data.iloc[:, -1]  # Add the label column to PCA DataFrame
    
             # Figure and axes
            fig, ax = plt.subplots() 
            sns.scatterplot(x="PC1", y="PC2", hue=data.columns[-1], data=pca_df, ax=ax)
            st.pyplot(fig)

        # UMAP
        if st.checkbox("Perform UMAP"):
            umap = UMAP(n_components=2)
            umap_result = umap.fit_transform(data.iloc[:, :-1])

            # UMAP results
            umap_df = pd.DataFrame(data=umap_result, columns=["UMAP1", "UMAP2"])
            umap_df[data.columns[-1]] = data.iloc[:, -1]  # Add the label column to the UMAP DataFrame

            # Figure and axes
            fig, ax = plt.subplots()
            scatter_plot = sns.scatterplot(x="UMAP1", y="UMAP2", hue=data.columns[-1], data=umap_df, ax=ax)
            st.pyplot(fig)

    # Feature Selection Tab
    elif tabs == "Feature Selection":
        st.header("Feature Selection Tab")
        def impPlot(imp, name):
            figure = px.bar(imp,
                            x=imp.values,
                            y=imp.keys(), labels = {'x':'Importance Value', 'index':'Columns'},
                            text=np.round(imp.values, 2),
                            title=name + ' Feature Selection Plot',
                            width=1000, height=600)
            figure.update_layout({
                'plot_bgcolor': 'rgba(0, 0, 0, 0)',
                'paper_bgcolor': 'rgba(0, 0, 0, 0)',
            })
            st.plotly_chart(figure)

        # Random Forest
        def randomForest(x, y):
            model = RandomForestClassifier()
            model.fit(x, y)
            feat_importances = pd.Series(model.feature_importances_, index=x.columns).sort_values(ascending=False)
            st.subheader('Random Forest Classifier:')
            impPlot(feat_importances, 'Random Forest Classifier')
            #st.write(feat_importances)
            st.write('\n')

        # Extra Trees
        def extraTress(x, y):
            model = ExtraTreesClassifier()
            model.fit(x, y)
            feat_importances = pd.Series(model.feature_importances_, index=x.columns).sort_values(ascending=False)
            st.subheader('Extra Trees Classifier:')
            impPlot(feat_importances, 'Extra Trees Classifier')
            st.write('\n')
        
        x = data.iloc[:, :-1]  # Using all column except for the last column as X
        y = data.iloc[:, -1]  # Selecting the last column as Y
        randomForest(x, y)
        extraTress(x, y)

    # Classification Tab
    elif tabs == "Classification":
        st.header("Classification Tab")

        # Classification
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model_options = st.selectbox("Select Model", ["KNN", "Decision Tree"])

        # KNN
        if model_options == "KNN":
            k_input = st.text_input("K value for KNN", "3")
            try:
                k = int(k_input)
                if k < 1 or k > 20:
                    st.error("K value must be between 1 and 20")
                    k = None
                else:
                    model = KNeighborsClassifier(n_neighbors=k)
                    
                    # Fit the model
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    # Metrics
                    accuracy = accuracy_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred, average='weighted')
                    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1], multi_class='ovr')

                    st.write(f"Accuracy: {accuracy}")
                    st.write(f"F1 Score: {f1}")
                    st.write(f"ROC AUC: {roc_auc}")

            except ValueError:
                st.error("Please enter a valid integer for K value")
                model = None
        # Decision Tree
        else:  
            model = DecisionTreeClassifier()

            # Fit model
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1], multi_class='ovr')

            st.write(f"Accuracy: {accuracy}")
            st.write(f"F1 Score: {f1}")
            st.write(f"ROC AUC: {roc_auc}")
    # Info Tab
    elif tabs == "Info":
        st.header("Info Tab")
        st.write("This app provides functionalities for data visualization, feature selection, and classification.")
        st.write("Developed by: Antreas Chatzivasili")
        st.write("Tasks:")
        st.write("- Data Loading and Display")
        st.write("- Visualization with PCA and UMAP")
        st.write("- Feature Selection")
        st.write("- Classification with KNN and Decision Tree Classifier")
