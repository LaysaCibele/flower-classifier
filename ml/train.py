import os
import joblib
import pandas as pd
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, log_loss, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

def train_and_compare_models():
    #carregamento dos dados
    iris = load_iris()
    X, y = iris.data, iris.target
    target_names =  list(iris.target_names)
    feature_names = ['sepal_lenght', 'sepal_width', 'petal_lenght', 'petal_width']
    
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=45, stratify=y
    )
    
    
    
    #definição dos modelos candidatos
    
    candidates = {
        "LogisticRegression": Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(random_state=42, max_iter=200)) 
        ]),
        "RandomForest": Pipeline([
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
        ]),
        "LightGBM":Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LGBMClassifier(random_state=42, verbose=-1))
        ])
    }
    
    results = []
    best_model_name = None
    best_f1 = -1.0
    best_pipeline = None

    print("\n="*50)
    print("COMPARATIVO DE ALGORITMOS DE CLASSIFICAÇÃO")

    #Treinar e avaliar cada um
    for name, pipeline in candidates.items():
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        loss = log_loss(y_test, y_proba)

        results.append({
            "Algoritmo": name,
            "Acurácia": f"{acc:.4f}",
            "F1-Score (Macro)": f"{f1:.4f}",
            "Log Loss": f"{loss:.4f}"
        })

        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            best_pipeline = pipeline

    #exibir tabela comparativa
    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False))
    print("="*50)
    print(
        f"Modelo Selecionado: {best_model_name} (F1-Score Macro: {best_f1:.4f})")
    print("="*50)

    #Relatório detalhado do vencedor
    y_pred_best = best_pipeline.predict(X_test)
    print(f"\nRelatório Detalhado ({best_model_name}):\n")
    print(classification_report(y_test, y_pred_best, target_names=target_names))

    # 5. Salvar o modelo vencedor e seus metadados
    output_dir = os.path.join(os.path.dirname(__file__), "artifacts")
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "iris_model.joblib")

    joblib.dump({
        'pipeline': best_pipeline,
        'model_name': best_model_name,
        'target_names': target_names,
        'feature_names': feature_names
    }, model_path)

    print(f"\nArtefato salvo em: {model_path}")


if __name__ == "__main__":
    train_and_compare_models()
