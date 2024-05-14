import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

dados = pd.read_csv("Aprendizado_de_Maquina/Entrega-01/dados_usuarios.csv", sep=",")

X = dados.drop(columns=['status_pedido']) 
y = dados['status_pedido'] 

X = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

modelo = RandomForestClassifier()

# Definir os hiperparâmetros para busca
parametros = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(estimator=modelo, param_grid=parametros, cv=5, scoring='accuracy')

grid_search.fit(X_train, y_train)

print("Melhores parâmetros:", grid_search.best_params_)


# Definindo o modelo com os melhores hiperparâmetros encontrados
modelo = RandomForestClassifier(max_depth=None, min_samples_split=2, n_estimators=200)

# Realizando a validação cruzada
scores = cross_val_score(modelo, X, y, cv=5)

print("Scores de validação cruzada:", scores)
print("Acurácia média:", scores.mean())
