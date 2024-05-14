import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


dados = pd.read_csv("Aprendizado_de_Maquina/Entrega-01/dados_usuarios.csv", sep=",")

X = dados.drop(columns=["status_pedido", "data_pedido", "codigo_rastreio"])
y = dados["status_pedido"]

X = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#Escolha do algoritmo de classificação
modelo = RandomForestClassifier(random_state=42)

#Treino do modelo
modelo.fit(X_train, y_train)


#Avaliação da precisão, recall e F1-Score
previsoes = modelo.predict(X_test)


precisao = accuracy_score(y_test, previsoes)
print("Precisão do modelo:", precisao)


recall = recall_score(y_test, previsoes, average='macro')

print("Recall do modelo:", recall)


f1 = f1_score(y_test, previsoes, average='macro')

print("F1-score (macro):", f1)

