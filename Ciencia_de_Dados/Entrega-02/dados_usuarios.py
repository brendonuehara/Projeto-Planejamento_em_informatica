import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
from scipy import stats
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

dados = pd.read_csv("CSV/dados_usuarios.csv", sep=",""")

##Parte 01

#Regressão Linear
X = dados['idade']
y = dados['valor_pago']

X = sm.add_constant(X)

modelo = sm.OLS(y, X).fit()

print(modelo.summary())

plt.figure(figsize=(10, 6))
sns.regplot(x='idade', y='valor_pago', data=dados, line_kws={"color": "red"})
plt.title('Regressão Linear: Valor Pago vs. Idade')
plt.xlabel('Idade')
plt.ylabel('Valor Pago (R$)')
plt.show()

#Análise de Variância
modelo_anova = ols('valor_pago ~ C(tipo_pagamento)', data=dados).fit()

anova_tabela = sm.stats.anova_lm(modelo_anova, typ=2)

print(anova_tabela)

plt.figure(figsize=(10, 6))
sns.boxplot(x='tipo_pagamento', y='valor_pago', data=dados, palette='pastel')
plt.title('ANOVA: Valor Pago por Tipo de Pagamento')
plt.xlabel('Tipo de Pagamento')
plt.ylabel('Valor Pago (R$)')
plt.show()


## Parte 02 / Parte 03 / Parte 04 / Parte 05

#Regressão Linear

X = dados.drop(columns=['nome', 'status_pedido', 'data_pedido', 'codigo_rastreio', 'valor_pago'])
y = dados['valor_pago']

X = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

modelo_regressao = LinearRegression()
modelo_regressao.fit(X_train, y_train)

y_pred = modelo_regressao.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Coeficientes do Modelo:")
print(modelo_regressao.coef_)
print("\nIntercepto do Modelo:")
print(modelo_regressao.intercept_)
print("\nMean Squared Error (MSE):", mse)
print("R² Score:", r2)


#Classificação usando Árvore de Decisão

dados['status_entregue'] = np.where(dados['status_pedido'] == 'entregue', 1, 0)

dados = pd.get_dummies(dados, columns=['tipo_pagamento'], drop_first=True)

X = dados[['idade', 'valor_pago', 'tipo_pagamento_cartão de crédito', 'tipo_pagamento_pix']]
y = dados['status_entregue']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

modelo_arvore = DecisionTreeClassifier(random_state=42)
modelo_arvore.fit(X_train, y_train)

y_pred = modelo_arvore.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("Acurácia do Modelo:", accuracy)
print("\nRelatório de Classificação:")
print(class_report)
print("\nMatriz de Confusão:")
print(conf_matrix)

plt.figure(figsize=(20, 10))
plot_tree(modelo_arvore, feature_names=X.columns, class_names=['Não Entregue', 'Entregue'], filled=True)
plt.title("Árvore de Decisão")
plt.show()

#Parte 06

mse_regressao = mean_squared_error(y_test, y_pred)
r2_regressao = r2_score(y_test, y_pred)

accuracy_arvore = accuracy_score(y_test, y_pred)

if mse_regressao < accuracy_arvore:
    print('O modelo de regressão linear é mais eficaz.')
else:
    print('O modelo de classificação com árvore de decisão é mais eficaz.')
