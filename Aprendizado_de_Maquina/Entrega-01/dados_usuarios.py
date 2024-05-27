from scipy import stats
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Extraindo dados do arquivo csv
import pandas as pd

tabela = pd.read_csv("CSV/dados_usuarios.csv", sep=",")
print(tabela)

# Tratamentos dos dados

dados_limpos = tabela.dropna()

dados_limpos = tabela.dropna(axis=1)

z_scores = stats.zscore(tabela['valor_pago'])

outliers = tabela[(z_scores > 3) | (z_scores < -3)]

dados_inconsistentes = tabela[tabela['idade'] < 0]
dados_inconsistentes = tabela[tabela['valor_pago'] < 0]

tabela.loc[tabela['idade'] < 0, 'idade'] = 0
tabela.loc[tabela['valor_pago'] < 0, 'valor_pago'] = 0

tabela['valor_pago'] = tabela['valor_pago'].apply(lambda x: x * 1000 if x < 1000 else x)

# Matriz Confusão

y_true = [1, 0, 1, 1, 0, 1]

y_pred = [1, 0, 1, 0, 0, 1]

matriz_confusao = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(matriz_confusao, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Rótulos Previstos')
plt.ylabel('Rótulos Verdadeiros')
plt.title('Matriz de Confusão')
plt.show()

