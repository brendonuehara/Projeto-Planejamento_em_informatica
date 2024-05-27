import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

dados = pd.read_csv("CSV/dados_usuarios.csv", sep=",")

#Parte 01

media_idade = dados['idade'].mean()
print(f"Média das idades: {media_idade}")

mediana_idade = dados['idade'].median()
print(f"Mediana das idades: {mediana_idade}")

desvio_padrao_idade = dados['idade'].std()
print(f"Desvio padrão das idades: {desvio_padrao_idade}")

media_valor_pago = dados['valor_pago'].mean()
print(f"Média dos valores pagos: {media_valor_pago}")

mediana_valor_pago = dados['valor_pago'].median()
print(f"Mediana dos valores pagos: {mediana_valor_pago}")

desvio_padrao_valor_pago = dados['valor_pago'].std()
print(f"Desvio padrão dos valores pagos: {desvio_padrao_valor_pago}")


## Parte 02

contagem_produto = dados['tipo_produto'].value_counts()

plt.figure(figsize=(10, 6))
sns.barplot(x=contagem_produto.index, y=contagem_produto.values, palette="viridis")
plt.title('Contagem de Pedidos por Tipo de Produto')
plt.xlabel('Tipo de Produto')
plt.ylabel('Contagem de Pedidos')
plt.xticks(rotation=45) 
plt.show()

sns.countplot(data=dados, x='tipo_pagamento', palette='pastel')
plt.title('Contagem de Tipos de Pagamento')
plt.xlabel('Tipo de Pagamento')
plt.ylabel('Contagem')

plt.tight_layout()
plt.show()


## Parte 03

padrao_tipo_pagamento = dados['tipo_pagamento'].value_counts()
print("Padrões no Tipo de Pagamento:")
print(padrao_tipo_pagamento)
print()

padrao_tipo_produto = dados['tipo_produto'].value_counts()
print("Padrões no Tipo de Produto:")
print(padrao_tipo_produto)
print()

padrao_status_pedido = dados['status_pedido'].value_counts()
print("Padrões no Status do Pedido:")
print(padrao_status_pedido)







