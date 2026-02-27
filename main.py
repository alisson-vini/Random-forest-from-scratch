import pandas as pd
from models import random_forest
from models import decision_tree
from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# DATASET
tabela = load_digits(as_frame=True)
X = tabela.data
y = tabela.target

entrada_treino, entrada_teste, target_treino, target_teste = train_test_split(X, y, test_size=0.2, random_state=23)

# Random forest implementação manual
forest = random_forest.Radom_forest(n_estimators=50)
forest.train(entrada_treino, target_treino)

predict = forest.predict(entrada_teste)

quantidade_acertos = (predict.values == target_teste.values).sum()
print(f"Quantidade de acertos manual: {quantidade_acertos} - {quantidade_acertos / len(target_teste) * 100}%")

# Random forest do Sklearn
random_sk = RandomForestClassifier(n_estimators=50, random_state=23)
random_sk.fit(entrada_treino, target_treino)
predict = random_sk.predict(entrada_teste)

quantidade_acertos = (predict == target_teste).sum()
print(f"Quantidade de acertos sklern: {quantidade_acertos} - {quantidade_acertos / len(target_teste) * 100}%")

# arvore de decisão manual
tree = decision_tree.Decision_tree(random_state=23)
tree.create_tree(entrada_treino, target_treino)
predict = tree.predict(entrada_teste)

quantidade_acertos = (predict.values == target_teste.values).sum()
print(f"Quantidade de acertos manual: {quantidade_acertos} - {quantidade_acertos / len(target_teste) * 100}%")

# arvore de decisão do Sklearn
tree = DecisionTreeClassifier(random_state=23)
tree.fit(entrada_treino, target_treino)
predict = tree.predict(entrada_teste)

quantidade_acertos = (predict == target_teste).sum()
print(f"Quantidade de acertos sklearn: {quantidade_acertos} - {quantidade_acertos / len(target_teste) * 100}%")