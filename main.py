import pandas as pd
from models import random_forest
from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# DATA SET IRIS
tabela = load_digits(as_frame=True)
X = tabela.data
y = tabela.target

entrada_treino, entrada_teste, target_treino, target_teste = train_test_split(X, y, test_size=0.2, random_state=23)

forest = random_forest.Radom_forest(n_estimators=100)
forest.train(entrada_treino, target_treino)

predict = forest.predict(entrada_teste)

quantidade_acertos = (predict.values == target_teste.values).sum()

print(f"Quantidade de acertos manual: {quantidade_acertos} - {quantidade_acertos / len(target_teste) * 100}%")

random_sk = RandomForestClassifier(n_estimators=50, random_state=23)
random_sk.fit(entrada_treino, target_treino)
predict = random_sk.predict(entrada_teste)

quantidade_acertos = (predict == target_teste).sum()
print(f"Quantidade de acertos sklern: {quantidade_acertos} - {quantidade_acertos / len(target_teste) * 100}%")