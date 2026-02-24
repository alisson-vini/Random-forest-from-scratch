import pandas as pd
import numpy as np

class Radom_forest:
    def __init__(
        self,
        n_estimators:int=100,
        bootstrap:bool=True,
        max_samples:float=None,
        max_features:int="sqrt",
        random_state:int=23,

        max_deep:int=None,
        min_samples_split:int=2,
        min_samples_leaf:int=1,
        min_impurity_decrease:float=0.0
    ):
        # Atributos da Random forest
        self.n_estimators = n_estimators                   # Quantidade de árvores de decisão
        self.bootstrap = bootstrap                         # True para fazer bootstrap | False para não fazer
        self.max_samples = max_samples                     # Quantidade de elementos aleatorios (com reposição) que o bootstrap vai pegar (Len(table) * max_samples)
        self.max_features = max_features                   # Quantidade de features aleatórias que vão ser passadas para cada uma das arvores menores
        self.random_state = random_state                   # É como uma seed, faz com que os evento aleatorios sempre sejam os mesmo para determianda seed

        # Atriutos da Decision tree
        self.max_deep = max_deep                           # profundida máxima da árvore
        self.min_samples_split = min_samples_split         # Quantidade mínima de amostras para que tente fazer o split
        self.min_samples_leaf = min_samples_leaf           # Quantidade mínima de amostras em qualquer uma das folhas para que o split seja feito
        self.min_impurity_decrease = min_impurity_decrease # Decrescimento de impureza mínimo para fazer um split
        
        # Atributos utilitários
        self.trees = np.array()                            # Uma lista que armazena todas as árvores

        # função criar as tabelas com bootstrap
        def bootstrap(self, table:pd.DataFrame) -> pd.DataFrame:
            """
            
            """

            qt_amostras = max_samples * len(table)
            # gera um array np que contem várias tabelas, cada uma delas é para ser usada para construção das max_samples arvores
            array_tables = np.array( [table.sample(n=qt_amostras)] for _ in range(max_samples) )

            return array_tables

        def treinar_arvores(self):
            pass
