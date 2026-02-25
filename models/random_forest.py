import pandas as pd
import numpy as np
from tqdm import tqdm
from .decision_tree import Decision_tree

class Radom_forest:
    def __init__(
        self,
        n_estimators:int=100,
        bootstrap:bool=True,
        max_samples:float=None,
        max_features:int | str = "sqrt",
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
        self.trees:list[Decision_tree] = list()                  # Uma lista que armazena todas as árvores
        self.rng = np.random.default_rng(seed=self.random_state) # Cria um gerador de números aleatório com uma semente reprodutível


    # Função auxiliar
    def get_more_votes(self, table:np.ndarray) -> pd.Series:
        """
        
        """

        array_target = []

        # percorre cada linha da tabela transposta (cada coluna)
        for linha in table.T:
            valores, contagens = np.unique(linha, return_counts=True)
            array_target.append( valores[np.argmax(contagens)] )

        return pd.Series(array_target)

    # função criar as tabelas com bootstrap
    def make_bootstrap(self, table:pd.DataFrame, qt_amostras:int) -> pd.DataFrame:
        """
        
        """
        
        # gera o DataFrame com bootstrap
        bootstrap_table = table.sample(n=qt_amostras, replace=True, random_state=self.rng.integers(1, 100_000_000))

        return bootstrap_table

    def random_feature(self, array_coluns:pd.Series) -> pd.Series:
        """
        
        """

        if self.max_features == "sqrt":
            qt_features = int(len(array_coluns) ** (1/2))
        elif isinstance(self.max_features, int):
            qt_features = self.max_features
        else:
            raise ValueError("Valor invalido para QT. features")

        # um array com as n features selecionadas aleatóriamente de todas as efatures
        featrues = array_coluns.sample(n=qt_features, replace=False, random_state=self.rng.integers(1, 100_000_000))

        return featrues

    def train(self, table:pd.DataFrame, target:pd.Series) -> None:
        """
        
        """

        # reseta a floresta de árvores, para o caso da função ser chamda multiplas vezes isso é muito importante
        self.trees = []

        if self.max_samples == None: qt_amostras = len(table)
        else: qt_amostras = int(self.max_samples * len(table))

        # cria e treina a N arvores
        for _ in tqdm(range(self.n_estimators)):
            
            if self.bootstrap: temp_table = self.make_bootstrap(table, qt_amostras) # aplica o bootstrap para pegar N linhas do dataset original
            else: temp_table = table

            features = self.random_feature(table.columns.to_series()).to_list()     # pega quais vão ser as N features que vão ser usadas
            temp_table = temp_table.loc[:, features]                                # retira das colunas as features que não foram selecionadas

            temp_target = target.loc[temp_table.index]

            # instanciando a classe
            tree = Decision_tree(
                self.max_deep,
                self.min_samples_split,
                self.min_samples_leaf,
                self.min_impurity_decrease,
                self.rng.integers(1,100_000_000),
                self.max_features
            )
            # treina a árvore
            tree.create_tree(temp_table, temp_target)

            # adiciona a arvore na floresta
            self.trees.append(tree)
    
    def predict(self, table:pd.DataFrame) -> pd.Series:
        """
        
        """

        # matriz onde cada linha são os resultados de uma arvore dado esse conjunto de entradas
        matriz_resultado = []

        for tree in self.trees:
            targets_tree = tree.predict(table)
            matriz_resultado.append( np.array(targets_tree) )

        matriz_resultado = np.array(matriz_resultado)
        return self.get_more_votes(matriz_resultado)