import pandas as pd
import numpy as np
from .decision_tree import Decision_tree
from tqdm import tqdm

class Radom_forest:

    # função para inicializar a classe e seus atributos
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
        

    # Função auxiliar para pegar o target que mais se repete dentre as respostas da Decision Tree
    def get_more_votes(self, table:np.ndarray) -> pd.Series:
        """
        Função serve para pegar o elemento que mais se repete entre os predicts de cada linha do dataset, resultante das várais árvores

        Parâmetros:
            table -> um matriz formada por array numpy (2D) onde cada linha é o conjunto de saídas (predict) de cada uma das árvores
        
        Return:
            Uma pd.Series com predict mais votado final respectivo para cada amostra
        """

        array_target = [] # array que vai conter o conjunto de respostas final

        # percorre cada linha da tabela transposta (cada coluna, todos os predicts para uma amostra de dados)
        for linha in table.T:
            valores, contagens = np.unique(linha, return_counts=True) # pega todos os resultados e a contagem de cada um deles
            array_target.append( valores[np.argmax(contagens)] )      # adiciona o valor que mais se repete a tabela com os resultados finais

        return pd.Series(array_target)

    # função criar as tabelas com bootstrap
    def make_bootstrap(self, table:pd.DataFrame, qt_amostras:int) -> pd.DataFrame:
        """
        Função para fazer o bootstrap, pegar N colunas de forma aleatória com reposição

        Parâmetros:
            table -> tabela que vai ser usada como base para o bootstrap
            qt_amostras -> quantidade de linhas que o boostrap vai pegar

        return:
            pd.DataFrame
        """
        
        # gera o DataFrame com bootstrap
        bootstrap_table = table.sample(n=qt_amostras, replace=True, random_state=self.rng.integers(1, 100_000_000))

        return bootstrap_table

    # função para retornar a quantidade de linhas que a tabela vai retornar e a quantidade de colunas
    def table_feature(self, table:pd.DataFrame) -> pd.Series:
        """
        Função que serve para retornar a quantidade de linhas que a tabela vai ter e a quantidade de features aleatórias que o modelo vai usar
        para escolher a melhor coluna

        Parâmetros:
            table -> tabela original usada para o treino da árvore
        """

        # quantidade de linhas que a tabela vai ter (usado no bootstrap)
        if self.max_samples == None: qt_amostras = len(table)
        else: qt_amostras = int(self.max_samples * len(table))

        # quantidade de colunas aleatorias do todo que vão ser selecionadas em cada nó da arvore durante o treino para escolher a melhor
        if self.max_features == "sqrt":
            qt_features = int(len(table.columns) ** (1/2))
        elif isinstance(self.max_features, int):
            qt_features = self.max_features
        else:
            raise ValueError("Valor invalido para QT. features")
        
        return qt_amostras, qt_features

    # função para treinar a árvore
    def train(self, table:pd.DataFrame, target:pd.Series) -> None:
        """
        Função que serve para treinar a Random forest

        Parâmetros:
            table -> A tabela original
        """

        # reseta a floresta de árvores, para o caso da função ser chamda multiplas vezes isso é muito importante
        self.trees = []

        # pega a quantidade de linhas do boostrap e a quantidade de colunas que precisa selecionar aleatoriamente em cada split para olhar qual a melhor
        qt_amostras, qt_features = self.table_feature(table)

        # cria e treina a N arvores
        for _ in tqdm(range(self.n_estimators)):
            
            if self.bootstrap: temp_table = self.make_bootstrap(table, qt_amostras) # aplica o bootstrap para pegar N linhas do dataset original
            else: temp_table = table                                                # para o caso de boostrap == False pega a quantidade total de linhas da tabela

            temp_target = target.loc[temp_table.index] # pega os targets das linhas selecionadas da tabela usando o bootstrap

            # instanciando a classe
            tree = Decision_tree(
                self.max_deep,
                self.min_samples_split,
                self.min_samples_leaf,
                self.min_impurity_decrease,
                self.rng.integers(1,100_000_000),
                qt_features
            )

            # treina a árvore
            tree.create_tree(temp_table, temp_target)

            # adiciona a arvore na floresta
            self.trees.append(tree)
    
    # função para fazer o predict usando todas as árvores
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