# Radom Forest

O algoritmo **Radom Forest** é uma evolução natural das Árvores de decisão, ele basicamente funciona criando várias árvores de decisão menores e combinando o resultado delas. Mais informações sobre como esse algorítmo funciona são encontradas posteriormente nesse README.

## O que esse código faz

Os códigos presentes nesse repositório servem para criar um algoritmo de Radom Forest, no main existe um teste comparando o desempenho desse modelo em comparação com seu antecessor (árvores de decisão) e também compara minha implementação manual com a do scikit-learn no dataset digits (dataset com várias imagens 8x8 que representam imagens de dígitos), mas os códigos desenvolvidos aqui também podem ser usados em datasets próprios da mesma forma.

## O que é preciso para rodar o código

    Para rodar esse código as seguintes bibliotecas são necessárias:
    - pandas
    - numpy
    - scikit-learn (para a validação do modelo manual)
    
    caso não tenha todas as bibliotecas instaladas rode esse código no CMD:
    pip install pandas numpy scikit-lear

## Como rodar o código

Você pode simplismente rodar o código presente em `main.py` para poder ver como o desempenho do modelo no dataset digits, mas caso queira utilizar outro dataset os passos são:

1. Se sertifique de ter todas as bibliotecas informadas a cima instaladas
1. Crie um arquivo .py ou use o próprio main.py presente aqui
1. Import o módulo `random_forest.py` da pasta `models` da seguinte forma: `from .decision_tree import Decision_tree`
1. Baixe o dataset que você quer utilizar
1. Separe os dados em treino e teste
1. Instancie a classe Random_forest da seguinte forma: `forest = random_forest.Radom_forest()`
1. Treine o objeto usando o método `.train` passando o conjunto de dados de treino
1. Utilize o método `.predict` passando um Datafrema com as mesmas carácteristicas do Dataframe usado para treinar o modelo e você vai ser uma pd.Series com os predicts para cada linha dos dados passados
1. Rode o código 

Você pode inspecionar o código presente no main para ver melhor como cada uma dessas etápas funcionam

## Explicação detalhada da lógica e matemática por trás do funcionamento do modelo Random Forest

OBS: Antes de começar vale a pena estudar Árvores de decisão, pois elas são a base para o funcionamento desse modelo, nesse link está um repositório do GitHub onde eu explico mais profundamente como elas funcionam [Decision Tree](https://github.com/alisson-vini/Decision_tree_from_scratch)

Agora partindo para realmente para Random Forest, elas tem esse nome justamente por serem formadas por várias arvores de decisões menores, pegamos a previsão de todas essas arvores e vemos qual o valor que mais se repete; A intuição matemática por trás do ganho de performance pode ser entendida através do **Teorema do Júri de Condorcet**. aqui vai uma explicação sobre ele para reforçar mais o entendimento do porque isso funciona:

### Teorema do júri

O teorema do júri afirma que desde que tenhamos votantes minimamente capazes, ou seja a probabilidade deles tomarem uma decisão correta for >0.5, e que cada voto seja independente, quanto mais votantes tivermos maior é a probabilidade deles juntos tomarem uma decisão correta, para entender que isso é verdade podemos tomar o seguinte raciocínio lógico:

A escolha do voto final é definida pela decisão mais escolhida, portando precisamos que (N+1) / 2 votos sejam do mesmo grupo para esse ser a decisão final

Ex de votação: ✓ ✓ ✓ X X -> decisão final do júri seria ✓ (considerando essas amostras o júri tem 60% de chance de acertar e 40% de chance de errar)

considerando isso precisamos que no final dos votos a quantidade de votos corretos seja >50% do todo para essa ser a decisão final, pela teoria dos grande números quanto mais amostras temos maior é a chance dela representar a proporção real das probabilidades, ou seja, caso cada um dos votantes tenha 51% de chance de acertas se tivermos um número suficiente grande de votantes a essa probabilidade vai se concretizar e isso por sua vez já faria com que a decisão final fosse correta já que só precisamos ter a maioria dos votos corretos

Ex para entender melhor: Vamos imaginar que todos os votantantes tem 60% de chance de acertar e 40% de chance de errar, se tivermos apenas 3 votantes poderiamos ter uma confiração de votos como essa:

[✓ X X]

Mas se ao inves de 3 votantes tivermos 100 votantes a teoria dos grande números confirmar que a probabilidade de termos umas dispersão de votos mais próximo 60%/40% é muito maior do que com apenas 3, e assim que o números de votos certos for >50% do total já temos o voto certo como resposta final.

Outra forma de entender isso é matematicamente onde temos a seguinte formula que calcula a probabilidade do resultado final ser correto levando em consideração o número de votantes e a probabilidade de cada um deles escolher a resposta certa:

$$
P_{\text{majority}} =
\sum_{i=\left\lceil \frac{N+1}{2} \right\rceil}^{N}
\binom{N}{i}
P^i (1-P)^{N-i}
$$

- N = quantidade total de votantes
- P = probabilidade de cada votante acertar

se calcularmos o limite dessa função quando N tente ao infinito para P < 0.5, P = 0.5 e P > 0.5 vamos comprovar o que foi explicado anteriormente.

**OBS**: Lembrando que isso só acontece quando não existe correlação entre os votantes, ou seja, o voto de uma pessoa não enfluencia **de forma alguma** o voto de outra

levando isso em consideração podemos extender esse raciocínio para as árvores de decisão e entender que, desde que cada uma das árvores não tenha corelação entre sí, a junção de cada uma delas vai nos dar um bom resultado, dessa forma partimos para o problema principal, como criar várisa arvores de forma que uma não tenha corelação com a outra se elas são construidas com base no mesmo conjunto de dados e dessa forma tendem a cometer os mesmo erros, é como se cada arvores tivesse crescido na mesma casa, morava na mesma rua, estudava na mesma escola com o mesmos professores, dessa forma cada uma delas tende a ter um conjunto de experiências (respostas) parecidas, e precisamos que cada uma delas responda com uma chance de acertar >0.5 **ao mesmo tempo** que elas não dão o mesmo resultado para o mesmo conjunto de entradas (ou seja, o requesito de indepemdência do teorema do jurí)

