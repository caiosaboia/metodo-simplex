# UFC/DEMA, Programacao Linear, 2023.2
# Um implementacao (limitada e ineficiente) do metodo Simplex.
#
# "Incompleta" porque nao realiza a primeira fase do algoritmo. Em vez disso,
# simplesmente assume que a submatriz formada pelas ultimas m linhas e'
# nao-singular e que a base associada a tal matriz e' viavel (ambas as hipoteses
# nao sao verdadeiras no caso geral).
#
# "Ineficiente" porque a cada iteracao a submatriz correspondente 'a base atual e'
# explicitamente invertida, o que e' um processo computacionalmente caro. Em vez
# disso, o algoritmo podeeria calcular o proximo dicionario sem explicitamente
# inverter a matriz AB.
#
# Corrigir estas duas limitacoes do codigo e' sugerido como exercicio.

import numpy as np

TOL = 1e-6 # Constante para comparacoes aproximadas com zero.

# Dados do problema.
# Forma padrao: min <c,x> s.a Ax=b, x>=0.

'''
Formato para o site https://online-optimizer.appspot.com/?model=builtin:default.mod :

var x1 >= 0;
var x2 >= 0;

maximize z:     8*x1 + 4*x2;

subject to c1: 3*x1 +  x2   <= 7;
subject to c2:   x1 +  x2   <= 5;
subject to c3:   x1         <= 2;

end;
'''

#problema original
c = np.array([ -8, -4])
A = np.array([[ 3, 1],
              [ 1, 1],
              [ 1, 0]])
b = np.array([[7], [5], [2]])
(m,n) = A.shape

#problema alterado com var de folga
identidade = np.eye(m)
A_nova = np.hstack((A, identidade))
c_nova = np.hstack((c,np.zeros(A_nova.shape[1]- len(c))))
(p,q) = A_nova.shape #novo

print(f"Problema com {m} linhas e {n} colunas.\n")

# ------------------------------------------------------------------------------------------------------
# Base inicial
B = list(range(q-p,q))
N = list(range(q-p))

iteracao = 0

while True:
    
    iteracao += 1 # Aumentamos o contador de iteracoes. A primeira iteracao e' contada com <iteracao>=1.
    
    # ------------------------------------------------------------------------------------------------------
    # Particao em torno de variaveis basicas e nao-basicas.
    AB = A_nova[:,B]
    AN = A_nova[:,N]
    cB = c_nova[B]
    cN = c_nova[N]

    # ------------------------------------------------------------------------------------------------------
    # Dados do dicionario
    AB_1 = np.linalg.inv(AB)
    xB = AB_1@b
    AB_1AN = AB_1@AN
    z = cB.T@xB
    cr = cN.T - cB.T@(AB_1AN)
    print(f"B={B}, N={N}, z={z}")
    print(f"Sol.: {xB.T}")
    
    D = np.block([[xB, -AB_1AN], [z, cr]]) # Parte do dicionario
    print("Dicionario:")
    print(D)

    # ------------------------------------------------------------------------------------------------------
    # Verifica se há índices em B que não correspondem às colunas da matriz original A
    indices_invalidos = [var for var in B if var >= n]
    if indices_invalidos:
        print(f"Variáveis básicas em B que não correspondem a colunas em A: {indices_invalidos}")

    # Determinar a variavel de entrada
    candidatas = [(N[i], i, custo) for (i,custo) in enumerate(cr) if custo <= -TOL]
    if len(candidatas) == 0:
        print("Solucao otima.")
        break

    # Escolher variavel de indice minimo dentre as variaveis nao-basicas com custo reduzido negativo
    (varEntrada,indiceVarEntrada,crVarEntrada) = min(candidatas)
    print(f"Variavel de entrada: {varEntrada}")

    # ------------------------------------------------------------------------------------------------------
    # Determinar a variavel de saida
    colunaVarEntrada = AB_1AN[:,indiceVarEntrada]
    candidatas = [(xB[i]/pivo, B[i], pivo) for (i,pivo) in enumerate(colunaVarEntrada) if pivo > TOL]
    if len(candidatas) == 0:
        print("Problema ilimitado.")
        break

    # Escolher a variavel de indice minimo dentre aquelas que atingem a razao minima entre o valor da
    # variavel basica e o negativo do valor do pivo. Remeta-se ao conceito de "teste da razao" para
    # perceber como foi escolhida a variavel de saida.
    (razaoMinima,varSaida,_) = min(candidatas)
    print(f"Variavel de saida: {varSaida}")

    # ------------------------------------------------------------------------------------------------------
    # Troca de base
    N = [varSaida if v == varEntrada else v for v in N]
    B = [varEntrada if v == varSaida else v for v in B]

    print("")

