import numpy as np

TOL = 1e-6 # Constante para comparacoes aproximadas com zero.


'''
Formato para o site https://online-optimizer.appspot.com/?model=builtin:default.mod :

var x1 >= 0;
var x2 >= 0;

maximize z:     8*x1 + 4*x2;

subject to c1: 3*x1 +  x2 <= 7;
subject to c2:   x1 +  x2 <= 5;
subject to c3:   x1       <= 2;

end;
'''
# Exemplo Que nao precisa de var artificial
# c = np.array([ -8, -4, 0, 0, 0])
# A = np.array([[ 3, 1, 1, 0, 0],
#               [ 1, 1, 0, 1, 0],
#               [ 1, 0, 0, 0, 1]])
# b = np.array([[7], [5], [2]])


# Exemplo Que PRECISA de var artificial
c = np.array([ -6, 1, 0,  0])
A = np.array([[ 4, 1, 1,  0],
              [ 2, 3, 0, -1],
              [-1, 1, 0,  0]])
b = np.array([[21], [13], [1]])

(m,n) = A.shape

print(f"Problema com {m} linhas e {n} colunas.\n")

# Verificar se as n últimas colunas de A formam uma boa base inicial
matriz_identidade = np.eye(m, m)
ultimas_colunas_A = A[:, -m:]

if np.allclose(matriz_identidade, ultimas_colunas_A, atol=TOL):
    print("As últimas colunas de A formam uma boa base.")
else:
    print("As últimas colunas de A NÃO formam uma boa base.")
    A_nova = nova_matriz = np.hstack((A, np.eye(m)))
    c_artificial = np.zeros(n)
    c_artificial = np.hstack((c_artificial, np.ones(m)))
    c = c_artificial
    A = A_nova
    (m,n) = A_nova.shape
    

print("")

# ------------------------------------------------------------------------------------------------------
# Base inicial
B = list(range(n-m,n))
N = list(range(n-m))

iteracao = 0


while True:

    iteracao += 1 # Aumentamos o contador de iteracoes. A primeira iteracao e' contada com <iteracao>=1.
    
    # ------------------------------------------------------------------------------------------------------
    # Particao em torno de variaveis basicas e nao-basicas.
    AB = A[:,B]
    AN = A[:,N]
    cB = c[B]
    cN = c[N]

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
    print("")
    
    #Verifica quais sao originais
    originais_de_A = [v for v in B if v > m]
    print(f"Variáveis Artificias em B: {originais_de_A}")


    # ------------------------------------------------------------------------------------------------------
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

