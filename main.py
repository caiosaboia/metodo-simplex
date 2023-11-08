import numpy as np

TOL = 1e-6 # Constante para comparacoes aproximadas com zero.

#Devemos levar em consideracao que o nosso vetor b sempre será possitivo, por isso, devemos escrever de forma adequada na entrada dos valores.

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



matriz_identidade = np.eye(m, m) #criamos uma matriz identidade para compararmos à uma base boa
ultimas_colunas_A = A[:, -m:] #E aqui serao as colunas na qual vamos comparar




# Verificar se as n últimas colunas de A são uma boa base ou nao
if np.allclose(matriz_identidade, ultimas_colunas_A, atol=TOL):
    print("As últimas colunas de A formam uma boa base.") #caso sim, ela percorre o código original normalmente
else:
    print("As últimas colunas de A NÃO formam uma boa base.") #caso nao...
    A_nova = nova_matriz = np.hstack((A, np.eye(m))) #criamos uma nova matriz com as var artificias
    c_artificial = np.zeros(n)  #um novo c
    c_artificial = np.hstack((c_artificial, np.ones(m))) #e adicionamos ao c nossas var atificias
    c = c_artificial        #aqui é meramente para manter a sintaxe do código
    A = A_nova              #aqui o mesmo
    (m,n) = A_nova.shape    #o mesmo
    

print("")

# ------------------------------------------------------------------------------------------------------
# Base inicial
B = list(range(n-m,n))
N = list(range(n-m))

iteracao = 0


#Aqui abaixo o código corre normamente, exceto por uma acrescimo(line 85) para mostrarmos quais var na Base sao artificias.

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

