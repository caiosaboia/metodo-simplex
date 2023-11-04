import numpy as np

TOL = 1e-6 # Constante para comparações aproximadas com zero.

# Dados do problema.
# Forma padrão: min <c, x> s.a Ax = b, x >= 0.

# Matrizes e vetores originais
c = np.array([ -8, -4])
A = np.array([[ 3, 1],
              [ 1, 1],
              [ 1, 0]])
b = np.array([[7], [5], [2]])

(m, n) = A.shape

# Matrizes e vetores modificados
identidade = np.eye(m)
A_nova = np.hstack((A, identidade))
c_nova = np.hstack((c, np.zeros(A_nova.shape[1] - len(c))))

(p, q) = A_nova.shape  # Novo

# Determine o conjunto de B_inicial que corresponde apenas às colunas da matriz original A
B_inicial = list(range(n))

print(f"Problema com {m} linhas e {n} colunas.\n")

# Conjunto B
B = B_inicial
N = list(range(n, q))

iteracao = 0

while True:
    
    iteracao += 1  # Aumentamos o contador de iterações. A primeira iteração é contada com <iteracao>=1.

    # Partição em torno de variáveis básicas e não básicas.
    AB = A_nova[:, B]
    AN = A_nova[:, N]
    cB = c_nova[B]
    cN = c_nova[N]

    # Dados do dicionário
    AB_1 = np.linalg.pinv(AB)
    xB = AB_1 @ b
    AB_1AN = AB_1 @ AN
    z = cB.T @ xB
    cr = cN.T - cB.T @ (AB_1AN)
    print(f"B={B}, N={N}, z={z}")
    print(f"Sol.: {xB.T}")
    
    print("Dicionário:")
    with np.printoptions(precision=2, suppress=True):
        print(np.block([[xB, -AB_1AN], [z, cr]]))

    # Determinar a variável de entrada
    candidatas = [(N[i], i, custo) for (i, custo) in enumerate(cr) if custo <= -TOL]
    if len(candidatas) == 0:
        print("Solução ótima.")
        break

    (varEntrada, indiceVarEntrada, crVarEntrada) = min(candidatas)
    print(f"Variável de entrada: {varEntrada}")

    colunaVarEntrada = AB_1AN[:, indiceVarEntrada]
    candidatas = [(xB[i] / pivo, B[i], pivo) for (i, pivo) in enumerate(colunaVarEntrada) if pivo > TOL]
    if len(candidatas) == 0:
        print("Problema ilimitado.")
        break

    (razaoMinima, varSaida, _) = min(candidatas)
    print(f"Variável de saída: {varSaida}")

    # Verificar quais variáveis em B não correspondem a colunas em A
    variaveis_nao_correspondentes = [var for var in B if var not in B_inicial]
    if variaveis_nao_correspondentes:
        print(f"Variáveis básicas em B que não correspondem a colunas em A: {variaveis_nao_correspondentes}")

    N = [varSaida if v == varEntrada else v for v in N]
    B = [varEntrada if v == varSaida else v for v in B]

    # Remova variáveis de B que não correspondem a colunas originais em A
    B = [var for var in B if var in B_inicial]

    print("")
