import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def carregar_dados_arquivo(caminho_arquivo):
    """
    Carrega os dados de anos e população a partir de um arquivo .dat.

    Args:
        caminho_arquivo (str): Caminho para o arquivo .dat.

    Retorna:
        anos (np.ndarray): Anos absolutos.
        x (np.ndarray): Anos relativos a 1940.
        y (np.ndarray): População.
    """
    # Carrega os dados ignorando a primeira linha (cabeçalho)
    dados = np.loadtxt(caminho_arquivo, skiprows=1)

    anos = dados[:, 0].astype(int)
    populacao = dados[:, 1].astype(int)

    x = anos - 1940  # Anos relativos a 1940
    y = populacao

    return anos, x, y



# -------------------
# Definição dos modelos matemáticos
# -------------------

def modelo_exponencial(x, a, b):
    """Modelo Exponencial: y = a * exp(b * x)"""
    return a * np.exp(b * x)


def modelo_hiperbolico(x, a, b):
    """Modelo Hiperbólico: y = a / (b + x)"""
    return a / (b + x)


def modelo_geometrico(x, a, b):
    """Modelo Geométrico (Potencial): y = a * x^b"""
    return a * x**b


# -------------------
# Ajuste dos modelos
# -------------------

def ajustar_modelos(x, y):
    """
    Ajusta os dados aos modelos polinomiais, exponencial, hiperbólico e geométrico.

    Args:
        x (np.ndarray): Anos relativos a 1940.
        y (np.ndarray): População.

    Retorna:
        poly2 (np.poly1d): Polinômio de grau 2 ajustado.
        poly3 (np.poly1d): Polinômio de grau 3 ajustado.
        params_exp (tuple): Parâmetros (a, b) do modelo exponencial.
        params_hip (tuple): Parâmetros (a, b) do modelo hiperbólico.
        params_geo (tuple): Parâmetros (a, b) do modelo geométrico (potencial).
        x_geo (np.ndarray): x ajustado para o modelo geométrico (evita zero).
    """
    coef_poly2 = np.polyfit(x, y, 2)
    coef_poly3 = np.polyfit(x, y, 3)

    poly2 = np.poly1d(coef_poly2)
    poly3 = np.poly1d(coef_poly3)

    params_exp, _ = curve_fit(modelo_exponencial, x, y, p0=(10000, 0.03))

    # ----- Hiperbólico -----
    a0 = max(y) * (max(x) + 1)
    b0 = 1

    try:
        params_hip, _ = curve_fit(
            modelo_hiperbolico,
            x,
            y,
            p0=(max(y), 1),
            bounds=([0, 0], [1e12, 50])
        )
    except RuntimeError:
        print("Falha no ajuste hiperbólico!")
        params_hip = (np.nan, np.nan)

    # ----- Geométrico -----
    x_geo = x.copy()
    x_geo[x_geo == 0] = 0.1

    params_geo, _ = curve_fit(modelo_geometrico, x_geo, y, p0=(10000, 1))
    
    media_real = np.mean(y)
    n = len(y)
    valores_estimados_poly2 = poly2(x)
    r2_poly2 = calcular_coeficiente_determinacao(y, valores_estimados_poly2, media_real, n)
    
    valores_estimados_poly3 = poly3(x)
    r2_poly3 = calcular_coeficiente_determinacao(y, valores_estimados_poly3, media_real, n)
    
    valores_estimados_exp = modelo_exponencial(x, *params_exp)
    r2_exp = calcular_coeficiente_determinacao(y, valores_estimados_exp, media_real, n)

    valores_estimados_hip = modelo_hiperbolico(x, *params_hip)
    r2_hip = calcular_coeficiente_determinacao(y, valores_estimados_hip, media_real, n)

    valores_estimados_geo = modelo_geometrico(x_geo, *params_geo)
    r2_geo = calcular_coeficiente_determinacao(y, valores_estimados_geo, media_real, n)
    
    coeficientes = {
        "poly2": r2_poly2,
        "poly3": r2_poly3,
        "exp": r2_exp,
        "hip": r2_hip,
        "geo": r2_geo,
    }
    
    return poly2, poly3, params_exp, params_hip, params_geo, x_geo, coeficientes



# -------------------
# Plotagem dos modelos
# -------------------

def plotar_polinomio(anos, x, y, poly, grau, cor):
    """
    Plota o ajuste polinomial.

    Args:
        anos (np.ndarray): Anos absolutos.
        x (np.ndarray): Anos relativos a 1940.
        y (np.ndarray): População.
        poly (np.poly1d): Polinômio ajustado.
        grau (int): Grau do polinômio (2 ou 3).
        cor (str): Cor da linha no gráfico.
    """
    plt.figure(figsize=(8, 5))
    plt.scatter(anos, y, color='black', label='Dados reais')

    x_cont = np.linspace(min(x), max(x), 500)
    plt.plot(anos[0] + x_cont, poly(x_cont), color=cor, label=f'Polinomial grau {grau}')

    plt.title(f'Ajuste Polinomial Grau {grau}')
    plt.xlabel('Ano')
    plt.ylabel('População')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.savefig(f'output/polinomio_grau_{grau}.png')


def plotar_modelo(anos, x, y, func, params, label, color, x_model=None):
    """
    Plota o ajuste de um modelo não polinomial.

    Args:
        anos (np.ndarray): Anos absolutos.
        x (np.ndarray): Anos relativos a 1940.
        y (np.ndarray): População.
        func (function): Função do modelo (exponencial, hiperbólico ou geométrico).
        params (tuple): Parâmetros ajustados do modelo.
        label (str): Nome do modelo para legenda.
        color (str): Cor da linha no gráfico.
        x_model (np.ndarray, opcional): Vetor de x usado no modelo (para tratar casos como o geométrico).
    """
    plt.figure(figsize=(8, 5))
    plt.scatter(anos, y, color='black', label='Dados reais')

    if x_model is None:
        x_model = x

    x_cont = np.linspace(min(x_model), max(x_model), 500)
    plt.plot(anos[0] + x_cont, func(x_cont, *params), color=color, label=label)

    plt.title(f'Ajuste {label}')
    plt.xlabel('Ano')
    plt.ylabel('População')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.savefig(f'output/{label.lower()}.png')

def calcular_coeficiente_determinacao(valores_reais, valores_estimados, media_real, n):
    """
    Calcula o coeficiente de determinação (R²) para avaliar a qualidade do ajuste de um modelo.
    Args:
        valores_reais (list or array-like): Lista dos valores reais observados.
        valores_estimados (list or array-like): Lista dos valores estimados pelo modelo.
        media_real (float): Média dos valores reais observados.
        n (int): Número de elementos nas listas de valores.
    """
    soma_numerador = 0
    for i in range(0, n):
        soma_numerador += (valores_reais[i] - valores_estimados[i])**2
    
    soma_denominador = 0
    for i in range(0, n):
        soma_denominador += (valores_reais[i] - media_real)**2
        
    return 1 - soma_numerador/soma_denominador

def imprimir_melhor_modelo_e_estimativa(poly2, poly3, params_exp, params_hip, params_geo, coeficientes):
    """
    Imprime o melhor modelo de ajuste de dados com base no coeficiente de determinação (R²) e estima o valor para o ano de 2030.
    Args:
        poly2 (callable): Função do modelo polinomial de grau 2.
        poly3 (callable): Função do modelo polinomial de grau 3.
        params_exp (tuple): Parâmetros ajustados para o modelo exponencial.
        params_hip (tuple): Parâmetros ajustados para o modelo hiperbólico.
        params_geo (tuple): Parâmetros ajustados para o modelo geométrico (potencial).
        coeficientes (dict): Dicionário com os nomes dos modelos como chaves e seus respectivos valores de R² como valores.
    """
    
    maior_r2 = max(coeficientes.values())
    melhor_modelo = [k for k, v in coeficientes.items() if v == maior_r2][0]
    nomes = {
        "poly2": "Polinomial Grau 2",
        "poly3": "Polinomial Grau 3",
        "exp": "Exponencial",
        "hip": "Hiperbólico",
        "geo": "Geométrico (Potencial)"
    }
    print(f'\nMelhor ajuste: {nomes[melhor_modelo]} (R² = {maior_r2:.4f})')

    x_2030 = 2030 - 1940 

    modelos = {
        "poly2": lambda x: poly2(x),
        "poly3": lambda x: poly3(x),
        "exp": lambda x: modelo_exponencial(x, *params_exp),
        "hip": lambda x: modelo_hiperbolico(x, *params_hip),
        "geo": lambda x: modelo_geometrico(x if x != 0 else 0.1, *params_geo)
    }

    estimativa = modelos[melhor_modelo](x_2030)
    print(f"Estimativa para o ano de 2030 pelo melhor modelo ({nomes[melhor_modelo]}): {estimativa:.2f}")
    return melhor_modelo, maior_r2, nomes, estimativa

def salvar_resultados_em_arquivo(poly2, poly3, params_exp, params_hip, params_geo, coeficientes, melhor_modelo, maior_r2, nomes, estimativa):
    """
    Salva os resultados dos ajustes de diferentes modelos matemáticos em um arquivo de texto.

    Args:
        poly2 (str): Representação da função ajustada do modelo polinomial de grau 2.
        poly3 (str): Representação da função ajustada do modelo polinomial de grau 3.
        params_exp (tuple): Parâmetros (a, b) do modelo exponencial na forma y = a * exp(b * x).
        params_hip (tuple): Parâmetros (a, b) do modelo hiperbólico na forma y = a / (x + b).
        params_geo (tuple): Parâmetros (a, b) do modelo geométrico (potencial) na forma y = a * x^b.
        coeficientes (dict): Dicionário contendo os coeficientes de determinação R² para cada modelo, com as chaves 'poly2', 'poly3', 'exp', 'hip' e 'geo'.
        melhor_modelo (int): Índice do modelo com o melhor ajuste (maior R²).
        maior_r2 (float): Valor do maior coeficiente de determinação R² encontrado.
        nomes (list): Lista com os nomes dos modelos, usados para exibição do melhor ajuste.
        estimativa (float): Valor estimado para o ano de 2030 pelo melhor modelo.
    """
    with open('output/output.txt', 'w') as f:
        f.write('Resultados dos Modelos Ajustados:\n\n')
        f.write('1) Modelo Polinomial Grau 2:\n')
        f.write(f'   Função: \n\t{poly2}\n')
        f.write(f'   R²: {coeficientes["poly2"]:.4f}\n')
        if coeficientes["poly2"] < 0:
            f.write('   R² negativo, portanto, modelo não adequado para esses dados\n')
        f.write('\n')

        f.write('2) Modelo Polinomial Grau 3:\n')
        f.write(f'   Função: \n\t{poly3}\n')
        f.write(f'   R²: {coeficientes["poly3"]:.4f}\n')
        if coeficientes["poly3"] < 0:
            f.write('   R² negativo, portanto, modelo não adequado para esses dados\n')
        f.write('\n')

        f.write('3) Modelo Exponencial:\n')
        f.write(f'   Função: \n\ty = {params_exp[0]:.2f} * exp({params_exp[1]:.5f} * x)\n')
        f.write(f'   R²: {coeficientes["exp"]:.4f}\n')
        if coeficientes["exp"] < 0:
            f.write('   R² negativo, portanto, modelo não adequado para esses dados\n')
        f.write('\n')

        f.write('4) Modelo Hiperbólico:\n')
        f.write(f'   Função: \n\ty = {params_hip[0]:.2f} / (x + {params_hip[1]:.5f})\n')
        f.write(f'   R²: {coeficientes["hip"]:.4f}\n')
        if coeficientes["hip"] < 0:
            f.write('   R² negativo, portanto, modelo não adequado para esses dados\n')
        f.write('\n')

        f.write('5) Modelo Geométrico (Potencial):\n')
        f.write(f'   Função: \n\ty = {params_geo[0]:.2f} * x^{params_geo[1]:.5f}\n')
        f.write(f'   R²: {coeficientes["geo"]:.4f}\n')
        if coeficientes["geo"] < 0:
            f.write('   R² negativo, portanto, modelo não adequado para esses dados\n')
        f.write('\n')

        f.write(f'Melhor ajuste: {nomes[melhor_modelo]} (R² = {maior_r2:.4f})\n')
        f.write(f'Estimativa para o ano de 2030 pelo melhor modelo ({nomes[melhor_modelo]}): {estimativa:.2f}\n')

# -------------------
# Impressão dos modelos
# -------------------

def imprimir_modelos(poly2, poly3, params_exp, params_hip, params_geo, coeficientes):
    """
    Imprime as equações dos modelos ajustados.

    Args:
        poly2 (np.poly1d): Polinômio grau 2.
        poly3 (np.poly1d): Polinômio grau 3.
        params_exp (tuple): Parâmetros exponencial.
        params_hip (tuple): Parâmetros hiperbólico.
        params_geo (tuple): Parâmetros geométrico.
        coeficientes (dict): Coeficientes de determinação para cada modelo
    """

    print('1) Modelo Polinomial Grau 2:')
    print(poly2)
    print(f'R² Polinomial Grau 2: {coeficientes["poly2"]:.4f}')

    print('\n2) Modelo Polinomial Grau 3:')
    print(poly3)
    print(f'R² Polinomial Grau 3: {coeficientes["poly3"]:.4f}')

    print('\n3) Modelo Exponencial:')
    print(f'y = {params_exp[0]:.2f} * exp({params_exp[1]:.5f} * x)')
    print(f'R² Exponencial: {coeficientes["exp"]:.4f}')

    print('\n4) Modelo Hiperbólico:')
    print(f'y = {params_hip[0]:.2f} / (x + {params_hip[1]:.5f})')
    print(f'R² Hipebólica: {coeficientes["hip"]:.4f}')

    print('\n5) Modelo Geométrico (Potencial):')
    print(f'y = {params_geo[0]:.2f} * x^{params_geo[1]:.5f}')
    print(f'R² Geométrica: {coeficientes["geo"]:.4f}')

    melhor_modelo, maior_r2, nomes, estimativa = imprimir_melhor_modelo_e_estimativa(poly2, poly3, params_exp, params_hip, params_geo, coeficientes)
    salvar_resultados_em_arquivo(poly2, poly3, params_exp, params_hip, params_geo, coeficientes, melhor_modelo, maior_r2, nomes, estimativa)
