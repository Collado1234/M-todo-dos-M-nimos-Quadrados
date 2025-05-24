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
            p0=(max(y) * 10, 1000),
            bounds=([0, -1e6], [1e12, 1e6])
    )
    except RuntimeError:
        print("Falha no ajuste hiperbólico!")
        params_hip = (np.nan, np.nan)

    # ----- Geométrico -----
    x_geo = x.copy()
    x_geo[x_geo == 0] = 0.1

    params_geo, _ = curve_fit(modelo_geometrico, x_geo, y, p0=(10000, 1))

    return poly2, poly3, params_exp, params_hip, params_geo, x_geo



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


# -------------------
# Impressão dos modelos
# -------------------

def imprimir_modelos(poly2, poly3, params_exp, params_hip, params_geo):
    """
    Imprime as equações dos modelos ajustados.

    Args:
        poly2 (np.poly1d): Polinômio grau 2.
        poly3 (np.poly1d): Polinômio grau 3.
        params_exp (tuple): Parâmetros exponencial.
        params_hip (tuple): Parâmetros hiperbólico.
        params_geo (tuple): Parâmetros geométrico.
    """
    print('1) Modelo Polinomial Grau 2:')
    print(poly2)

    print('\n2) Modelo Polinomial Grau 3:')
    print(poly3)

    print('\n3) Modelo Exponencial:')
    print(f'y = {params_exp[0]:.2f} * exp({params_exp[1]:.5f} * x)')

    print('\n4) Modelo Hiperbólico:')
    print(f'y = {params_hip[0]:.2f} / (x + {params_hip[1]:.5f})')

    print('\n5) Modelo Geométrico (Potencial):')
    print(f'y = {params_geo[0]:.2f} * x^{params_geo[1]:.5f}')


