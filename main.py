import min_quadrados as mq

# -------------------
# Função principal
# -------------------

def main():
    """
    Função principal do script.
    Executa o carregamento dos dados, ajuste dos modelos, geração dos gráficos e impressão dos modelos.
    """
    # 1. Carregar dados
    anos, x, y = mq.carregar_dados_arquivo("Populacao_PresidentePrudente.dat")



    # 2. Ajustar modelos
    poly2, poly3, params_exp, params_hip, params_geo, x_geo, coeficientes = mq.ajustar_modelos(x, y)

    # 3. Plotar modelos
    mq.plotar_polinomio(anos, x, y, poly2, grau=2, cor='blue')
    mq.plotar_polinomio(anos, x, y, poly3, grau=3, cor='red')

    mq.plotar_modelo(anos, x, y, mq.modelo_exponencial, params_exp, 'Exponencial', 'green')
    mq.plotar_modelo(anos, x, y, mq.modelo_hiperbolico, params_hip, 'Hiperbólico', 'purple')
    mq.plotar_modelo(anos, x, y, mq.modelo_geometrico, params_geo, 'Geométrico (Potencial)', 'orange', x_model=x_geo)

    # 4. Imprimir modelos encontrados
    mq.imprimir_modelos(poly2, poly3, params_exp, params_hip, params_geo, coeficientes)


# -------------------
# Execução do script
# -------------------

if __name__ == "__main__":
    main()
