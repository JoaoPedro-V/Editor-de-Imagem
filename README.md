# PixelLab — Editor de Imagens

Editor didático de processamento digital de imagens desenvolvido em Python. A interface é construída com Gradio e as operações de imagem utilizam OpenCV e NumPy.

O projeto permite carregar uma imagem, aplicar técnicas de ajustes, filtros, transformações geométricas e segmentação/detecção, comparar a imagem original com o resultado e acompanhar as alterações.

## Tecnologias

- Python
- Gradio
- OpenCV
- NumPy
- Matplotlib
- Pillow

## Funcionalidades

- Ajustes de brilho, contraste, gama, saturação, negativo, escala de cinza, equalização e canais de cor.
- Filtros de suavização e realce, incluindo Gaussiano, Mediana, Bilateral, Sobel, Canny e Laplaciano.
- Transformações geométricas, como rotação, escala e espelhamento.
- Segmentação por limiarização fixa, Otsu, método adaptativo e faixa de cor HSV.
- Pré-visualização, confirmação e descarte de alterações.
- Desfazer, refazer e redefinir a imagem.
- Comparação entre imagem original e resultado, histograma RGB, detalhes e histórico.
- Exportação da imagem processada em PNG.

> O PixelLab é um editor de processamento digital de imagens. A segmentação HSV isola faixas de cor escolhidas pelo usuário; ela não faz reconhecimento semântico ou classificação automática de objetos.

## Como executar no Google Colab

1. Abra `PixelLab_Editor_de_Imagens_responsivo.ipynb` no Google Colab.
2. Execute as células em ordem. A primeira célula instala as dependências.
3. Na última célula, o Gradio inicia a aplicação e apresenta um link temporário para abrir o editor.

O notebook contém o código da aplicação em uma célula e não depende do arquivo Python separado para funcionar no Colab.

## Como executar localmente

Requer Python 3.10 ou superior.

```bash
python -m venv .venv
```

Ative o ambiente virtual e instale as dependências:

```bash
pip install -r requirements.txt
```

Inicie a aplicação:

```bash
python pixellab_app_responsivo.py
```

Por padrão, a execução local não cria um link público. Para compartilhar intencionalmente a interface, defina `PIXELLAB_SHARE=1` antes de executar.

## Testes automatizados

Com as dependências instaladas, execute:

```bash
python -m unittest -v test_pixellab_responsivo.py
```

A suíte contém testes para operações de processamento, estados de edição, pré-visualização, histórico, exportação e estrutura da interface.

## Estrutura do repositório

```text
PixelLab-Editor-de-Imagens/
├── PixelLab_Editor_de_Imagens_responsivo.ipynb
├── pixellab_app_responsivo.py
├── requirements.txt
├── test_pixellab_responsivo.py
├── README.md
├── .gitignore
└── docs/
    ├── GUIA_TECNICO.md
    └── RELATORIO_LAYOUT_RESPONSIVO.md
```

## Observações

- As imagens e o histórico são mantidos na sessão de execução; o histórico não é persistido automaticamente em disco.
- O link público do Gradio é temporário e só permanece disponível enquanto a sessão estiver ativa.
- O comportamento visual deve ser conferido no navegador em diferentes resoluções. Os testes automatizados não substituem testes manuais de interface.

## Contexto acadêmico

Projeto acadêmico para demonstrar técnicas de processamento digital de imagens, manipulação de pixels e análise dos resultados por meio de uma interface interativa.
