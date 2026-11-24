# Guia técnico e roteiro de apresentação — PixelLab

## 1. Objetivo

Demonstrar técnicas de processamento digital de imagens manipulando os pixels de uma imagem escolhida pelo usuário e tornando os parâmetros acessíveis por uma interface gráfica.

## 2. Pipeline da aplicação

1. **Entrada:** Gradio recebe a imagem e a normaliza para uma matriz NumPy RGB (`uint8`).
2. **Escolha do método:** a interface recolhe a operação e os parâmetros.
3. **Processamento:** o callback chama a função Python correspondente e executa o algoritmo com OpenCV/NumPy.
4. **Visualização:** o usuário observa resultado, uma etapa intermediária, métricas e histograma RGB.
5. **Iteração:** o estado processado pode receber outra operação; o histórico permite desfazer/refazer.
6. **Saída:** Pillow exporta o estado atual para PNG.

## 3. O que explicar de cada técnica

| Técnica | Conceito que o grupo deve saber explicar | Parâmetros/decisões |
|---|---|---|
| Brilho e contraste | Transformação linear aplicada a cada canal | β desloca a intensidade; α amplia/reduz diferenças |
| Gama | Transformação não linear por tabela de consulta (LUT) | γ altera a distribuição dos tons; γ menor que 1 tende a clarear com a fórmula usada |
| Negativo | Inversão dos níveis de intensidade | Cada canal vira `255 − pixel` |
| Escala de cinza | Conversão RGB para luminância | Usa `cv2.COLOR_RGB2GRAY` |
| Equalização | Redistribuição de níveis do histograma | Equaliza a luminância no YCrCb para reduzir alteração de matiz |
| Gaussiano | Suavização por convolução com pesos gaussianos | Janela ímpar e sigma |
| Mediana | Filtro não linear que usa a mediana da vizinhança | Tamanho ímpar da janela |
| Bilateral | Suavização considerando distância espacial e diferença de cor | Diâmetro e sigma |
| Sobel | Estimativa dos gradientes horizontal e vertical | Magnitude dos gradientes |
| Canny | Sucessão de etapas para obter bordas finas | Limiar inferior e superior |
| Laplaciano | Segunda derivada espacial para realçar mudanças de intensidade | Sensível a ruído |
| Rotação/escala | Transformação geométrica por matriz afim | Ângulo e fator de escala |
| Limiar fixo | Separa pixels pelo cruzamento de um valor escolhido | T em `[0,255]` |
| Otsu | Estima automaticamente um limiar global pelo histograma | Não exige T manual |
| Adaptativa | Calcula limiares locais | Tamanho de vizinhança ímpar |
| HSV | Cria máscara em matiz/saturação/valor | Faixa H e mínimos S/V |

## 4. Plano de validação para a entrega

Usem exemplos diferentes dos utilizados durante a programação e preencham esta tabela depois de observar os resultados reais.

| Caso | Entrada | Técnica/parâmetros | Resultado observado | Falha/limitação identificada |
|---|---|---|---|---|
| 1 | Fotografia colorida com boa iluminação | Brilho/contraste e histograma | **Preencher após testar** | **Preencher** |
| 2 | Imagem com ruído visível | Gaussiano vs. mediana | **Preencher após testar** | **Preencher** |
| 3 | Imagem com bordas definidas | Sobel vs. Canny | **Preencher após testar** | **Preencher** |
| 4 | Imagem com objeto contrastante do fundo | Otsu vs. adaptativa | **Preencher após testar** | **Preencher** |
| 5 | Imagem com objeto colorido | Segmentação HSV | **Preencher após testar** | **Preencher** |
| 6 | Imagem não usada para escolher parâmetros | Combinação de 2 operações | **Preencher após testar** | **Preencher** |

Não atribuam percentuais de acerto sem definir uma métrica apropriada e medir os resultados. Para este editor, é útil comparar visualmente o antes/depois, os histogramas, a presença de artefatos e o comportamento de cada algoritmo quando os parâmetros mudam.

## 5. Roteiro sugerido para apresentação de 20–30 minutos

1. **Problema e escopo (3 min):** por que um editor didático é apropriado para demonstrar processamento de pixels.
2. **Arquitetura (3 min):** Colab, Gradio, OpenCV, NumPy, Matplotlib e Pillow.
3. **Ajustes e filtros (6 min):** demonstrar brilho/contraste, gama, equalização, suavização e detecção de bordas.
4. **Transformações e segmentação (5 min):** mudar parâmetros ao vivo e explicar as etapas intermediárias.
5. **Validação crítica (4 min):** mostrar exemplos com bons e maus resultados, além dos testes realizados.
6. **Código e participação (4–6 min):** cada integrante explica a função ou técnica pela qual contribuiu.

## 6. Limitações que merecem ser discutidas

- A imagem grande pode ser reduzida para manter responsividade; o arquivo exportado reflete o estado processado da sessão.
- Otsu costuma funcionar melhor com separação tonal clara; iluminação irregular pode prejudicar um único limiar global.
- HSV não identifica objetos: apenas isola faixas de cor escolhidas pelo usuário.
- Filtros de suavização trocam ruído por perda de detalhe; nitidez pode acentuar ruído; detecção de borda depende dos parâmetros.
- As técnicas não são um diagnóstico médico, reconhecimento facial, OCR ou classificador semântico. A proposta entregue é um editor de processamento de imagens.
