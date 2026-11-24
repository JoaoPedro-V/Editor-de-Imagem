"""PixelLab — Editor de Imagens
Aplicação didática de processamento digital de imagens com OpenCV, NumPy e Gradio.
Compatível com execução em notebook/Google Colab.
"""

from __future__ import annotations

import html
import os
import sys
import tempfile
import uuid
from datetime import datetime
from typing import Any

import cv2
import gradio as gr
import numpy as np
from matplotlib.figure import Figure
from PIL import Image

# Limite usado apenas para evitar que uma transformação geométrica gere uma
# imagem desproporcionalmente grande. Não interfere em operações comuns.
MAX_TRANSFORM_PIXELS = 12_000_000
MAX_HISTORY = 20
PREVIEW_DEBOUNCE_MS = 150
PREVIEW_DEBOUNCE_JS = f"(...args) => new Promise(resolve => {{ window.clearTimeout(window.__pixellabPreviewTimer); window.__pixellabPreviewTimer = window.setTimeout(() => resolve(args), {PREVIEW_DEBOUNCE_MS}); }})"
PREVIEW_CANCEL_JS = "(...args) => { window.clearTimeout(window.__pixellabPreviewTimer); return args; }"


def normalize_image(image: Any, max_side: int = 2800) -> np.ndarray | None:
    """Normaliza imagens recebidas do Gradio para uint8 no formato RGB."""
    if image is None:
        return None

    array = np.asarray(image)
    if array.size == 0:
        return None

    if array.ndim == 2:
        array = cv2.cvtColor(array.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    elif array.ndim == 3 and array.shape[2] == 4:
        # O tipo numpy do Gradio normalmente entrega RGB/RGBA.
        array = cv2.cvtColor(array.astype(np.uint8), cv2.COLOR_RGBA2RGB)
    elif array.ndim == 3 and array.shape[2] == 1:
        array = cv2.cvtColor(array[:, :, 0].astype(np.uint8), cv2.COLOR_GRAY2RGB)
    elif array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("Formato de imagem não suportado. Use uma imagem RGB ou em escala de cinza.")

    if array.dtype != np.uint8:
        if np.issubdtype(array.dtype, np.floating) and array.size and float(np.nanmax(array)) <= 1.0:
            array = array * 255.0
        array = np.nan_to_num(array, nan=0.0, posinf=255.0, neginf=0.0)
        array = np.clip(array, 0, 255).astype(np.uint8)
    else:
        array = array.copy()

    height, width = array.shape[:2]
    largest_side = max(height, width)
    if largest_side > max_side:
        ratio = max_side / largest_side
        new_size = (max(1, int(width * ratio)), max(1, int(height * ratio)))
        array = cv2.resize(array, new_size, interpolation=cv2.INTER_AREA)

    return np.ascontiguousarray(array)


def to_gray(rgb: np.ndarray) -> np.ndarray:
    """Converte RGB para escala de cinza usando a conversão correta do OpenCV."""
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def gray_as_rgb(gray: np.ndarray) -> np.ndarray:
    """Converte uma matriz de intensidade em uma imagem RGB exibível."""
    if gray.ndim == 3:
        gray = gray[:, :, 0]
    return cv2.cvtColor(gray.astype(np.uint8), cv2.COLOR_GRAY2RGB)


def apply_adjustment(
    image: np.ndarray,
    mode: str,
    brightness: float = 0,
    contrast: float = 1.0,
    gamma: float = 1.0,
    saturation: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, str, str]:
    """Aplica ajustes de pixels e retorna (resultado, etapa intermediária, rótulo, explicação)."""
    source = image.copy()
    gray = to_gray(source)

    if mode == "Brilho e contraste":
        # Transformação linear: novo_pixel = pixel * contraste + brilho.
        result = np.clip(source.astype(np.float32) * float(contrast) + float(brightness), 0, 255).astype(np.uint8)
        stage = gray_as_rgb(gray)
        explanation = f"Brilho {brightness:+.0f} e contraste ×{contrast:.2f}. A luminância em cinza é exibida como etapa de análise."
        label = f"Brilho e contraste (brilho={brightness:+.0f}, contraste={contrast:.2f})"

    elif mode == "Saturação":
        hsv = cv2.cvtColor(source, cv2.COLOR_RGB2HSV)
        saturation_factor = max(0.0, min(float(saturation), 3.0))
        hsv[:, :, 1] = np.clip(hsv[:, :, 1].astype(np.float32) * saturation_factor, 0, 255).astype(np.uint8)
        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        stage = gray_as_rgb(hsv[:, :, 1])
        explanation = f"Saturação multiplicada por {saturation_factor:.2f} no canal S do espaço HSV. Valores abaixo de 1 reduzem a intensidade das cores."
        label = f"Saturação (×{saturation_factor:.2f})"

    elif mode == "Correção de gama":
        safe_gamma = max(float(gamma), 0.01)
        lut = np.array([min(255, max(0, round(((value / 255.0) ** safe_gamma) * 255))) for value in range(256)], dtype=np.uint8)
        result = cv2.LUT(source, lut)
        stage = gray_as_rgb(gray)
        explanation = f"Correção de gama γ={safe_gamma:.2f} por tabela LUT de 256 níveis. Valores abaixo de 1 tendem a clarear a imagem."
        label = f"Correção de gama (γ={safe_gamma:.2f})"

    elif mode == "Negativo":
        result = 255 - source
        stage = source.copy()
        explanation = "Inversão de intensidade em cada canal RGB: pixel_novo = 255 − pixel_original."
        label = "Negativo"

    elif mode == "Escala de cinza":
        result = gray_as_rgb(gray)
        stage = result.copy()
        explanation = "Conversão de RGB para luminância monocromática com cv2.COLOR_RGB2GRAY."
        label = "Escala de cinza"

    elif mode == "Equalização do histograma":
        # Equalizar apenas o canal de luminância evita alterar tanto a tonalidade das cores.
        ycrcb = cv2.cvtColor(source, cv2.COLOR_RGB2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        result = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
        stage = gray_as_rgb(ycrcb[:, :, 0])
        explanation = "Equalização global do histograma no canal Y (luminância) do espaço YCrCb, preservando melhor a informação de cor."
        label = "Equalização do histograma"

    elif mode in ("Canal vermelho", "Canal verde", "Canal azul"):
        channel_index = {"Canal vermelho": 0, "Canal verde": 1, "Canal azul": 2}[mode]
        channel = source[:, :, channel_index]
        result = np.zeros_like(source)
        result[:, :, channel_index] = channel
        stage = gray_as_rgb(channel)
        explanation = f"Isolamento do canal {mode.removeprefix('Canal ').lower()} para visualizar sua contribuição por pixel."
        label = mode

    else:
        raise ValueError(f"Ajuste desconhecido: {mode}")

    return result, stage, label, explanation


def apply_filter(
    image: np.ndarray,
    mode: str,
    kernel_size: int = 5,
    sigma: float = 2.0,
    canny_low: int = 60,
    canny_high: int = 150,
) -> tuple[np.ndarray, np.ndarray, str, str]:
    """Aplica filtros espaciais, suavização ou detecção de bordas."""
    source = image.copy()
    gray = to_gray(source)
    kernel_size = int(max(3, min(31, kernel_size)))
    if kernel_size % 2 == 0:
        kernel_size += 1

    if mode == "Gaussiano":
        result = cv2.GaussianBlur(source, (kernel_size, kernel_size), float(sigma))
        stage = gray_as_rgb(gray)
        explanation = f"Suavização gaussiana com janela {kernel_size}×{kernel_size} e σ={sigma:.1f}; reduz ruído de alta frequência."
        label = f"Filtro Gaussiano ({kernel_size}×{kernel_size})"

    elif mode == "Mediana":
        result = cv2.medianBlur(source, kernel_size)
        stage = gray_as_rgb(gray)
        explanation = f"Filtro de mediana {kernel_size}×{kernel_size}; substitui o pixel central pela mediana da vizinhança."
        label = f"Filtro de mediana ({kernel_size}×{kernel_size})"

    elif mode == "Bilateral":
        result = cv2.bilateralFilter(source, kernel_size, float(sigma) * 20.0, float(sigma) * 20.0)
        stage = gray_as_rgb(gray)
        explanation = f"Filtro bilateral com diâmetro {kernel_size}; suaviza regiões mantendo melhor as bordas."
        label = f"Filtro bilateral (diâmetro={kernel_size})"

    elif mode == "Nitidez (sharpen)":
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        result = cv2.filter2D(source, -1, kernel)
        stage = gray_as_rgb(gray)
        explanation = "Convolução com máscara de realce que reforça diferenças locais de intensidade."
        label = "Nitidez (sharpen)"

    elif mode == "Sobel":
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(gx, gy)
        normalized = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        result = gray_as_rgb(normalized)
        stage = result.copy()
        explanation = "Gradientes horizontal e vertical calculados por Sobel e combinados em magnitude para destacar bordas."
        label = "Detecção de bordas (Sobel)"

    elif mode == "Canny":
        low = int(canny_low)
        high = max(int(canny_high), low + 1)
        edges = cv2.Canny(gray, low, high)
        result = gray_as_rgb(edges)
        stage = result.copy()
        explanation = f"Canny aplicado à imagem em cinza com limiares inferior={low} e superior={high}."
        label = f"Detecção de bordas (Canny, {low}/{high})"

    elif mode == "Laplaciano":
        lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
        absolute = cv2.convertScaleAbs(lap)
        result = gray_as_rgb(absolute)
        stage = result.copy()
        explanation = "Laplaciano de segunda derivada para realçar mudanças rápidas de intensidade."
        label = "Detecção de bordas (Laplaciano)"

    else:
        raise ValueError(f"Filtro desconhecido: {mode}")

    return result, stage, label, explanation


def apply_transform(
    image: np.ndarray,
    angle: float = 0,
    scale: float = 1.0,
    flip_mode: str = "Nenhum",
) -> tuple[np.ndarray, np.ndarray, str, str]:
    """Aplica rotação, escala e/ou espelhamento mantendo a imagem inteira no enquadramento."""
    source = image.copy()
    height, width = source.shape[:2]
    scale = max(0.1, min(float(scale), 3.0))
    angle = float(angle)

    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, scale)
    cos_value = abs(matrix[0, 0])
    sin_value = abs(matrix[0, 1])
    new_width = max(1, int((height * sin_value) + (width * cos_value)))
    new_height = max(1, int((height * cos_value) + (width * sin_value)))

    # Evita alocação excessiva quando a escala e a rotação são grandes.
    if new_width * new_height > MAX_TRANSFORM_PIXELS:
        reduction = (MAX_TRANSFORM_PIXELS / float(new_width * new_height)) ** 0.5
        scale *= reduction
        matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, scale)
        cos_value = abs(matrix[0, 0])
        sin_value = abs(matrix[0, 1])
        new_width = max(1, int((height * sin_value) + (width * cos_value)))
        new_height = max(1, int((height * cos_value) + (width * sin_value)))

    matrix[0, 2] += (new_width / 2.0) - (width / 2.0)
    matrix[1, 2] += (new_height / 2.0) - (height / 2.0)
    result = cv2.warpAffine(source, matrix, (new_width, new_height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    if flip_mode == "Horizontal":
        result = cv2.flip(result, 1)
    elif flip_mode == "Vertical":
        result = cv2.flip(result, 0)
    elif flip_mode == "Horizontal e vertical":
        result = cv2.flip(result, -1)

    stage = source.copy()
    details = f"rotação={angle:.0f}°, escala solicitada={scale:.2f}×, espelhamento={flip_mode.lower()}"
    label = f"Transformação geométrica ({details})"
    explanation = "Rotação/escala por matriz afim, com ajuste dos limites para preservar a imagem; espelhamento aplicado depois da transformação."
    return result, stage, label, explanation


def apply_segmentation(
    image: np.ndarray,
    mode: str,
    threshold: int = 127,
    block_size: int = 21,
    h_min: int = 0,
    h_max: int = 179,
    s_min: int = 50,
    v_min: int = 50,
) -> tuple[np.ndarray, np.ndarray, str, str]:
    """Segmenta por limiarização global, Otsu, método adaptativo ou faixa de cor HSV."""
    source = image.copy()
    gray = to_gray(source)

    if mode == "Limiarização fixa":
        _, mask = cv2.threshold(gray, int(threshold), 255, cv2.THRESH_BINARY)
        result = gray_as_rgb(mask)
        stage = result.copy()
        label = f"Segmentação por limiar fixo (T={int(threshold)})"
        explanation = f"Cada pixel é classificado em 0 ou 255 conforme a intensidade cruza o limiar T={int(threshold)}."

    elif mode == "Otsu":
        used_threshold, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        result = gray_as_rgb(mask)
        stage = result.copy()
        label = f"Segmentação por Otsu (T={used_threshold:.0f})"
        explanation = f"O método de Otsu estima automaticamente o limiar global a partir do histograma; limiar calculado={used_threshold:.0f}."

    elif mode == "Adaptativa":
        block_size = max(3, min(99, int(block_size)))
        if block_size % 2 == 0:
            block_size += 1
        mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, 2)
        result = gray_as_rgb(mask)
        stage = result.copy()
        label = f"Segmentação adaptativa (bloco={block_size})"
        explanation = f"A imagem é binarizada por limiares locais, usando vizinhança gaussiana de {block_size}×{block_size}; útil quando a iluminação varia."

    elif mode == "Cor (HSV)":
        hsv = cv2.cvtColor(source, cv2.COLOR_RGB2HSV)
        h_min = int(np.clip(h_min, 0, 179))
        h_max = int(np.clip(h_max, 0, 179))
        s_min = int(np.clip(s_min, 0, 255))
        v_min = int(np.clip(v_min, 0, 255))
        if h_min <= h_max:
            mask = cv2.inRange(hsv, np.array([h_min, s_min, v_min], dtype=np.uint8), np.array([h_max, 255, 255], dtype=np.uint8))
        else:
            # Permite intervalos de matiz que atravessam o limite 179 -> 0.
            mask_a = cv2.inRange(hsv, np.array([h_min, s_min, v_min], dtype=np.uint8), np.array([179, 255, 255], dtype=np.uint8))
            mask_b = cv2.inRange(hsv, np.array([0, s_min, v_min], dtype=np.uint8), np.array([h_max, 255, 255], dtype=np.uint8))
            mask = cv2.bitwise_or(mask_a, mask_b)
        result = cv2.bitwise_and(source, source, mask=mask)
        stage = gray_as_rgb(mask)
        label = f"Segmentação por cor HSV (H={h_min}–{h_max}, S≥{s_min}, V≥{v_min})"
        explanation = "O espaço HSV separa matiz (H), saturação (S) e valor (V); a máscara isola pixels dentro da faixa escolhida."

    else:
        raise ValueError(f"Método de segmentação desconhecido: {mode}")

    return result, stage, label, explanation


def process_image(image: np.ndarray, operation: str, params: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, str, str]:
    """Roteia a operação selecionada para a função de processamento correspondente."""
    if operation == "adjust":
        return apply_adjustment(
            image,
            mode=params.get("mode", "Brilho e contraste"),
            brightness=params.get("brightness", 0),
            contrast=params.get("contrast", 1.0),
            gamma=params.get("gamma", 1.0),
            saturation=params.get("saturation", 1.0),
        )
    if operation == "filter":
        return apply_filter(
            image,
            mode=params.get("mode", "Gaussiano"),
            kernel_size=params.get("kernel_size", 5),
            sigma=params.get("sigma", 2.0),
            canny_low=params.get("canny_low", 60),
            canny_high=params.get("canny_high", 150),
        )
    if operation == "transform":
        return apply_transform(
            image,
            angle=params.get("angle", 0),
            scale=params.get("scale", 1.0),
            flip_mode=params.get("flip_mode", "Nenhum"),
        )
    if operation == "segment":
        segment_mode = params.get("mode", "Otsu")
        if segment_mode == "Canny":
            # Reutiliza a implementação Canny já existente, mantendo os mesmos limiares.
            return apply_filter(
                image,
                "Canny",
                canny_low=params.get("canny_low", 60),
                canny_high=params.get("canny_high", 150),
            )
        return apply_segmentation(
            image,
            mode=segment_mode,
            threshold=params.get("threshold", 127),
            block_size=params.get("block_size", 21),
            h_min=params.get("h_min", 0),
            h_max=params.get("h_max", 179),
            s_min=params.get("s_min", 50),
            v_min=params.get("v_min", 50),
        )
    if operation in ("Negativo", "Escala de cinza", "Equalização do histograma", "Canal vermelho", "Canal verde", "Canal azul", "Correção de gama", "Brilho e contraste", "Saturação"):
        return apply_adjustment(image, operation)
    raise ValueError(f"Operação não reconhecida: {operation}")


def make_histogram(image: np.ndarray | None) -> Figure:
    """Cria um histograma RGB de 256 níveis usando Matplotlib."""
    fig = Figure(figsize=(6.7, 2.5), dpi=100, facecolor="#101a2c")
    ax = fig.add_subplot(111)
    ax.set_facecolor("#101a2c")

    if image is not None:
        colors = [(0, "#ff6b6b", "Vermelho"), (1, "#51cf66", "Verde"), (2, "#4dabf7", "Azul")]
        bins = np.arange(256)
        for channel, color, label in colors:
            histogram = cv2.calcHist([image], [channel], None, [256], [0, 256]).ravel()
            ax.plot(bins, histogram, color=color, linewidth=1.0, alpha=0.9, label=label)
        ax.legend(loc="upper right", frameon=False, fontsize=7, labelcolor="#cbd5e1", ncol=3)
    else:
        ax.text(0.5, 0.5, "Carregue uma imagem para ver o histograma", color="#94a3b8", ha="center", va="center", transform=ax.transAxes)

    ax.set_xlim(0, 255)
    ax.set_xlabel("Intensidade do pixel", color="#94a3b8", fontsize=8)
    ax.set_ylabel("Quantidade", color="#94a3b8", fontsize=8)
    ax.tick_params(colors="#94a3b8", labelsize=7)
    ax.grid(True, alpha=0.12, color="#94a3b8")
    for spine in ax.spines.values():
        spine.set_color("#334155")
    fig.tight_layout(pad=1.2)
    return fig


def _image_metrics(image: np.ndarray | None) -> str:
    if image is None:
        return "Sem imagem"
    height, width = image.shape[:2]
    mean_intensity = float(np.mean(to_gray(image)))
    return f"{width} × {height}px · RGB · luminância média {mean_intensity:.1f}/255"


def make_details_html(original: np.ndarray | None, result: np.ndarray | None, explanation: str = "") -> str:
    """Monta os dados laterais com informações calculadas da imagem, não simuladas."""
    original_text = html.escape(_image_metrics(original))
    result_text = html.escape(_image_metrics(result))
    explanation_text = html.escape(explanation or "Aguardando operação.")
    return f"""
    <div class="pl-detail-card">
      <div class="pl-detail-title">DETALHES DA IMAGEM</div>
      <div class="pl-metric"><span>ORIGINAL</span><strong>{original_text}</strong></div>
      <div class="pl-metric"><span>RESULTADO</span><strong>{result_text}</strong></div>
      <div class="pl-divider"></div>
      <div class="pl-detail-title">ETAPA / ALGORITMO</div>
      <p class="pl-explanation">{explanation_text}</p>
    </div>
    """


def history_to_text(history: list[str] | None) -> str:
    if not history:
        return "Nenhuma operação executada ainda."
    return "\n".join(f"{index + 1:02d}. {item}" for index, item in enumerate(history[-12:]))


def make_status(message: str, kind: str = "success") -> str:
    safe_message = html.escape(message)
    class_name = "pl-status pl-status-error" if kind == "error" else "pl-status"
    icon = "⚠" if kind == "error" else "✓"
    return f'<div class="{class_name}"><span class="pl-status-icon">{icon}</span>{safe_message}</div>'


def build_view_outputs(
    original: np.ndarray | None,
    result: np.ndarray | None,
    intermediate: np.ndarray | None,
    current: np.ndarray | None,
    undo_stack: list[np.ndarray],
    redo_stack: list[np.ndarray],
    history: list[str],
    explanation: str,
    status: str,
    status_kind: str = "success",
    preview_payload: dict[str, Any] | None = None,
):
    return (
        result,
        intermediate,
        make_histogram(result),
        make_details_html(original, result, explanation),
        current,
        undo_stack,
        redo_stack,
        history,
        history_to_text(history),
        make_status(status, status_kind),
        preview_payload,
        gr.update(visible=(result is None)),
    )


def load_image(image: Any):
    """Inicializa o estado da sessão após o usuário carregar uma imagem."""
    try:
        normalized = normalize_image(image)
        if normalized is None:
            return (None, None, [], [], [], None, None, make_histogram(None), make_details_html(None, None), history_to_text([]), make_status("Selecione uma imagem para começar.", "error"), None, gr.update(visible=True))
        history = ["Imagem carregada"]
        message = "Imagem carregada e pronta para processamento."
        if np.asarray(image).shape[:2] != normalized.shape[:2]:
            message += " Imagem grande redimensionada para manter a interface responsiva."
        return (
            normalized,
            normalized.copy(),
            [],
            [],
            history,
            normalized.copy(),
            normalized.copy(),
            make_histogram(normalized),
            make_details_html(normalized, normalized, "A imagem foi normalizada para RGB e está pronta para receber operações."),
            history_to_text(history),
            make_status(message),
            None,
            gr.update(visible=False),
        )
    except Exception as exc:
        return (None, None, [], [], [], None, None, make_histogram(None), make_details_html(None, None), history_to_text([]), make_status(f"Não foi possível abrir a imagem: {exc}", "error"), None, gr.update(visible=True))


def make_process_callback(operation: str):
    """Cria callbacks reutilizáveis para os diferentes painéis de operações."""
    def callback(original, current, undo_stack, redo_stack, history, *parameters):
        original = normalize_image(original)
        current = normalize_image(current)
        base = current if current is not None else original
        undo_stack = list(undo_stack or [])
        redo_stack = list(redo_stack or [])
        history = list(history or [])

        if base is None:
            return build_view_outputs(original, None, None, current, undo_stack, redo_stack, history, "Carregue uma imagem antes de executar operações.", "Nenhuma imagem foi carregada.", "error")

        # Cada aba recebe apenas os parâmetros relevantes para sua operação.
        if operation == "adjust":
            names = ["mode", "brightness", "contrast", "gamma", "saturation"]
        elif operation == "filter":
            names = ["mode", "kernel_size", "sigma"]
        elif operation == "transform":
            names = ["angle", "scale", "flip_mode"]
        elif operation == "segment":
            names = ["mode", "threshold", "block_size", "h_min", "h_max", "s_min", "v_min", "canny_low", "canny_high"]
        else:
            names = []

        params = dict(zip(names, parameters)) if names else {}
        if not names:
            params = {}
        try:
            result, intermediate, label, explanation = process_image(base, operation, params)
            undo_stack.append(base.copy())
            undo_stack = undo_stack[-MAX_HISTORY:]
            redo_stack = []
            history.append(label)
            if len(history) > 50:
                history = history[-50:]
            return build_view_outputs(original, result, intermediate, result.copy(), undo_stack, redo_stack, history, explanation, f"Operação aplicada: {label}.")
        except Exception as exc:
            return build_view_outputs(original, base, base.copy(), base.copy(), undo_stack, redo_stack, history, "A operação não foi concluída; veja a mensagem abaixo.", f"Erro ao processar imagem: {exc}", "error")

    return callback


def make_preview_callback(operation: str):
    """Cria uma prévia temporária sempre calculada a partir da imagem confirmada atual."""
    def callback(original, current, *parameters):
        base = normalize_image(current)
        if base is None:
            base = normalize_image(original)
        if base is None:
            empty = None
            return empty, empty, make_histogram(None), make_details_html(None, None), None, make_status("Carregue uma imagem antes de visualizar alterações.", "error")

        names_by_operation = {
            "adjust": ["mode", "brightness", "contrast", "gamma", "saturation"],
            "filter": ["mode", "kernel_size", "sigma"],
            "segment": ["mode", "threshold", "block_size", "h_min", "h_max", "s_min", "v_min", "canny_low", "canny_high"],
        }
        params = dict(zip(names_by_operation[operation], parameters))
        mode = params.get("mode")
        supported = {
            "adjust": {"Brilho e contraste", "Correção de gama", "Saturação"},
            "filter": {"Gaussiano", "Mediana", "Bilateral", "Nitidez (sharpen)"},
            "segment": {"Limiarização fixa", "Adaptativa", "Cor (HSV)", "Canny"},
        }
        if mode not in supported[operation]:
            return base, base.copy(), make_histogram(base), make_details_html(original, base, "A ferramenta selecionada não utiliza prévia ao vivo."), None, make_status("Use Aplicar para executar esta operação.")

        try:
            result, intermediate, label, explanation = process_image(base, operation, params)
            payload = {
                "result": result,
                "intermediate": intermediate,
                "label": label,
                "explanation": explanation,
                "operation": operation,
            }
            return (
                result,
                intermediate,
                make_histogram(result),
                make_details_html(original, result, explanation),
                payload,
                make_status(f"Prévia: {label}. Confirme para incorporar ou descarte para voltar ao estado anterior."),
            )
        except Exception as exc:
            return base, base.copy(), make_histogram(base), make_details_html(original, base), None, make_status(f"Falha ao gerar prévia: {exc}", "error")
    return callback


def confirm_preview(original, current, undo_stack, redo_stack, history, preview_payload):
    """Incorpora a prévia ao estado de trabalho e registra a operação no histórico."""
    original = normalize_image(original)
    current = normalize_image(current)
    undo_stack = list(undo_stack or [])
    redo_stack = list(redo_stack or [])
    history = list(history or [])
    if not preview_payload or preview_payload.get("result") is None:
        base = current if current is not None else original
        return build_view_outputs(original, base, base, base, undo_stack, redo_stack, history, "Não há prévia pendente para confirmar.", "Ajuste um parâmetro para gerar uma prévia.", "error")
    if current is None:
        current = original
    if current is not None:
        undo_stack.append(current.copy())
        undo_stack = undo_stack[-MAX_HISTORY:]
    result = normalize_image(preview_payload["result"])
    intermediate = normalize_image(preview_payload.get("intermediate"))
    label = str(preview_payload.get("label", "Operação confirmada"))
    explanation = str(preview_payload.get("explanation", ""))
    history.append(label)
    history = history[-50:]
    return build_view_outputs(original, result, intermediate, result.copy(), undo_stack, [], history, explanation, f"Prévia confirmada: {label}.")


def cancel_preview(original, current, undo_stack, redo_stack, history):
    """Descarta a prévia e volta a exibir o último estado confirmado."""
    original = normalize_image(original)
    current = normalize_image(current)
    undo_stack = list(undo_stack or [])
    redo_stack = list(redo_stack or [])
    history = list(history or [])
    base = current if current is not None else original
    return build_view_outputs(original, base, base.copy() if base is not None else None, base, undo_stack, redo_stack, history, "Prévia descartada. O estado confirmado foi mantido.", "Prévia descartada.")


def make_quick_callback(mode: str):
    """Atalho para botões de operações frequentes na lateral."""
    def callback(original, current, undo_stack, redo_stack, history):
        return make_process_callback(mode)(original, current, undo_stack, redo_stack, history)
    return callback


def undo_operation(original, current, undo_stack, redo_stack, history):
    original = normalize_image(original)
    current = normalize_image(current)
    undo_stack = list(undo_stack or [])
    redo_stack = list(redo_stack or [])
    history = list(history or [])
    if current is None or not undo_stack:
        base = current if current is not None else original
        return build_view_outputs(original, base, base, base, undo_stack, redo_stack, history, "Não há operação anterior para desfazer.", "O histórico de desfazer está vazio.", "error")
    restored = undo_stack.pop().copy()
    redo_stack.append(current.copy())
    history.append("Desfazer última operação")
    return build_view_outputs(original, restored, restored.copy(), restored.copy(), undo_stack, redo_stack, history, "Imagem restaurada para o estado anterior.", "Última operação desfeita.")


def redo_operation(original, current, undo_stack, redo_stack, history):
    original = normalize_image(original)
    current = normalize_image(current)
    undo_stack = list(undo_stack or [])
    redo_stack = list(redo_stack or [])
    history = list(history or [])
    if current is None or not redo_stack:
        base = current if current is not None else original
        return build_view_outputs(original, base, base, base, undo_stack, redo_stack, history, "Não há operação para refazer.", "O histórico de refazer está vazio.", "error")
    restored = redo_stack.pop().copy()
    undo_stack.append(current.copy())
    history.append("Refazer operação")
    return build_view_outputs(original, restored, restored.copy(), restored.copy(), undo_stack, redo_stack, history, "Imagem restaurada para o estado seguinte.", "Operação refeita.")


def restore_original(original, current, undo_stack, redo_stack, history):
    original = normalize_image(original)
    current = normalize_image(current)
    undo_stack = list(undo_stack or [])
    redo_stack = list(redo_stack or [])
    history = list(history or [])
    if original is None:
        return build_view_outputs(None, None, None, None, undo_stack, redo_stack, history, "Carregue uma imagem antes de restaurar.", "Nenhuma imagem foi carregada.", "error")
    if current is not None:
        undo_stack.append(current.copy())
        undo_stack = undo_stack[-MAX_HISTORY:]
    restored = original.copy()
    redo_stack = []
    history.append("Restaurar imagem original")
    return build_view_outputs(original, restored, restored.copy(), restored.copy(), undo_stack, redo_stack, history, "Estado original restaurado.", "Imagem original restaurada.")


def export_image(current):
    current = normalize_image(current)
    if current is None:
        return None, make_status("Carregue e processe uma imagem antes de exportar.", "error")
    try:
        handle, filepath = tempfile.mkstemp(prefix=f"pixellab_{uuid.uuid4().hex[:8]}_", suffix=".png")
        os.close(handle)
        Image.fromarray(current).save(filepath, format="PNG", optimize=True)
        return filepath, make_status(f"Resultado exportado em PNG ({current.shape[1]} × {current.shape[0]} px).")
    except Exception as exc:
        return None, make_status(f"Falha ao exportar a imagem: {exc}", "error")


def export_image_for_ui(current):
    """Callback de exportação: exibe o link apenas quando um PNG foi criado."""
    filepath, status = export_image(current)
    return gr.update(value=filepath, visible=bool(filepath)), status


CSS = r"""
:root {
  --pl-bg: #0b1220;
  --pl-panel: #111c2c;
  --pl-panel-soft: #152237;
  --pl-border: #293a52;
  --pl-text: #edf5ff;
  --pl-muted: #9aacc2;
  --pl-cyan: #43d5ed;
  --pl-blue: #398be5;
}
.gradio-container {
  width: 100% !important;
  max-width: none !important;
  min-height: 100vh !important;
  margin: 0 !important;
  padding: 10px 12px 18px !important;
  box-sizing:border-box !important;
  background: var(--pl-bg) !important;
  color: var(--pl-text) !important;
  font-family: Inter, ui-sans-serif, system-ui, sans-serif !important;
}
body { background: var(--pl-bg) !important; }
footer { display: none !important; }
#pl-header {
  padding: 14px 18px;
  border: 1px solid var(--pl-border);
  border-radius: 16px;
  margin-bottom: 16px;
  background: linear-gradient(115deg, #111e32 0%, #101a2a 70%, #102437 100%);
}
.pl-header-row { display:flex; align-items:center; justify-content:space-between; gap:16px; }
.pl-brand { display:flex; align-items:center; gap:12px; min-width:0; }
.pl-logo {
  display:flex; align-items:center; justify-content:center; width:42px; height:42px; flex:0 0 42px;
  border-radius:12px; background:linear-gradient(145deg,#48d9ed,#378ce8); color:#071321;
  font-size:25px; font-weight:900;
}
.pl-title { font-size:24px; line-height:1.15; font-weight:800; letter-spacing:-.5px; }
.pl-subtitle { color:var(--pl-muted); font-size:12px; margin-top:3px; }
.pl-status-chip { border:1px solid #31526a; background:#122b3c; color:#83e4f3; padding:7px 10px; border-radius:999px; font-size:10px; letter-spacing:.5px; white-space:nowrap; }
.pl-tagline { color:#b3c1d3; font-size:12px; margin-top:10px; }
.pl-section-label { color:#7f9ab6; font-size:10px; font-weight:800; letter-spacing:1.2px; margin:2px 0 8px; text-transform:uppercase; }
.pl-footnote { color:var(--pl-muted); font-size:11px; line-height:1.45; }
/* Layout desktop: menu e inspetor usam larguras fluidas; o canvas recebe o espaço restante. */
#pl-workspace {
  display:grid !important;
  grid-template-columns:clamp(150px, 11vw, 184px) minmax(0, 1fr) clamp(285px, 22vw, 350px) !important;
  align-items:start !important;
  gap:clamp(10px, 1vw, 16px) !important;
  width:100% !important; min-width:0 !important; min-height:0 !important; height:auto !important;
}
#pl-workspace > * {
  min-width:0 !important; max-width:100% !important; width:100% !important;
  box-sizing:border-box !important;
}
#pl-nav-panel {
  grid-column:1; min-width:0 !important; width:100% !important;
  padding:14px 11px !important; border:1px solid var(--pl-border) !important; border-radius:15px !important;
  background:linear-gradient(180deg,#111d30,#0f1928) !important; gap:12px !important;
}
#pl-canvas-column {
  grid-column:2; min-width:0 !important; width:100% !important; max-width:100% !important;
  padding:2px 2px 0 !important; gap:10px !important; overflow:visible !important;
}
#pl-tools-panel {
  grid-column:3; min-width:0 !important; width:100% !important; max-width:100% !important;
  padding:14px !important; border:1px solid var(--pl-border) !important; border-radius:15px !important;
  background:linear-gradient(180deg,#111d30,#0f1928) !important; gap:12px !important;
  max-height:calc(100vh - 150px) !important; overflow-y:auto !important; overflow-x:hidden !important;
  box-sizing:border-box !important;
}
#pl-tools-panel, #pl-tools-panel *, #pl-tools-panel *::before, #pl-tools-panel *::after {
  min-width:0 !important; max-width:100% !important; box-sizing:border-box !important;
}
#pl-tools-panel input, #pl-tools-panel textarea, #pl-tools-panel select, #pl-tools-panel button {
  max-width:100% !important; box-sizing:border-box !important;
}
#pl-tools-panel label, #pl-tools-panel p, #pl-tools-panel .label-wrap,
#pl-tools-panel .info, #pl-tools-panel span {
  overflow-wrap:anywhere !important;
}
#pl-image-pair { display:flex !important; gap:clamp(9px, 1vw, 14px) !important; flex:1 1 auto !important; min-width:0 !important; min-height:0 !important; }
#pl-image-pair > div { flex:1 1 0 !important; min-width:0 !important; max-width:100% !important; }
#pl-original-image, #pl-result-image { border:1px solid var(--pl-border) !important; border-radius:13px !important; overflow:hidden !important; min-width:0 !important; max-width:100% !important; box-sizing:border-box !important; }
#pl-original-image .image-container, #pl-result-image .image-container {
  height:clamp(340px, 56vh, 620px) !important; min-height:340px !important; max-width:100% !important;
  background:#070d17 !important; display:flex !important; align-items:center !important; justify-content:center !important;
}
#pl-original-image img, #pl-result-image img { object-fit:contain !important; width:auto !important; max-width:100% !important; max-height:100% !important; }
#pl-original-image .upload-container { height:clamp(340px, 56vh, 620px) !important; min-height:340px !important; max-width:100% !important; }
#pl-nav-panel .pl-section-label { margin-bottom:2px; }
#pl-tool-nav fieldset { display:flex !important; flex-direction:column !important; gap:8px !important; }
#pl-tool-nav label { min-height:58px !important; display:flex !important; align-items:center !important; padding:10px 9px !important; }
#pl-tool-nav label span { font-size:12px !important; line-height:1.25 !important; }
#pl-tool-nav legend { display:none !important; }
#pl-original-image button { background:linear-gradient(100deg,#43d5ed,#398be5) !important; color:#061321 !important; font-weight:800 !important; border:0 !important; }
#pl-action-row { gap:8px !important; }
#pl-status { margin-top:2px; }
.pl-status { display:flex; gap:9px; align-items:flex-start; padding:10px 12px; border:1px solid #245346; border-radius:10px; background:#102a27; color:#a7f3d0; font-size:12px; line-height:1.4; }
.pl-status-error { border-color:#743b49; background:#321923; color:#fecdd3; }
.pl-status-icon { font-weight:900; }
.pl-detail-card { color:var(--pl-text); font-size:12px; }
.pl-detail-title { color:#7f9ab6; font-size:10px; font-weight:800; letter-spacing:1px; margin-bottom:8px; }
.pl-metric { display:flex; flex-direction:column; gap:3px; margin:8px 0; }
.pl-metric span { color:#7f9ab6; font-size:9px; font-weight:800; letter-spacing:.7px; }
.pl-metric strong { color:#dce8f6; font-size:12px; font-weight:600; }
.pl-divider { border-top:1px solid var(--pl-border); margin:12px 0; }
.pl-explanation { color:#b7c5d7; line-height:1.5; margin:4px 0 0; }
#pl-tool-nav fieldset { display:flex !important; flex-direction:column !important; gap:6px !important; border:0 !important; padding:0 !important; }
#pl-tool-nav legend { color:#8fa6bf !important; font-size:11px !important; font-weight:700 !important; }
#pl-tool-nav label {
  border:1px solid #2b3d56 !important; border-radius:10px !important; padding:10px 12px !important;
  background:#111e30 !important; transition:background .15s ease,border-color .15s ease !important;
}
#pl-tool-nav label:has(input:checked) { border-color:#3dbbd7 !important; background:#123044 !important; box-shadow:inset 3px 0 0 #43d5ed !important; }
#pl-tool-nav label:hover { border-color:#4a6f8b !important; }
#pl-tool-nav input { accent-color:#43d5ed !important; }
#pl-tool-settings { border-top:1px solid var(--pl-border); padding-top:14px; }
.pl-tool-heading { color:#f1f7ff; font-size:16px; font-weight:750; margin:0 0 5px; }
.pl-tool-help { color:var(--pl-muted); font-size:11px; line-height:1.45; margin:0 0 12px; }
.pl-tool-list-caption {
  color:#8fa6bf; font-size:10px; font-weight:800; letter-spacing:.8px;
  text-transform:uppercase; margin:2px 0 7px;
}
#pl-tools-panel .pl-tool-picker { margin:0 0 12px !important; }
#pl-tools-panel .pl-tool-picker fieldset {
  display:flex !important; flex-direction:column !important; gap:6px !important;
  border:0 !important; padding:0 !important; margin:0 !important;
}
#pl-tools-panel .pl-tool-picker fieldset > label {
  display:flex !important; align-items:center !important; min-height:39px !important;
  border:1px solid #2b3d56 !important; border-radius:9px !important;
  padding:8px 11px !important; margin:0 !important; cursor:pointer !important;
  background:#111e30 !important; color:#dceafa !important;
  transition:background .15s ease,border-color .15s ease,transform .15s ease !important;
}
#pl-tools-panel .pl-tool-picker fieldset > label:hover {
  border-color:#4a8aa8 !important; background:#162b41 !important;
}
#pl-tools-panel .pl-tool-picker fieldset > label:has(input:checked) {
  border-color:#3dbbd7 !important; background:#123044 !important;
  box-shadow:inset 3px 0 0 #43d5ed !important; color:#f2fbff !important;
}
#pl-tools-panel .pl-tool-picker input[type="radio"] {
  position:absolute !important; width:1px !important; height:1px !important;
  opacity:0 !important; pointer-events:none !important;
}
#pl-tools-panel .pl-tool-picker fieldset > label:has(input:focus-visible) {
  outline:2px solid #43d5ed !important; outline-offset:2px !important;
}
#pl-tools-panel .pl-tool-picker fieldset > label span {
  font-size:12px !important; line-height:1.3 !important; font-weight:600 !important;
}
#pl-tools-panel #pl-flip-tool-list fieldset {
  display:grid !important; grid-template-columns:repeat(2,minmax(0,1fr)) !important; gap:6px !important;
}
#pl-tools-panel #pl-flip-tool-list fieldset > label {
  min-height:36px !important; justify-content:center !important; text-align:center !important;
  padding:7px 6px !important;
}
#pl-tools-panel .primary { background:linear-gradient(100deg,#43d5ed,#398be5) !important; color:#061321 !important; border:0 !important; font-weight:800 !important; }
#pl-tools-panel .secondary, #pl-action-row .secondary { border:1px solid #344962 !important; background:#17263b !important; color:#dceafa !important; }
#pl-tools-panel input, #pl-tools-panel textarea { background:#0c1727 !important; }
/* Ajustes: evita overflow horizontal causado pela largura mínima dos rótulos longos. */
#pl-adjust-panel {
  width:100% !important; min-width:0 !important; max-width:100% !important;
  overflow-x:hidden !important; box-sizing:border-box !important;
}
#pl-adjust-panel > div,
#pl-adjust-panel .block,
#pl-adjust-panel .wrap,
#pl-adjust-panel .form,
#pl-adjust-tool-list,
#pl-adjust-tool-list fieldset {
  min-width:0 !important; max-width:100% !important; box-sizing:border-box !important;
}
#pl-adjust-tool-list { width:100% !important; }
#pl-adjust-tool-list fieldset {
  width:100% !important; min-inline-size:0 !important;
}
#pl-adjust-tool-list fieldset > label {
  width:100% !important; min-width:0 !important; max-width:100% !important;
  box-sizing:border-box !important; white-space:normal !important;
  overflow-wrap:anywhere !important;
}
#pl-adjust-tool-list fieldset > label span {
  min-width:0 !important; max-width:100% !important;
  white-space:normal !important; overflow-wrap:anywhere !important;
}
/* Evita que componentes internos do inspetor imponham uma largura mínima ao painel. */
#pl-tool-settings, #pl-tool-settings > *, #pl-tool-settings .block,
#pl-tool-settings .wrap, #pl-tool-settings .form,
#pl-adjust-panel, #pl-filter-panel, #pl-transform-panel, #pl-segment-panel {
  min-width:0 !important; max-width:100% !important; box-sizing:border-box !important;
}
#pl-support-panels, #pl-support-panels > *, #pl-support-panels .column,
#pl-support-panels .wrap {
  min-width:0 !important; max-width:100% !important; box-sizing:border-box !important;
}
#pl-support-panels {
  position:static !important; inset:auto !important; transform:none !important;
  float:none !important; clear:both !important; z-index:auto !important;
  display:flex !important; flex:0 0 auto !important; flex-direction:column !important;
  align-self:stretch !important; width:100% !important; margin:14px 0 0 !important;
}
#pl-support-panels .label-wrap { color:#c9d8e8 !important; }
#pl-support-panels .accordion { border:1px solid var(--pl-border) !important; border-radius:12px !important; background:#101a29 !important; }
.gradio-container button { border-radius:9px !important; }
.gradio-container input:focus-visible, .gradio-container button:focus-visible, .gradio-container [role="radio"]:focus-visible { outline:2px solid #43d5ed !important; outline-offset:2px !important; }
@media (max-width: 1250px) and (min-width: 851px) {
  /* Em telas intermediárias, o inspetor passa para uma segunda linha acessível. */
  #pl-workspace { grid-template-columns:minmax(145px, 180px) minmax(0, 1fr) !important; }
  #pl-nav-panel { grid-column:1 !important; }
  #pl-canvas-column { grid-column:2 !important; }
  #pl-tools-panel {
    grid-column:1 / -1 !important; max-height:none !important;
    overflow-y:visible !important; padding:13px !important;
  }
  #pl-tools-panel #pl-tool-settings { display:grid !important; grid-template-columns:repeat(2, minmax(0, 1fr)) !important; gap:12px !important; }
}
#pl-empty-state { margin:0 0 10px; }
.pl-empty-state { min-height:86px; display:flex; align-items:center; justify-content:center; flex-direction:column; gap:5px; border:1px dashed #42617d; border-radius:12px; background:#101e30; color:#dceafa; text-align:center; padding:14px; }
.pl-empty-state strong { font-size:13px; font-weight:700; }
.pl-empty-state span { color:var(--pl-muted); font-size:11px; }
.pl-empty-icon { color:#43d5ed; font-size:22px; line-height:1; }
.pl-tool-description { border-left:2px solid #43d5ed; padding:7px 9px; margin:0 0 10px; background:#10253a; border-radius:0 8px 8px 0; color:#bcd3e8; font-size:11px; line-height:1.45; }
#pl-preview-actions { gap:8px !important; margin:8px 0 2px; }
#pl-preview-actions .primary { background:linear-gradient(100deg,#43d5ed,#398be5) !important; color:#061321 !important; font-weight:800 !important; }
@media (max-width: 850px) {
  .gradio-container { padding:8px !important; }
  #pl-workspace { grid-template-columns:minmax(0, 1fr) !important; gap:10px !important; }
  #pl-nav-panel, #pl-canvas-column, #pl-tools-panel {
    grid-column:1 !important; width:100% !important; min-width:0 !important;
    max-width:100% !important; max-height:none !important; overflow:visible !important;
  }
  #pl-nav-panel { padding:10px !important; }
  #pl-tool-nav fieldset { flex-direction:row !important; flex-wrap:wrap !important; }
  #pl-tool-nav label { flex:1 1 145px !important; }
  #pl-image-pair { flex-direction:column !important; }
  #pl-image-pair > div { width:100% !important; }
  #pl-original-image .image-container, #pl-result-image .image-container,
  #pl-original-image .upload-container { height:clamp(280px, 58vw, 460px) !important; min-height:280px !important; }
  #pl-support-panels .row { flex-direction:column !important; }
  #pl-support-panels .column { width:100% !important; }
  #pl-tools-panel #pl-tool-settings { display:block !important; }
  .pl-header-row { align-items:flex-start; flex-direction:column; }
  .pl-status-chip { white-space:normal; }
  .pl-title { font-size:21px; }
}
"""

THEME = gr.themes.Soft(primary_hue="cyan", secondary_hue="blue", neutral_hue="slate")


def select_tool(category: str):
    """Mostra apenas a seção de configurações correspondente à categoria selecionada."""
    selected = {
        "🎚️ Ajustes": "adjust",
        "◌ Filtros": "filters",
        "↔ Transformações": "transform",
        "◉ Segmentação e detecção": "segment",
    }.get(category, "adjust")
    return tuple(gr.update(visible=(name == selected)) for name in ("adjust", "filters", "transform", "segment"))


def adjustment_controls_visibility(mode: str):
    is_linear = mode == "Brilho e contraste"
    is_gamma = mode == "Correção de gama"
    is_saturation = mode == "Saturação"
    return (
        gr.update(visible=is_linear),
        gr.update(visible=is_linear),
        gr.update(visible=is_gamma),
        gr.update(visible=is_saturation),
    )


def filter_controls_visibility(mode: str):
    show_kernel = mode in ("Gaussiano", "Mediana", "Bilateral")
    show_sigma = mode in ("Gaussiano", "Bilateral")
    return gr.update(visible=show_kernel), gr.update(visible=show_sigma)


def segmentation_controls_visibility(mode: str):
    return (
        gr.update(visible=(mode == "Limiarização fixa")),
        gr.update(visible=(mode == "Adaptativa")),
        gr.update(visible=(mode == "Cor (HSV)")),
        gr.update(visible=(mode == "Canny")),
    )


def _tool_description(operation: str, mode: str) -> str:
    """Retorna uma explicação curta para o modo atualmente selecionado."""
    descriptions = {
        "adjust": {
            "Brilho e contraste": "☀️ Ajusta a luminosidade geral e a diferença entre áreas claras e escuras.",
            "Saturação": "🎨 Controla a intensidade das cores sem alterar diretamente o brilho.",
            "Correção de gama": "◐ Corrige os meios-tons; valores menores que 1 clareiam e maiores que 1 escurecem.",
            "Negativo": "◒ Inverte os valores de intensidade dos canais RGB.",
            "Escala de cinza": "◍ Converte a imagem colorida em tons de cinza.",
            "Equalização do histograma": "▥ Redistribui a luminância para ampliar o contraste global.",
            "Canal vermelho": "🔴 Exibe somente a contribuição do canal vermelho.",
            "Canal verde": "🟢 Exibe somente a contribuição do canal verde.",
            "Canal azul": "🔵 Exibe somente a contribuição do canal azul.",
        },
        "filter": {
            "Gaussiano": "〰️ Desfoca a imagem e reduz ruídos de alta frequência.",
            "Mediana": "◌ Suaviza ruído impulsivo preservando melhor algumas bordas.",
            "Bilateral": "〰️ Suaviza regiões mantendo melhor os contornos.",
            "Nitidez (sharpen)": "✧ Realça detalhes e transições locais de intensidade.",
            "Sobel": "⌁ Detecta bordas usando gradientes horizontais e verticais.",
            "Canny": "⌁ Detecta contornos com dois limiares de histerese.",
            "Laplaciano": "⌁ Realça mudanças rápidas de intensidade pela segunda derivada.",
        },
        "segment": {
            "Limiarização fixa": "◧ Separa pixels claros e escuros usando um limiar definido por você.",
            "Otsu": "◧ Calcula automaticamente um limiar global a partir do histograma.",
            "Adaptativa": "◧ Calcula limiares locais, útil quando a iluminação não é uniforme.",
            "Cor (HSV)": "🎯 Isola uma faixa de matiz, saturação e brilho no espaço HSV.",
            "Canny": "⌁ Detecta contornos com dois limiares, destacando bordas relevantes.",
        },
    }
    text = descriptions.get(operation, {}).get(mode, "Escolha uma operação para ver sua finalidade.")
    return f'<div class="pl-tool-description">{html.escape(text)}</div>'


def build_app():
    """Cria o editor full-screen em três painéis, preservando a prévia temporária ao vivo."""
    with gr.Blocks(title="PixelLab — Editor de Imagens", fill_width=True, fill_height=True) as demo:
        original_state = gr.State(None)
        current_state = gr.State(None)
        undo_state = gr.State([])
        redo_state = gr.State([])
        history_state = gr.State([])
        preview_payload_state = gr.State(None)

        gr.HTML("""
        <header id="pl-header">
          <div class="pl-header-row">
            <div class="pl-brand">
              <div class="pl-logo" aria-hidden="true">◈</div>
              <div>
                <div class="pl-title">PixelLab <span style="color:#43d5ed;font-weight:500">/ Editor de Imagens</span></div>
                <div class="pl-subtitle">Editor de imagens · OpenCV + NumPy</div>
              </div>
            </div>
            <div class="pl-status-chip">● EDIÇÃO LOCAL</div>
          </div>
        </header>
        """)

        with gr.Row(elem_id="pl-workspace", equal_height=False):
            with gr.Column(scale=1, min_width=170, elem_id="pl-nav-panel"):
                gr.Markdown("<div class='pl-section-label'>Menu</div>")
                gr.Markdown("<div class='pl-tool-heading'>Ferramentas</div><p class='pl-tool-help'>Escolha uma categoria para abrir os controles no inspetor.</p>")
                tool_nav = gr.Radio(
                    choices=["🎚️ Ajustes", "◌ Filtros", "↔ Transformações", "◉ Segmentação e detecção"],
                    value="🎚️ Ajustes", label="Categorias", show_label=False,
                    container=False, elem_id="pl-tool-nav",
                )
                with gr.Accordion("Ajuda rápida", open=False):
                    gr.Markdown(
                        "1. **Carregue** uma imagem na área central.\n"
                        "2. **Escolha** uma categoria neste menu.\n"
                        "3. **Ajuste** os controles no inspetor à direita.\n"
                        "4. Confirme ou descarte a prévia e use o histórico para desfazer/refazer.\n"
                        "5. Exporte o resultado em PNG."
                    )

            with gr.Column(scale=7, min_width=400, elem_id="pl-canvas-column"):
                gr.Markdown("<div class='pl-section-label'>Canvas · Comparação original e resultado</div>")
                empty_state = gr.HTML(
                    "<div class='pl-empty-state'><div class='pl-empty-icon'>▧</div><strong>Carregue uma imagem para começar a editar</strong><span>Você também pode arrastar um arquivo para a área de imagem.</span></div>",
                    elem_id="pl-empty-state",
                    visible=True,
                )
                with gr.Row(elem_id="pl-image-pair", equal_height=False):
                    upload_image = gr.Image(
                        type="numpy", label="Abrir imagem · Original", sources=["upload", "clipboard"],
                        height=560, elem_id="pl-original-image",
                    )
                    result_image = gr.Image(
                        label="Resultado · Prévia / imagem final", interactive=False, height=560,
                        elem_id="pl-result-image",
                    )
                with gr.Row(elem_id="pl-preview-actions"):
                    confirm_preview_btn = gr.Button("✓ Confirmar prévia", variant="primary", size="sm")
                    cancel_preview_btn = gr.Button("✕ Descartar prévia", variant="secondary", size="sm")
                with gr.Row(elem_id="pl-action-row"):
                    undo_btn = gr.Button("↶ Desfazer", variant="secondary", size="sm")
                    redo_btn = gr.Button("↷ Refazer", variant="secondary", size="sm")
                    restore_btn = gr.Button("⟲ Redefinir", variant="secondary", size="sm")
                    export_btn = gr.Button("↓ Baixar imagem", variant="primary", size="sm")
                download_file = gr.File(label="Arquivo PNG pronto para baixar", interactive=False, visible=False)
                status_html = gr.HTML(make_status("Carregue uma imagem para começar a editar."), elem_id="pl-status")
                # Seções auxiliares no fluxo normal do canvas, depois de todos os botões de ação.
                with gr.Column(elem_id="pl-support-panels", visible=True):
                    with gr.Accordion("Análise da imagem · etapa intermediária, histograma e detalhes", open=False, visible=True):
                        with gr.Row(equal_height=False):
                            with gr.Column(scale=1, min_width=200):
                                intermediate_image = gr.Image(label="Etapa intermediária", interactive=False, height=230)
                            with gr.Column(scale=2, min_width=280):
                                histogram_plot = gr.Plot(label="Histograma RGB")
                            with gr.Column(scale=1, min_width=210):
                                details_html = gr.HTML(make_details_html(None, None))
                    with gr.Accordion("Histórico de alterações", open=False, visible=True):
                        history_text = gr.Textbox(value="Nenhuma operação executada ainda.", label="Operações confirmadas", lines=6, interactive=False)

            with gr.Column(scale=3, min_width=330, elem_id="pl-tools-panel"):
                gr.Markdown("<div class='pl-section-label'>Inspetor · Controles</div>")
                gr.Markdown("<div class='pl-tool-help'>Escolha uma ferramenta na lista abaixo. Os parâmetros aparecem diretamente, sem menus suspensos.</div>")
                with gr.Column(elem_id="pl-tool-settings"):
                    with gr.Group(visible=True, elem_id="pl-adjust-panel") as adjust_panel:
                        gr.HTML("<div class='pl-tool-heading'>🎚️ Ajustes</div><p class='pl-tool-help'>Controle brilho, contraste, cor e distribuição tonal. A prévia acompanha os controles compatíveis.</p>")
                        gr.HTML("<div class='pl-tool-list-caption'>Selecione um ajuste</div>")
                        adj_mode = gr.Radio(
                            choices=[
                                ("☀️  Brilho e contraste", "Brilho e contraste"),
                                ("🎨  Saturação", "Saturação"),
                                ("◐  Correção de gama", "Correção de gama"),
                                ("◒  Negativo", "Negativo"),
                                ("◍  Escala de cinza", "Escala de cinza"),
                                ("▥  Equalização do histograma", "Equalização do histograma"),
                                ("🔴  Canal vermelho", "Canal vermelho"),
                                ("🟢  Canal verde", "Canal verde"),
                                ("🔵  Canal azul", "Canal azul"),
                            ],
                            value="Brilho e contraste", show_label=False, container=False,
                            elem_id="pl-adjust-tool-list", elem_classes=["pl-tool-picker"],
                        )
                        adj_description = gr.HTML(_tool_description("adjust", "Brilho e contraste"))
                        brightness = gr.Slider(-100, 100, value=0, step=1, label="Brilho", info="Aumente para clarear; diminua para escurecer.")
                        contrast = gr.Slider(0.25, 2.5, value=1.0, step=0.05, label="Contraste", info="Valores maiores destacam diferenças entre claro e escuro.")
                        gamma = gr.Slider(0.2, 3.0, value=1.0, step=0.1, label="Gama", info="Abaixo de 1 clareia os meios-tons; acima de 1 escurece.", visible=False)
                        saturation = gr.Slider(0.0, 3.0, value=1.0, step=0.05, label="Saturação", info="0 remove a cor; acima de 1 intensifica as cores.", visible=False)
                        apply_adjust_btn = gr.Button("Aplicar ajuste", variant="primary")

                    with gr.Group(visible=False, elem_id="pl-filter-panel") as filter_panel:
                        gr.HTML("<div class='pl-tool-heading'>◌ Filtros</div><p class='pl-tool-help'>Reduza ruídos, desfoque, realce a nitidez ou encontre bordas.</p>")
                        gr.HTML("<div class='pl-tool-list-caption'>Selecione um filtro</div>")
                        filter_mode = gr.Radio(
                            choices=[
                                ("〰️  Gaussiano", "Gaussiano"),
                                ("◌  Mediana", "Mediana"),
                                ("◉  Bilateral", "Bilateral"),
                                ("✧  Nitidez (sharpen)", "Nitidez (sharpen)"),
                                ("⌁  Sobel", "Sobel"),
                                ("⌁  Laplaciano", "Laplaciano"),
                            ],
                            value="Gaussiano", show_label=False, container=False,
                            elem_id="pl-filter-tool-list", elem_classes=["pl-tool-picker"],
                        )
                        filter_description = gr.HTML(_tool_description("filter", "Gaussiano"))
                        kernel_size = gr.Slider(3, 15, value=5, step=2, label="Tamanho da janela", info="Janela maior aumenta o efeito de vizinhança.")
                        sigma = gr.Slider(0.5, 8.0, value=2.0, step=0.5, label="Intensidade da suavização", info="Valores maiores aumentam o desfoque/suavização.")
                        apply_filter_btn = gr.Button("Aplicar filtro", variant="primary")

                    with gr.Group(visible=False, elem_id="pl-transform-panel") as transform_panel:
                        gr.HTML("<div class='pl-tool-heading'>↔ Transformações</div><p class='pl-tool-help'>Gire, redimensione ou espelhe a imagem. A rotação ajusta o enquadramento para preservar a imagem inteira.</p>")
                        angle = gr.Slider(-180, 180, value=0, step=1, label="Rotação", info="Ângulo em graus.")
                        scale = gr.Slider(0.25, 2.0, value=1.0, step=0.05, label="Redimensionamento", info="1,0 mantém as dimensões relativas originais.")
                        gr.Markdown("**Espelhamento**")
                        flip_mode = gr.Radio(
                            choices=[
                                ("Sem espelhar", "Nenhum"),
                                ("↔  Horizontal", "Horizontal"),
                                ("↕  Vertical", "Vertical"),
                                ("✥  Ambos", "Horizontal e vertical"),
                            ],
                            value="Nenhum", show_label=False, container=False,
                            elem_id="pl-flip-tool-list", elem_classes=["pl-tool-picker", "pl-tool-picker-grid"],
                        )
                        apply_transform_btn = gr.Button("Aplicar transformação", variant="primary")

                    with gr.Group(visible=False, elem_id="pl-segment-panel") as segment_panel:
                        gr.HTML("<div class='pl-tool-heading'>◉ Segmentação e detecção</div><p class='pl-tool-help'>Isole regiões por intensidade ou cor e detecte contornos.</p>")
                        gr.HTML("<div class='pl-tool-list-caption'>Selecione uma técnica</div>")
                        segment_mode = gr.Radio(
                            choices=[
                                ("◧  Limiarização fixa", "Limiarização fixa"),
                                ("◧  Otsu automático", "Otsu"),
                                ("▦  Adaptativa", "Adaptativa"),
                                ("🎯  Segmentação por cor (HSV)", "Cor (HSV)"),
                                ("⌁  Bordas · Canny", "Canny"),
                            ],
                            value="Otsu", show_label=False, container=False,
                            elem_id="pl-segment-tool-list", elem_classes=["pl-tool-picker"],
                        )
                        segment_description = gr.HTML(_tool_description("segment", "Otsu"))
                        threshold = gr.Slider(0, 255, value=127, step=1, label="Limiar", info="Pixels acima do valor ficam brancos.", visible=False)
                        block_size = gr.Slider(3, 51, value=21, step=2, label="Tamanho do bloco adaptativo", info="Janela local ímpar usada para calcular o limiar.", visible=False)
                        with gr.Group(visible=False, elem_id="pl-hsv-controls") as hsv_group:
                            gr.Markdown("**Faixa de cor HSV**")
                            h_min = gr.Slider(0, 179, value=0, step=1, label="Matiz mínima (H)", info="Início da faixa de cor.")
                            h_max = gr.Slider(0, 179, value=179, step=1, label="Matiz máxima (H)", info="Fim da faixa de cor.")
                            s_min = gr.Slider(0, 255, value=50, step=1, label="Saturação mínima (S)", info="Ignora cores menos saturadas.")
                            v_min = gr.Slider(0, 255, value=50, step=1, label="Valor mínimo (V)", info="Ignora pixels mais escuros.")
                        with gr.Group(visible=False, elem_id="pl-canny-controls") as canny_group:
                            gr.Markdown("**Detecção de bordas · Canny**")
                            segment_canny_low = gr.Slider(0, 255, value=60, step=1, label="Limiar inferior", info="Controla a sensibilidade a bordas fracas.")
                            segment_canny_high = gr.Slider(1, 255, value=150, step=1, label="Limiar superior", info="Controla quais bordas fortes são mantidas.")
                        apply_segment_btn = gr.Button("Aplicar segmentação / detecção", variant="primary")



        upload_image.upload(
            fn=load_image,
            inputs=[upload_image],
            outputs=[original_state, current_state, undo_state, redo_state, history_state, result_image, intermediate_image, histogram_plot, details_html, history_text, status_html, preview_payload_state, empty_state],
            show_progress="minimal",
        )

        state_inputs = [original_state, current_state, undo_state, redo_state, history_state]
        view_outputs = [result_image, intermediate_image, histogram_plot, details_html, current_state, undo_state, redo_state, history_state, history_text, status_html, preview_payload_state, empty_state]
        preview_outputs = [result_image, intermediate_image, histogram_plot, details_html, preview_payload_state, status_html]

        tool_nav.change(
            fn=select_tool,
            inputs=[tool_nav],
            outputs=[adjust_panel, filter_panel, transform_panel, segment_panel],
            show_progress="hidden",
        )
        tool_nav.change(fn=cancel_preview, inputs=state_inputs, outputs=view_outputs, show_progress="hidden", js=PREVIEW_CANCEL_JS)

        adj_mode.change(
            fn=adjustment_controls_visibility,
            inputs=[adj_mode],
            outputs=[brightness, contrast, gamma, saturation],
            show_progress="hidden",
        )
        adj_mode.change(fn=lambda mode: _tool_description("adjust", mode), inputs=[adj_mode], outputs=[adj_description], show_progress="hidden")
        adj_mode.change(fn=cancel_preview, inputs=state_inputs, outputs=view_outputs, show_progress="hidden", js=PREVIEW_CANCEL_JS)
        filter_mode.change(
            fn=filter_controls_visibility,
            inputs=[filter_mode],
            outputs=[kernel_size, sigma],
            show_progress="hidden",
        )
        filter_mode.change(fn=lambda mode: _tool_description("filter", mode), inputs=[filter_mode], outputs=[filter_description], show_progress="hidden")
        filter_mode.change(fn=cancel_preview, inputs=state_inputs, outputs=view_outputs, show_progress="hidden", js=PREVIEW_CANCEL_JS)
        segment_mode.change(
            fn=segmentation_controls_visibility,
            inputs=[segment_mode],
            outputs=[threshold, block_size, hsv_group, canny_group],
            show_progress="hidden",
        )
        segment_mode.change(fn=lambda mode: _tool_description("segment", mode), inputs=[segment_mode], outputs=[segment_description], show_progress="hidden")
        segment_mode.change(fn=cancel_preview, inputs=state_inputs, outputs=view_outputs, show_progress="hidden", js=PREVIEW_CANCEL_JS)

        # Prévia ao vivo: o JavaScript aguarda 150 ms após a última mudança antes de enviar
        # os parâmetros ao Python. Cada cálculo parte de current_state, nunca da prévia anterior.
        adjust_preview_inputs = [original_state, current_state, adj_mode, brightness, contrast, gamma, saturation]
        filter_preview_inputs = [original_state, current_state, filter_mode, kernel_size, sigma]
        segment_preview_inputs = [original_state, current_state, segment_mode, threshold, block_size, h_min, h_max, s_min, v_min, segment_canny_low, segment_canny_high]
        preview_event_options = {"show_progress": "hidden", "js": PREVIEW_DEBOUNCE_JS, "trigger_mode": "always_last", "concurrency_limit": 1}
        for control in [brightness, contrast, gamma, saturation]:
            control.input(fn=make_preview_callback("adjust"), inputs=adjust_preview_inputs, outputs=preview_outputs, **preview_event_options)
        for control in [kernel_size, sigma]:
            control.input(fn=make_preview_callback("filter"), inputs=filter_preview_inputs, outputs=preview_outputs, **preview_event_options)
        for control in [threshold, block_size, h_min, h_max, s_min, v_min, segment_canny_low, segment_canny_high]:
            control.input(fn=make_preview_callback("segment"), inputs=segment_preview_inputs, outputs=preview_outputs, **preview_event_options)

        apply_adjust_btn.click(fn=make_process_callback("adjust"), inputs=state_inputs + [adj_mode, brightness, contrast, gamma, saturation], outputs=view_outputs, show_progress="minimal")
        apply_filter_btn.click(fn=make_process_callback("filter"), inputs=state_inputs + [filter_mode, kernel_size, sigma], outputs=view_outputs, show_progress="minimal")
        apply_transform_btn.click(fn=make_process_callback("transform"), inputs=state_inputs + [angle, scale, flip_mode], outputs=view_outputs, show_progress="minimal")
        apply_segment_btn.click(fn=make_process_callback("segment"), inputs=state_inputs + [segment_mode, threshold, block_size, h_min, h_max, s_min, v_min, segment_canny_low, segment_canny_high], outputs=view_outputs, show_progress="minimal")

        confirm_preview_btn.click(fn=confirm_preview, inputs=state_inputs + [preview_payload_state], outputs=view_outputs, show_progress="minimal")
        cancel_preview_btn.click(fn=cancel_preview, inputs=state_inputs, outputs=view_outputs, show_progress="hidden")
        restore_btn.click(fn=restore_original, inputs=state_inputs, outputs=view_outputs, show_progress="minimal")
        undo_btn.click(fn=undo_operation, inputs=state_inputs, outputs=view_outputs, show_progress="minimal")
        redo_btn.click(fn=redo_operation, inputs=state_inputs, outputs=view_outputs, show_progress="minimal")
        export_btn.click(fn=export_image_for_ui, inputs=[current_state], outputs=[download_file, status_html], show_progress="minimal")

    return demo


# Instância para execução como script ou importação no notebook.
demo = build_app()

if __name__ == "__main__":
    # Compartilhamento público apenas no Colab. Em execução local o padrão é privado.
    running_in_colab = "google.colab" in sys.modules or bool(os.getenv("COLAB_RELEASE_TAG"))
    share_link = os.getenv("PIXELLAB_SHARE", "1" if running_in_colab else "0").strip().lower() in {"1", "true", "yes", "sim"}
    demo.launch(share=share_link, debug=False, theme=THEME, css=CSS)
