"""Interfaz grafica de Gesture Rush (Pygame).

Pantallas
---------
- ``MenuScreen``: eleccion de minijuego (Duelo, Cadena y Prueba).
- ``DuelScreen``: partida clasica con una sola camara. Cada jugador ocupa una
  mitad del encuadre y compite por hacer el gesto objetivo primero.
- ``ChainScreen``: los dos jugadores reciben la misma cadena de gestos y gana
  el primero en completarla.
- ``PracticeScreen``: modo Prueba, sin reglas. Muestra la prediccion cruda, el
  top 3 del clasificador y el voto temporal, para explicar el modelo.

El dibujo se hace sobre una superficie de 1600x900 que pygame escala a la
ventana real (``pygame.SCALED``), asi que las coordenadas son siempre las
mismas y la interfaz se adapta al monitor.

Los tiempos se muestran siempre en milisegundos. La musica de fondo se
sintetiza en ``audio.py`` y el control de volumen se dibuja por encima de
cualquier pantalla; el confeti es una animacion generica que se lanza al
cerrar una ronda o carrera con ganador.
"""

from __future__ import annotations

import math
import random
import time
from pathlib import Path

import cv2
import numpy as np
import pygame

from . import config
from .audio import Music
from .game import (
    PHASE_COUNTDOWN,
    PHASE_FINISHED,
    PHASE_IDLE,
    PHASE_PLAYING,
    PHASE_RESULT,
    ChainRace,
    Duel,
    PracticeSession,
)
from .player_tracker import PlayerReading

# ---------------------------------------------------------------------------
# Lienzo
# ---------------------------------------------------------------------------
WIDTH = 1600
HEIGHT = 900
FPS = 60
MARGIN = 24

# ---------------------------------------------------------------------------
# Paleta
# ---------------------------------------------------------------------------
BG = (11, 14, 30)
CARD = (22, 28, 54)
CARD_SOFT = (28, 35, 66)
INNER = (12, 16, 36)
BORDER = (44, 53, 92)
TEXT = (234, 239, 252)
MUTED = (139, 152, 190)
AMBER = (247, 200, 72)
AMBER_DEEP = (204, 154, 44)
GOOD = (74, 222, 128)
BAD = (248, 113, 113)
DIM = (70, 82, 122)

PLAYER_COLORS = [(108, 162, 255), (255, 94, 138), (110, 231, 183)]
PLAYER_INNER = [(28, 44, 84), (80, 26, 48), (24, 70, 60)]

# Etiquetas cortas para la interfaz (config guarda las largas).
SHORT_LABELS = {
    "open_palm": "Mano abierta",
    "fist": "Puño",
    "thumbs_up": "Pulgar arriba",
    "victory": "Victoria",
    "pointing_index": "Índice",
    "ok_sign": "OK",
    "rock": "Rock",
    "shaka": "Shaka",
    "pinky_up": "Meñique",
    "l_shape": "Pistola",
}

# (dedos extendidos indice..meñique, pulgar extendido) para el icono vectorial.
HAND_POSES: dict[str, tuple[tuple[bool, bool, bool, bool], bool]] = {
    "open_palm": ((True, True, True, True), True),
    "fist": ((False, False, False, False), False),
    "thumbs_up": ((False, False, False, False), True),
    "victory": ((True, True, False, False), False),
    "pointing_index": ((True, False, False, False), False),
    "ok_sign": ((True, False, False, False), True),
    "rock": ((True, False, False, True), False),
    "shaka": ((False, False, False, True), True),
    "pinky_up": ((False, False, False, True), False),
    "l_shape": ((True, False, False, False), True),
}

# Geometria del icono (lienzo de 120x120): x del dedo, ancho, largo y desvio.
FINGER_SPECS = (
    (52, 11, 33, -2),  # indice
    (63, 12, 39, 0),   # medio
    (73, 11, 35, 2),   # anular
    (83, 9, 26, 4),    # menique
)
PALM_RECT = (45, 52, 44, 42)
WRIST_RECT = (52, 88, 30, 18)

_GLYPH_CACHE: dict[tuple[str, int], pygame.Surface] = {}
_MASK_CACHE: dict[tuple[int, int, int], pygame.Surface] = {}


def short_label(gesture: str | None) -> str:
    if not gesture:
        return "—"
    return SHORT_LABELS.get(gesture, gesture)


# ---------------------------------------------------------------------------
# Utilidades de dibujo
# ---------------------------------------------------------------------------
class Fonts:
    """Cache de tipografias (Segoe UI en Windows, Arial como respaldo)."""

    def __init__(self) -> None:
        self._cache: dict[tuple[int, bool], pygame.font.Font] = {}

    def get(self, size: int, bold: bool = False) -> pygame.font.Font:
        key = (size, bold)
        font = self._cache.get(key)
        if font is None:
            font = pygame.font.SysFont("segoeui,arial,tahoma", size, bold=bold)
            self._cache[key] = font
        return font


def draw_text(
    surface: pygame.Surface,
    fonts: Fonts,
    text: str,
    size: int,
    color: tuple[int, int, int],
    pos: tuple[int, int],
    *,
    bold: bool = False,
    anchor: str = "topleft",
) -> pygame.Rect:
    image = fonts.get(size, bold).render(text, True, color)
    rect = image.get_rect(**{anchor: pos})
    surface.blit(image, rect)
    return rect


def wrap_text(text: str, font: pygame.font.Font, max_width: int) -> list[str]:
    lines: list[str] = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if not current or font.size(candidate)[0] <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def draw_paragraph(
    surface: pygame.Surface,
    fonts: Fonts,
    text: str,
    size: int,
    color: tuple[int, int, int],
    pos: tuple[int, int],
    max_width: int,
    *,
    line_height: int | None = None,
) -> int:
    font = fonts.get(size)
    step = line_height or int(size * 1.45)
    for i, line in enumerate(wrap_text(text, font, max_width)):
        draw_text(surface, fonts, line, size, color, (pos[0], pos[1] + i * step))
    return pos[1] + step * max(1, len(wrap_text(text, font, max_width)))


def draw_card(
    surface: pygame.Surface,
    rect: pygame.Rect,
    color: tuple[int, int, int] = CARD,
    *,
    radius: int = 20,
    border: tuple[int, int, int] | None = None,
    border_width: int = 2,
) -> None:
    pygame.draw.rect(surface, color, rect, border_radius=radius)
    if border is not None:
        pygame.draw.rect(surface, border, rect, width=border_width, border_radius=radius)


def draw_pill(
    surface: pygame.Surface,
    fonts: Fonts,
    text: str,
    pos: tuple[int, int],
    bg: tuple[int, int, int],
    fg: tuple[int, int, int],
    *,
    size: int = 15,
    pad_x: int = 14,
    pad_y: int = 7,
    anchor: str = "topleft",
    bold: bool = False,
) -> pygame.Rect:
    label = fonts.get(size, bold).render(text, True, fg)
    rect = pygame.Rect(0, 0, label.get_width() + pad_x * 2, label.get_height() + pad_y * 2)
    setattr(rect, anchor, pos)
    pygame.draw.rect(surface, bg, rect, border_radius=rect.height // 2)
    surface.blit(label, label.get_rect(center=rect.center))
    return rect


def draw_check(
    surface: pygame.Surface,
    center: tuple[int, int],
    size: int,
    color: tuple[int, int, int] = GOOD,
    width: int = 4,
) -> None:
    x, y = center
    points = [
        (x - size * 0.45, y + size * 0.02),
        (x - size * 0.12, y + size * 0.36),
        (x + size * 0.5, y - size * 0.34),
    ]
    pygame.draw.lines(surface, color, False, points, width)


_GLYPH_UNITS = 120
_GLYPH_SS = 4  # se dibuja 4x y se reduce: bordes suaves


def _g(x: float, y: float) -> tuple[int, int]:
    return (int(round(x * _GLYPH_SS)), int(round(y * _GLYPH_SS)))


def _capsule(
    surface: pygame.Surface,
    color: tuple[int, int, int],
    start: tuple[float, float],
    end: tuple[float, float],
    width: float,
) -> None:
    """Segmento con extremos redondeados (dedo o falange)."""
    a = _g(*start)
    b = _g(*end)
    w = max(2, int(round(width * _GLYPH_SS)))
    pygame.draw.line(surface, color, a, b, w)
    radius = w // 2
    pygame.draw.circle(surface, color, a, radius)
    pygame.draw.circle(surface, color, b, radius)


def _taper(
    surface: pygame.Surface,
    color: tuple[int, int, int],
    start: tuple[float, float],
    end: tuple[float, float],
    width_start: float,
    width_end: float,
) -> None:
    """Falange que se afina hacia la punta."""
    sx, sy = start
    ex, ey = end
    dx, dy = ex - sx, ey - sy
    length = math.hypot(dx, dy)
    if length < 0.01:
        return
    nx, ny = -dy / length, dx / length
    quad = (
        (sx + nx * width_start / 2, sy + ny * width_start / 2),
        (ex + nx * width_end / 2, ey + ny * width_end / 2),
        (ex - nx * width_end / 2, ey - ny * width_end / 2),
        (sx - nx * width_start / 2, sy - ny * width_start / 2),
    )
    pygame.draw.polygon(surface, color, [_g(x, y) for x, y in quad])
    pygame.draw.circle(surface, color, _g(sx, sy), int(width_start * _GLYPH_SS / 2))
    pygame.draw.circle(surface, color, _g(ex, ey), int(width_end * _GLYPH_SS / 2))


def _finger_glyph(
    surface: pygame.Surface,
    x: float,
    width: float,
    length: float,
    splay: float,
    color: tuple[int, int, int] = AMBER,
) -> None:
    """Dedo extendido: dos falanges que se afinan hacia la punta."""
    base = (x, 58.0)
    knuckle = (x + splay * 0.45, 58.0 - length * 0.5)
    tip = (x + splay, 58.0 - length)
    _taper(surface, color, base, knuckle, width, width * 0.95)
    _taper(surface, color, knuckle, tip, width * 0.95, width * 0.78)


def _knuckle_glyph(
    surface: pygame.Surface,
    x: float,
    width: float,
    color: tuple[int, int, int] = AMBER,
) -> None:
    """Dedo recogido: el nudillo asoma sobre la palma."""
    _taper(surface, color, (x, 62.0), (x, 45.0), width, width * 0.88)


def _rounded_polygon(
    surface: pygame.Surface,
    color: tuple[int, int, int],
    points: tuple[tuple[float, float], ...],
    radius: float,
    steps: int = 10,
) -> None:
    """Poligono con esquinas suavizadas (bezier cuadratica en cada vertice)."""
    outline: list[tuple[float, float]] = []
    count = len(points)
    for index in range(count):
        prev = points[index - 1]
        point = points[index]
        nxt = points[(index + 1) % count]
        v1 = (point[0] - prev[0], point[1] - prev[1])
        v2 = (nxt[0] - point[0], nxt[1] - point[1])
        len1 = math.hypot(*v1) or 1.0
        len2 = math.hypot(*v2) or 1.0
        v1 = (v1[0] / len1, v1[1] / len1)
        v2 = (v2[0] / len2, v2[1] / len2)
        r = min(radius, len1 / 2, len2 / 2)
        start = (point[0] - v1[0] * r, point[1] - v1[1] * r)
        end = (point[0] + v2[0] * r, point[1] + v2[1] * r)
        for step in range(steps + 1):
            t = step / steps
            u = 1.0 - t
            outline.append(
                (
                    u * u * start[0] + 2 * u * t * point[0] + t * t * end[0],
                    u * u * start[1] + 2 * u * t * point[1] + t * t * end[1],
                )
            )
    pygame.draw.polygon(surface, color, [_g(x, y) for x, y in outline])


def _palm_glyph(surface: pygame.Surface, color: tuple[int, int, int] = AMBER) -> None:
    # Palma y muñeca en una sola silueta: mas ancha en los nudillos.
    _rounded_polygon(
        surface, color,
        ((43, 50), (87, 50), (86, 88), (79, 104), (53, 104), (42, 88)),
        11,
    )


def _thumb_out_glyph(surface: pygame.Surface, *, horizontal: bool = False) -> None:
    if horizontal:
        # Forma de "L": el pulgar sale de costado.
        _capsule(surface, AMBER, (50, 82), (28, 74), 14)
        return
    _capsule(surface, AMBER, (52, 84), (42, 68), 15)
    _capsule(surface, AMBER, (42, 68), (37, 51), 13)


def _thumb_up_glyph(surface: pygame.Surface) -> None:
    # Sale por fuera de la palma para que se lea por encima del puño.
    _capsule(surface, AMBER, (52, 86), (42, 64), 15)
    _capsule(surface, AMBER, (42, 64), (38, 37), 13)


def _thumb_across_glyph(surface: pygame.Surface) -> None:
    """Pulgar plegado sobre la palma; tono mas oscuro para separarlo."""
    _capsule(surface, AMBER_DEEP, (54, 88), (73, 76), 13)


def _fist_creases(surface: pygame.Surface) -> None:
    """Pliegues entre los dedos recogidos."""
    for x in (57.0, 68.0, 78.0):
        _capsule(surface, AMBER_DEEP, (x, 54), (x, 74), 2)


def _ok_glyph(surface: pygame.Surface) -> None:
    """OK: el indice y el pulgar forman un anillo, el resto estirado."""
    pygame.draw.circle(
        surface, AMBER, _g(52, 43), int(11 * _GLYPH_SS), int(7 * _GLYPH_SS)
    )
    _capsule(surface, AMBER, (52, 88), (45, 66), 14)
    _capsule(surface, AMBER, (45, 66), (47, 54), 12)
    for x, width, length, splay in FINGER_SPECS[1:]:
        _finger_glyph(surface, x, width, length, splay)
    _palm_glyph(surface)


def _outline_glyph(
    surface: pygame.Surface, color: tuple[int, int, int] = AMBER_DEEP
) -> None:
    """Contorno del dibujo completo; le da acabado de icono."""
    mask = pygame.mask.from_surface(surface)
    outline = mask.outline()
    if len(outline) > 2:
        pygame.draw.lines(surface, color, True, outline, int(_GLYPH_SS * 1.6))


_HAND_DIR = Path(__file__).resolve().parent.parent / "assets" / "hands"
_HAND_ASSETS: dict[str, pygame.Surface | None] = {}
# Ajustes de orientacion por gesto: (espejo horizontal, giro en grados).
_HAND_TURNS: dict[str, tuple[bool, int]] = {"l_shape": (True, 90)}


def _hand_asset(gesture: str) -> pygame.Surface | None:
    """Imagen del gesto (assets/hands/<gesto>.png), recortada y orientada.

    Devuelve ``None`` si no hay archivo: la pantalla usa el dibujo vectorial.
    """
    if gesture in _HAND_ASSETS:
        return _HAND_ASSETS[gesture]

    path = _HAND_DIR / f"{gesture}.png"
    image: pygame.Surface | None = None
    if path.exists():
        image = pygame.image.load(str(path))
        if pygame.display.get_surface() is not None:
            image = image.convert_alpha()
        bounds = image.get_bounding_rect(min_alpha=10)
        if bounds.width and bounds.height:
            image = image.subsurface(bounds).copy()
        mirror, angle = _HAND_TURNS.get(gesture, (False, 0))
        if mirror:
            image = pygame.transform.flip(image, True, False)
        if angle:
            image = pygame.transform.rotate(image, angle)

    _HAND_ASSETS[gesture] = image
    return image


def _fit_hand(image: pygame.Surface, size: int) -> pygame.Surface:
    """Escala la imagen sin deformar y la centra en un cuadro cuadrado."""
    target = max(1, int(round(size * 0.96)))
    width, height = image.get_size()
    scale = min(target / width, target / height)
    scaled = pygame.transform.smoothscale(
        image, (max(1, int(width * scale)), max(1, int(height * scale)))
    )
    canvas = pygame.Surface((size, size), pygame.SRCALPHA)
    canvas.blit(scaled, scaled.get_rect(center=(size // 2, size // 2)))
    return canvas


def hand_glyph(gesture: str | None, size: int) -> pygame.Surface:
    """Icono de la mano para el gesto pedido.

    Usa la imagen de ``assets/hands`` si existe; si no, dibuja el icono
    vectorial.
    """
    gesture = gesture or "fist"
    key = (gesture, size)
    cached = _GLYPH_CACHE.get(key)
    if cached is not None:
        return cached

    asset = _hand_asset(gesture)
    if asset is not None:
        icon = _fit_hand(asset, size)
        _GLYPH_CACHE[key] = icon
        return icon

    canvas = _GLYPH_UNITS * _GLYPH_SS
    surf = pygame.Surface((canvas, canvas), pygame.SRCALPHA)

    if gesture == "ok_sign":
        _ok_glyph(surf)
    else:
        fingers, thumb = HAND_POSES.get(gesture, HAND_POSES["fist"])
        if thumb:
            if gesture == "thumbs_up":
                _thumb_up_glyph(surf)
            else:
                _thumb_out_glyph(surf, horizontal=gesture == "l_shape")
        for (x, width, length, splay), extended in zip(FINGER_SPECS, fingers):
            if extended:
                _finger_glyph(surf, x, width, length, splay)
            else:
                _knuckle_glyph(surf, x, width)
        _palm_glyph(surf)
        if not thumb:
            _thumb_across_glyph(surf)
            _fist_creases(surf)

    _outline_glyph(surf)
    scaled = pygame.transform.smoothscale(surf, (size, size))
    _GLYPH_CACHE[key] = scaled
    return scaled


def rounded_mask(size: tuple[int, int], radius: int) -> pygame.Surface:
    key = (size[0], size[1], radius)
    mask = _MASK_CACHE.get(key)
    if mask is None:
        mask = pygame.Surface(size, pygame.SRCALPHA)
        mask.fill((0, 0, 0, 0))
        pygame.draw.rect(mask, (255, 255, 255, 255), mask.get_rect(), border_radius=radius)
        _MASK_CACHE[key] = mask
    return mask


def draw_avatar(
    surface: pygame.Surface,
    center: tuple[int, int],
    color: tuple[int, int, int],
    scale: float = 1.0,
) -> None:
    cx, cy = center
    pygame.draw.circle(surface, color, (cx, cy - int(36 * scale)), int(14 * scale))
    pygame.draw.line(surface, color, (cx, cy - int(20 * scale)), (cx, cy + int(22 * scale)), max(2, int(5 * scale)))
    pygame.draw.line(
        surface, color,
        (cx, cy - int(12 * scale)), (cx - int(20 * scale), cy + int(6 * scale)),
        max(2, int(5 * scale)),
    )
    pygame.draw.line(
        surface, color,
        (cx, cy - int(12 * scale)), (cx + int(20 * scale), cy + int(6 * scale)),
        max(2, int(5 * scale)),
    )
    pygame.draw.line(
        surface, color,
        (cx, cy + int(22 * scale)), (cx - int(16 * scale), cy + int(48 * scale)),
        max(2, int(5 * scale)),
    )
    pygame.draw.line(
        surface, color,
        (cx, cy + int(22 * scale)), (cx + int(16 * scale), cy + int(48 * scale)),
        max(2, int(5 * scale)),
    )


def frame_surface(
    frame: np.ndarray,
    player: int,
    num_players: int,
    size: tuple[int, int],
) -> tuple[pygame.Surface, tuple[int, int]]:
    """Recorta la franja del jugador (o el encuadre completo) y la ajusta sin deformar."""
    height, width = frame.shape[:2]
    part = max(1, width // num_players)
    crop = frame[:, player * part:(player + 1) * part]
    ch, cw = crop.shape[:2]
    scale = min(size[0] / cw, size[1] / ch)
    target = (max(1, int(cw * scale)), max(1, int(ch * scale)))
    resized = cv2.resize(crop, target, interpolation=cv2.INTER_AREA)
    rgb = np.ascontiguousarray(resized[:, :, ::-1])
    return pygame.image.frombuffer(rgb.tobytes(), target, "RGB"), target


def format_ms(value: int | None) -> str:
    """Tiempo en milisegundos -> texto (``1234 ms``)."""
    if value is None:
        return "—"
    return f"{int(round(value))} ms"


def points_label(points: int) -> str:
    return f"{points} pt" if points == 1 else f"{points} pts"


def draw_arena_header(
    surface: pygame.Surface,
    fonts: Fonts,
    title: str,
    num_players: int,
    *,
    chip: str | None = None,
) -> None:
    """Cabecera comun a las pantallas."""
    draw_text(surface, fonts, title, 30, TEXT, (MARGIN, 26), bold=True)
    draw_text(
        surface, fonts, "ESC · menú", 15, MUTED,
        (WIDTH - MARGIN - 214 - 52, 34), anchor="topright",
    )
    people = "1 persona" if num_players == 1 else f"{num_players} personas"
    draw_pill(
        surface, fonts,
        chip or f"1 cámara · {people}",
        (WIDTH - MARGIN - 52, 22), CARD_SOFT, MUTED, size=16, anchor="topright",
    )


def row_panels(
    count: int,
    top: int,
    height: int,
    *,
    gap: int = 16,
    max_width: int | None = None,
) -> list[pygame.Rect]:
    """Reparte ``count`` paneles en una fila centrada.

    Con un solo jugador el panel se limita a ``max_width`` para que no quede
    estirado de lado a lado del lienzo.
    """
    count = max(1, count)
    width = (WIDTH - 2 * MARGIN - gap * (count - 1)) // count
    if max_width is not None:
        width = min(width, max_width)
    total = width * count + gap * (count - 1)
    x0 = (WIDTH - total) // 2
    return [
        pygame.Rect(x0 + i * (width + gap), top, width, height)
        for i in range(count)
    ]


def draw_meter(
    surface: pygame.Surface,
    rect: pygame.Rect,
    value: float,
    color: tuple[int, int, int],
    background: tuple[int, int, int] = CARD_SOFT,
) -> None:
    """Barra horizontal 0..1 tipo medidor."""
    pygame.draw.rect(surface, background, rect, border_radius=rect.h // 2)
    value = max(0.0, min(1.0, float(value)))
    if value <= 0:
        return
    filled = pygame.Rect(rect.x, rect.y, max(rect.h, int(rect.w * value)), rect.h)
    pygame.draw.rect(surface, color, filled, border_radius=rect.h // 2)


CONFETTI_COLORS = (
    AMBER,
    GOOD,
    (108, 162, 255),
    (255, 94, 138),
    (110, 231, 183),
    (255, 255, 255),
    (255, 168, 76),
)


class Confetti:
    """Animacion generica de confeti para celebrar el cierre de una ronda.

    Cada papel es un rectangulo que sale disparado, cae con gravedad, gira y se
    desvanece hacia el color del fondo. No usa imagenes: sirve para cualquier
    pantalla del juego.
    """

    def __init__(self, *, gravity: float = 980.0) -> None:
        self.gravity = float(gravity)
        self._rng = random.Random()
        self._particles: list[dict] = []
        self._last_now: float | None = None

    @property
    def active(self) -> bool:
        return bool(self._particles)

    def burst(
        self,
        center: tuple[int, int],
        *,
        count: int = 130,
        speed: tuple[float, float] = (260.0, 720.0),
    ) -> None:
        """Lanza ``count`` papeles desde ``center``."""
        cx, cy = center
        for _ in range(count):
            angle = self._rng.uniform(-math.pi * 0.92, -math.pi * 0.08)
            velocity = self._rng.uniform(*speed)
            self._particles.append(
                {
                    "x": cx + self._rng.uniform(-40.0, 40.0),
                    "y": cy + self._rng.uniform(-18.0, 18.0),
                    "vx": math.cos(angle) * velocity,
                    "vy": math.sin(angle) * velocity,
                    "angle": self._rng.uniform(0.0, math.tau),
                    "spin": self._rng.uniform(-9.0, 9.0),
                    "size": self._rng.uniform(5.0, 12.0),
                    "life": self._rng.uniform(1.5, 3.0),
                    "age": 0.0,
                    "color": self._rng.choice(CONFETTI_COLORS),
                }
            )

    def clear(self) -> None:
        self._particles.clear()
        self._last_now = None

    def update(self, now: float) -> None:
        if self._last_now is None:
            self._last_now = now
        dt = max(0.0, min(0.05, now - self._last_now))
        self._last_now = now
        if dt <= 0 or not self._particles:
            return
        alive: list[dict] = []
        for particle in self._particles:
            particle["age"] += dt
            if particle["age"] >= particle["life"]:
                continue
            particle["vy"] += self.gravity * dt
            drag = max(0.0, 1.0 - 0.55 * dt)
            particle["vx"] *= drag
            particle["vy"] *= drag
            particle["x"] += particle["vx"] * dt
            particle["y"] += particle["vy"] * dt
            particle["angle"] += particle["spin"] * dt
            if particle["y"] > HEIGHT + 40:
                continue
            alive.append(particle)
        self._particles = alive

    def draw(self, surface: pygame.Surface) -> None:
        for particle in self._particles:
            fade = 1.0 - particle["age"] / particle["life"]
            color = tuple(
                int(BG[i] + (particle["color"][i] - BG[i]) * fade) for i in range(3)
            )
            # El ancho se encoge con el giro: da sensacion de papel que voltea.
            half_w = particle["size"] * (0.35 + 0.65 * abs(math.cos(particle["angle"] * 1.4))) / 2
            half_h = particle["size"] / 2
            cos_a, sin_a = math.cos(particle["angle"]), math.sin(particle["angle"])
            corners = []
            for dx, dy in (
                (-half_w, -half_h), (half_w, -half_h),
                (half_w, half_h), (-half_w, half_h),
            ):
                corners.append(
                    (
                        int(particle["x"] + dx * cos_a - dy * sin_a),
                        int(particle["y"] + dx * sin_a + dy * cos_a),
                    )
                )
            pygame.draw.polygon(surface, color, corners)


HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)


def draw_hand_landmarks(
    surface: pygame.Surface,
    viewport: pygame.Rect,
    image_size: tuple[int, int],
    landmarks: np.ndarray,
    *,
    index: int = 0,
    num_players: int = 1,
) -> None:
    """Superpone los 21 landmarks sobre la imagen ya escalada del recuadro.

    ``landmarks`` viene normalizado respecto del frame completo; la franja
    muestra ``1/num_players`` del ancho, asi que se reescala la X.
    """
    offset_x = viewport.x + (viewport.w - image_size[0]) / 2.0
    offset_y = viewport.y + (viewport.h - image_size[1]) / 2.0
    points: list[tuple[int, int]] = []
    for row in np.asarray(landmarks)[:21]:
        x = offset_x + (float(row[0]) * num_players - index) * image_size[0]
        y = offset_y + float(row[1]) * image_size[1]
        points.append((int(x), int(y)))

    for a, b in HAND_CONNECTIONS:
        pygame.draw.line(surface, (70, 214, 160), points[a], points[b], 2)
    for position, point in enumerate(points):
        radius = 5 if position == 0 else 3
        pygame.draw.circle(surface, (10, 16, 30), point, radius + 1)
        pygame.draw.circle(
            surface, AMBER if position == 0 else (240, 246, 255), point, radius
        )


def frame_cover(
    frame: np.ndarray,
    size: tuple[int, int],
) -> tuple[pygame.Surface, tuple[float, float]]:
    """Escala y recorta el frame para llenar ``size`` (sin deformar).

    Devuelve la imagen y el tramo horizontal visible del frame original, que
    hace falta para ubicar las franjas de los jugadores.
    """
    height, width = frame.shape[:2]
    target_w, target_h = size
    scale = max(target_w / width, target_h / height)
    new_w = max(target_w, int(round(width * scale)))
    new_h = max(target_h, int(round(height * scale)))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    x0 = (new_w - target_w) // 2
    y0 = (new_h - target_h) // 2
    crop = resized[y0:y0 + target_h, x0:x0 + target_w]
    rgb = np.ascontiguousarray(crop[:, :, ::-1])
    visible = (x0 / new_w, (x0 + target_w) / new_w)
    return pygame.image.frombuffer(rgb.tobytes(), (target_w, target_h), "RGB"), visible


def _top_mask(size: tuple[int, int], radius: int) -> pygame.Surface:
    """Mascara con las esquinas de arriba redondeadas (abajo recto)."""
    mask = pygame.Surface(size, pygame.SRCALPHA)
    mask.fill((0, 0, 0, 0))
    pygame.draw.rect(
        mask, (255, 255, 255, 255), mask.get_rect(), border_radius=radius
    )
    # Abajo recto: la barra de datos sigue sobre el fondo de la tarjeta.
    pygame.draw.rect(
        mask, (255, 255, 255, 255),
        pygame.Rect(0, size[1] // 2, size[0], size[1] - size[1] // 2),
    )
    return mask


def _dim_other_bands(
    surface: pygame.Surface,
    image_rect: pygame.Rect,
    index: int,
    num_players: int,
    color: tuple[int, int, int],
    visible: tuple[float, float] = (0.0, 1.0),
) -> None:
    """Atenua las franjas de los demas y marca el borde interno de la propia."""
    left_norm, right_norm = visible
    span = max(1e-6, right_norm - left_norm)

    def to_px(normalized: float) -> int:
        value = (normalized - left_norm) / span * image_rect.w
        return int(max(0.0, min(float(image_rect.w), value)))

    left = to_px(index / num_players)
    right = to_px((index + 1) / num_players)

    overlay = pygame.Surface(image_rect.size, pygame.SRCALPHA)
    overlay.fill((6, 10, 24, 150))
    overlay.fill((0, 0, 0, 0), pygame.Rect(left, 0, right - left, image_rect.h))
    surface.blit(overlay, image_rect.topleft)

    if index > 0:
        pygame.draw.line(
            surface, color,
            (image_rect.x + left, image_rect.y),
            (image_rect.x + left, image_rect.bottom), 2,
        )
    if index < num_players - 1:
        pygame.draw.line(
            surface, color,
            (image_rect.x + right, image_rect.y),
            (image_rect.x + right, image_rect.bottom), 2,
        )


def draw_camera_viewport(
    surface: pygame.Surface,
    fonts: Fonts,
    area: pygame.Rect,
    *,
    index: int,
    num_players: int,
    frame: np.ndarray | None,
    demo: bool,
    color: tuple[int, int, int],
    avatar_scale: float = 1.25,
    landmarks: np.ndarray | None = None,
) -> pygame.Rect:
    """Dibuja el encuadre completo dentro de ``area`` y devuelve su rectangulo.

    La imagen se ajusta al aspecto real de la camara (sin franjas vacias dentro
    del marco). Con mas de un jugador, la franja propia queda iluminada y el
    resto atenuado.
    """
    inner = PLAYER_INNER[index % len(PLAYER_INNER)]
    pygame.draw.rect(surface, inner, area, border_radius=14)

    if frame is not None:
        image, size = frame_surface(frame, 0, 1, area.size)
        rect = pygame.Rect((0, 0), size)
        rect.center = area.center
        masked = image.copy()
        masked.blit(rounded_mask(size, 12), (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
        surface.blit(masked, rect)
        if num_players > 1:
            _dim_other_bands(surface, rect, index, num_players, color)
        if landmarks is not None:
            draw_hand_landmarks(surface, rect, size, landmarks)
    else:
        rect = area
        draw_avatar(surface, (area.centerx, area.centery - 6), color, avatar_scale)
        caption = "sin cámara · modo demo" if demo else "sin señal de cámara"
        draw_text(
            surface, fonts, caption, 14, MUTED,
            (area.centerx, area.bottom - 12), anchor="midbottom",
        )

    pygame.draw.rect(surface, color, rect, width=2, border_radius=12)
    return rect


def draw_player_panel(
    surface: pygame.Surface,
    fonts: Fonts,
    rect: pygame.Rect,
    *,
    index: int,
    num_players: int,
    frame: np.ndarray | None,
    demo: bool,
    label: str | None,
    border: tuple[int, int, int] | None = None,
    bar_height: int = 66,
) -> pygame.Rect:
    """Ficha de un jugador: video arriba y una barra de datos abajo.

    Devuelve el rectangulo de la barra para que el llamador dibuje ahi el
    tiempo y el progreso.
    """
    color = PLAYER_COLORS[index % len(PLAYER_COLORS)]
    draw_card(surface, rect, CARD, radius=20)

    tile = pygame.Rect(rect.x, rect.y, rect.w, rect.h - bar_height)
    mask = _top_mask(tile.size, 18)

    if frame is not None:
        image, visible = frame_cover(frame, tile.size)
        masked = image.copy()
        masked.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
        surface.blit(masked, tile)
        if num_players > 1:
            _dim_other_bands(surface, tile, index, num_players, color, visible)
    else:
        fill = pygame.Surface(tile.size, pygame.SRCALPHA)
        fill.fill((16, 21, 40, 255))
        fill.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
        surface.blit(fill, tile)
        draw_avatar(surface, (tile.centerx, tile.centery - 6), color, 1.35)

    # Nombre y gesto arriba a la izquierda; la franja, arriba a la derecha.
    if label is not None:
        text = f"Jugador {index + 1} · {short_label(label)}"
        chip_bg, chip_fg = color, (14, 18, 34)
    else:
        text = f"Jugador {index + 1} · sin mano"
        chip_bg, chip_fg = (16, 20, 38), MUTED
    draw_pill(
        surface, fonts, text, (tile.x + 14, tile.y + 14),
        chip_bg, chip_fg, size=15, bold=True, anchor="topleft",
    )
    if num_players > 1:
        side = "izquierda" if index == 0 else "derecha"
        draw_text(
            surface, fonts, f"mitad {side}", 13, (226, 232, 246),
            (tile.right - 14, tile.y + 20), anchor="topright",
        )
    if frame is None:
        caption = "sin cámara · modo demo" if demo else "sin señal de cámara"
        draw_text(
            surface, fonts, caption, 13, MUTED,
            (tile.centerx, tile.bottom - 24), anchor="midbottom",
        )

    pygame.draw.rect(surface, border or BORDER, rect, width=2, border_radius=20)
    bar = pygame.Rect(rect.x, rect.bottom - bar_height, rect.w, bar_height)
    return pygame.Rect(bar.x + 18, bar.y + 6, bar.w - 36, bar.h - 12)


# ---------------------------------------------------------------------------
# Pantallas
# ---------------------------------------------------------------------------
class Screen:
    """Pantalla base: la app delega eventos, update y dibujo."""

    def __init__(self, app: "GestureRushApp") -> None:
        self.app = app
        self.now = 0.0

    def handle_event(self, event: pygame.event.Event) -> None:
        pass

    def update(self, now: float) -> None:
        self.now = now

    def draw(self) -> None:
        pass


class MenuScreen(Screen):
    """Menu de minijuegos."""

    CARDS = (
        {
            "key": "duel",
            "title": "Duelo de gestos",
            "desc": "El mismo gesto para todos: el primero en hacerlo se lleva el punto.",
            "tag": "1 cámara · 1 o 2 jugadores",
            "glyph": "fist",
            "enabled": True,
        },
        {
            "key": "chain",
            "title": "Cadena de gestos",
            "desc": "La misma secuencia para todos. El primero en completarla gana la carrera.",
            "tag": "1 cámara · 1 o 2 jugadores",
            "glyph": "victory",
            "enabled": True,
        },
        {
            "key": "practice",
            "title": "Prueba del modelo",
            "desc": "Sin reglas: mirá la predicción, el top 3 y el voto temporal del modelo.",
            "tag": "1 cámara · diagnóstico",
            "glyph": "thumbs_up",
            "enabled": True,
        },
    )

    def __init__(self, app: "GestureRushApp") -> None:
        super().__init__(app)
        card_w, card_h, gap = 368, 340, 24
        total = card_w * len(self.CARDS) + gap * (len(self.CARDS) - 1)
        x0 = (WIDTH - total) // 2
        self.rects = [
            pygame.Rect(x0 + i * (card_w + gap), 250, card_w, card_h)
            for i in range(len(self.CARDS))
        ]
        self.hover: int | None = None

        # Selectores del menu: jugadores y rondas.
        self.round_rects = self._segment_rects(
            WIDTH - MARGIN, 70, ((3, 58), (5, 58), (8, 58))
        )
        self.rounds_label_right = min(r.x for r in self.round_rects.values()) - 14
        self.player_rects = self._segment_rects(
            self.rounds_label_right - 78, 70, ((1, 112), (2, 128))
        )
        self.players_label_right = min(r.x for r in self.player_rects.values()) - 14

    # -- Selectores ----------------------------------------------------------
    @staticmethod
    def _segment_rects(right: int, y: int, specs) -> dict[int, pygame.Rect]:
        gap = 8
        total = sum(width for _, width in specs) + gap * (len(specs) - 1)
        x = right - total
        rects: dict[int, pygame.Rect] = {}
        for value, width in specs:
            rects[value] = pygame.Rect(x, y, width, 40)
            x += width + gap
        return rects

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE):
                self.app.start_game("duel")
            elif event.key == pygame.K_1:
                self.app.start_game("duel")
            elif event.key == pygame.K_2:
                self.app.start_game("chain")
            elif event.key == pygame.K_3:
                self.app.start_game("practice")
            elif event.key == pygame.K_p:
                self.app.set_players(1 if self.app.players == 2 else 2)
            elif event.key == pygame.K_r:
                cycle = (3, 5, 8)
                index = cycle.index(self.app.rounds) if self.app.rounds in cycle else 0
                self.app.set_rounds(cycle[(index + 1) % len(cycle)])
            elif event.key == pygame.K_ESCAPE:
                self.app.running = False
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for value, rect in self.player_rects.items():
                if rect.collidepoint(event.pos):
                    self.app.set_players(value)
                    return
            for value, rect in self.round_rects.items():
                if rect.collidepoint(event.pos):
                    self.app.set_rounds(value)
                    return
            for index, rect in enumerate(self.rects):
                card = self.CARDS[index]
                if rect.collidepoint(event.pos) and card["enabled"]:
                    self.app.start_game(card["key"])
                    return

    def update(self, now: float) -> None:
        super().update(now)
        mouse = pygame.mouse.get_pos()
        self.hover = next(
            (i for i, r in enumerate(self.rects) if r.collidepoint(mouse)), None
        )

    def _draw_selectors(self, surface: pygame.Surface) -> None:
        """Jugadores y rondas del duelo, elegidos desde el menu."""
        fonts = self.app.fonts
        draw_text(
            surface, fonts, "Jugadores", 15, MUTED,
            (self.players_label_right, 82), anchor="topright",
        )
        self._draw_segment(
            surface, self.player_rects, self.app.players,
            {1: "1 jugador", 2: "2 jugadores"},
        )
        draw_text(
            surface, fonts, "Rondas", 15, MUTED,
            (self.rounds_label_right, 82), anchor="topright",
        )
        self._draw_segment(
            surface, self.round_rects, self.app.rounds,
            {3: "3", 5: "5", 8: "8"},
        )

    def _draw_segment(
        self,
        surface: pygame.Surface,
        rects: dict[int, pygame.Rect],
        selected: int,
        texts: dict[int, str],
    ) -> None:
        fonts = self.app.fonts
        for value, rect in rects.items():
            is_selected = value == selected
            bg = AMBER if is_selected else CARD_SOFT
            fg = (26, 22, 8) if is_selected else MUTED
            pygame.draw.rect(surface, bg, rect, border_radius=20)
            draw_text(
                surface, fonts, texts[value], 15, fg, rect.center,
                bold=is_selected, anchor="center",
            )

    def draw(self) -> None:
        surface = self.app.screen
        fonts = self.app.fonts
        surface.fill(BG)

        draw_text(surface, fonts, "Gesture Rush", 46, TEXT, (MARGIN, 68), bold=True)
        draw_text(
            surface, fonts, "Elegí un minijuego", 22, MUTED, (MARGIN, 132)
        )
        self._draw_selectors(surface)

        for index, (card, rect) in enumerate(zip(self.CARDS, self.rects)):
            enabled = card["enabled"]
            hovered = enabled and self.hover == index
            border = PLAYER_COLORS[0] if hovered else (BORDER if enabled else (30, 36, 62))
            draw_card(surface, rect, CARD if enabled else (18, 22, 42), border=border)
            if hovered:
                glow = pygame.Rect(rect.x - 3, rect.y - 3, rect.w + 6, rect.h + 6)
                pygame.draw.rect(surface, (40, 58, 104), glow, width=3, border_radius=23)

            glyph = hand_glyph(card["glyph"], 120)
            if not enabled:
                glyph = glyph.copy()
                glyph.set_alpha(90)
            surface.blit(glyph, glyph.get_rect(center=(rect.centerx, rect.y + 120)))

            color = TEXT if enabled else (120, 130, 160)
            draw_text(
                surface, fonts, card["title"], 27, color,
                (rect.centerx, rect.y + 196), bold=True, anchor="midtop",
            )
            draw_paragraph(
                surface, fonts, card["desc"], 17, MUTED,
                (rect.x + 30, rect.y + 240), rect.w - 60,
            )

            if enabled:
                draw_pill(
                    surface, fonts, card["tag"], (rect.x + 26, rect.bottom - 46),
                    CARD_SOFT, MUTED,
                )
                draw_pill(
                    surface, fonts, "Jugar", (rect.right - 26, rect.bottom - 46),
                    AMBER, (26, 22, 8), size=16, bold=True, anchor="topright",
                )
            else:
                draw_pill(
                    surface, fonts, "Próximamente", (rect.x + 26, rect.bottom - 46),
                    (26, 30, 52), (110, 120, 150),
                )

        draw_text(
            surface, fonts,
            "1, 2, 3 minijuego · P jugadores · R rondas · V volumen · F11 pantalla · ESC salir",
            17, MUTED, (WIDTH // 2, 842), anchor="midtop",
        )

        # Guia corta, util para explicar el juego en la feria.
        guide = pygame.Rect(MARGIN + 20, 700, WIDTH - 2 * MARGIN - 40, 96)
        draw_card(surface, guide, (18, 22, 42), radius=18, border=(34, 42, 74))
        draw_text(
            surface, fonts, "Cómo jugar", 16, AMBER,
            (guide.x + 22, guide.y + 16), bold=True,
        )
        steps = (
            "Elegí el minijuego y cuántos juegan.",
            "Cada jugador se para frente a su parte del encuadre.",
            "Duelo y Cadena compiten; Prueba muestra lo que ve el modelo.",
        )
        for index, step in enumerate(steps):
            x = guide.x + 22 + index * ((guide.w - 44) // 3)
            draw_pill(
                surface, fonts, str(index + 1), (x, guide.y + 48),
                CARD_SOFT, AMBER, size=14, pad_x=10, pad_y=4, bold=True,
            )
            draw_paragraph(
                surface, fonts, step, 15, MUTED, (x + 36, guide.y + 46),
                (guide.w - 44) // 3 - 48, line_height=19,
            )


class ThumbsUpGate:
    """Inicia la partida cuando todos los jugadores levantan el pulgar.

    Se arma al entrar en una fase de espera. Si alguien ya venia con el pulgar
    arriba en ese momento, tiene que bajarlo y volver a levantarlo: asi una
    ronda no encadena sola cuando el gesto objetivo era el pulgar.
    """

    def __init__(self, num_players: int, gesture: str = "thumbs_up") -> None:
        self.num_players = max(1, num_players)
        self.gesture = gesture
        self.ready = [False] * self.num_players
        self._locked = [False] * self.num_players
        self._active = False

    def reset(self) -> None:
        self.ready = [False] * self.num_players
        self._locked = [False] * self.num_players
        self._active = False

    def update(self, labels: list[str | None], active: bool) -> bool:
        """Devuelve ``True`` cuando todos dieron el pulgar."""
        if not active:
            self.reset()
            return False

        showing = [
            (labels[i] if i < len(labels) else None) == self.gesture
            for i in range(self.num_players)
        ]
        if not self._active:
            # Primer frame de la espera: lo que ya estaba arriba queda bloqueado.
            self._active = True
            self._locked = list(showing)
            self.ready = [False] * self.num_players
            return False

        for index in range(self.num_players):
            if not showing[index]:
                self._locked[index] = False
                self.ready[index] = False
            elif not self._locked[index]:
                self.ready[index] = True
        return all(self.ready)

    def is_locked(self, player: int) -> bool:
        return self._locked[player]


def start_hint(gate: ThumbsUpGate) -> str:
    """Texto que explica a quien le falta el pulgar."""
    if gate.num_players == 1:
        return "Pulgar arriba para empezar"
    missing = [str(i + 1) for i in range(gate.num_players) if not gate.ready[i]]
    if not missing:
        return "¡Listo!"
    if len(missing) == gate.num_players:
        return "Los dos con el pulgar arriba para empezar"
    return f"Falta el pulgar del jugador {' y '.join(missing)}"


class DuelScreen(Screen):
    """Partida del modo clásico."""

    def __init__(self, app: "GestureRushApp") -> None:
        super().__init__(app)
        self.duel: Duel = app.duel
        self.frame: np.ndarray | None = None
        self.readings = []
        self.error: str | None = None

        self.hero = pygame.Rect(MARGIN, 78, WIDTH - 2 * MARGIN, 226)
        self.panels = row_panels(self.duel.num_players, 318, 348, max_width=620)
        self.button_rect = pygame.Rect(MARGIN, 704, 300, 64)
        self.note_rect = pygame.Rect(MARGIN, 782, 300, 102)
        self.history_rect = pygame.Rect(
            MARGIN + 316, 682, WIDTH - 2 * MARGIN - 316, 202
        )
        self.history_offset = 0
        self.hover_button = False
        self.gate = ThumbsUpGate(self.duel.num_players)
        self.confetti = Confetti()
        self._rounds_seen = 0

    # -- Historial -----------------------------------------------------------
    def _history_rows(self) -> int:
        return max(1, (self.history_rect.h - 62) // 26)

    def _history_max_offset(self) -> int:
        return max(0, len(self.duel.history) - self._history_rows())

    def _scroll_history(self, delta: int) -> None:
        self.history_offset = max(
            0, min(self._history_max_offset(), self.history_offset + delta)
        )

    # -- Eventos -------------------------------------------------------------
    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.app.show_menu()
            elif event.key in (pygame.K_SPACE, pygame.K_RETURN, pygame.K_KP_ENTER):
                self._start_round()
            elif event.key == pygame.K_r:
                self.duel.reset()
                self.confetti.clear()
                self._rounds_seen = 0
            elif event.key == pygame.K_UP:
                self._scroll_history(-1)
            elif event.key == pygame.K_DOWN:
                self._scroll_history(1)
            elif event.key == pygame.K_PAGEUP:
                self._scroll_history(-self._history_rows())
            elif event.key == pygame.K_PAGEDOWN:
                self._scroll_history(self._history_rows())
            elif event.key in (pygame.K_1, pygame.K_2, pygame.K_3):
                player = event.key - pygame.K_1
                force = getattr(self.app.pipeline, "force", None)
                if force is not None and player < self.duel.num_players:
                    force(player)
        elif event.type == pygame.MOUSEWHEEL:
            if self.history_rect.collidepoint(pygame.mouse.get_pos()):
                self._scroll_history(-event.y)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.button_rect.collidepoint(event.pos) and self._can_start():
                self._start_round()

    def _can_start(self) -> bool:
        return self.duel.phase in (PHASE_IDLE, PHASE_RESULT, PHASE_FINISHED)

    def _start_round(self) -> None:
        if self._can_start():
            self.duel.start(self.now)

    # -- Actualizacion -------------------------------------------------------
    def update(self, now: float) -> None:
        super().update(now)
        self.confetti.update(now)
        try:
            self.frame, self.readings = self.app.pipeline.read(now)
            self.error = None
        except Exception as exc:  # camara desconectada a mitad de partida
            self.frame, self.readings = None, []
            self.error = str(exc)

        self.duel.tick(now)
        labels: list[str | None] = [None] * self.duel.num_players
        for reading in self.readings:
            score = self.duel.players[reading.player]
            score.live_label = reading.label
            score.live_confidence = reading.confidence
            if 0 <= reading.player < len(labels):
                labels[reading.player] = reading.stable
            if reading.stable is not None:
                self.duel.observe(reading.player, reading.stable, reading.confidence, now)

        # Pulgar arriba de todos = iniciar (ademas del boton y de ENTER).
        if self.gate.update(labels, self._can_start()):
            self._start_round()

        self._celebrate_closed_round()

        mouse = pygame.mouse.get_pos()
        self.hover_button = self.button_rect.collidepoint(mouse) and self._can_start()

    def _celebrate_closed_round(self) -> None:
        """Lanza confeti cuando se cierra una ronda con ganador."""
        rounds = len(self.duel.history)
        if rounds < self._rounds_seen:
            self._rounds_seen = rounds  # partida reiniciada
            return
        if rounds <= self._rounds_seen:
            return
        self._rounds_seen = rounds

        record = self.duel.current_round
        if record is None or record.winner is None:
            return
        panel = self.panels[record.winner % len(self.panels)]
        self.confetti.burst((panel.centerx, panel.y + 110))
        if self.duel.phase == PHASE_FINISHED:
            self.confetti.burst((self.hero.centerx, self.hero.centery), count=90)

    # -- Dibujo --------------------------------------------------------------
    def draw(self) -> None:
        surface = self.app.screen
        surface.fill(BG)
        self._draw_header(surface)
        self._draw_hero(surface)
        for index, panel in enumerate(self.panels):
            self._draw_panel(surface, index, panel)
        self._draw_bottom(surface)
        self._draw_history(surface)
        self.confetti.draw(surface)

    def _draw_header(self, surface: pygame.Surface) -> None:
        draw_arena_header(surface, self.app.fonts, "Duelo de gestos", self.duel.num_players)

    def _draw_hero(self, surface: pygame.Surface) -> None:
        fonts = self.app.fonts
        duel = self.duel
        hero = self.hero
        draw_card(surface, hero, CARD, radius=22)
        cx = hero.centerx

        if duel.phase == PHASE_COUNTDOWN:
            value = duel.countdown_value(self.now)
            draw_text(
                surface, fonts, str(value), 120, AMBER,
                (cx, hero.centery - 34), bold=True, anchor="center",
            )
            draw_text(
                surface, fonts, "Preparen el gesto", 22, MUTED,
                (cx, hero.bottom - 36), anchor="midbottom",
            )
            return

        if duel.phase == PHASE_PLAYING:
            glyph = hand_glyph(duel.target, 96)
            surface.blit(glyph, glyph.get_rect(midtop=(cx, hero.y + 14)))
            draw_text(
                surface, fonts, short_label(duel.target), 44, TEXT,
                (cx, hero.y + 118), bold=True, anchor="midtop",
            )
            remaining = max(0.0, duel.round_timeout - duel.play_elapsed(self.now))
            draw_text(
                surface, fonts,
                f"Ronda {duel.round_number} de {duel.total_rounds or '∞'} · "
                f"{int(round(remaining * 1000))} ms",
                19, MUTED, (cx, hero.y + 174), anchor="midtop",
            )
            self._draw_progress(surface)
            return

        if duel.phase in (PHASE_RESULT, PHASE_FINISHED):
            record = duel.current_round
            glyph = hand_glyph(record.gesture if record else duel.target, 72)
            surface.blit(glyph, glyph.get_rect(midtop=(cx, hero.y + 10)))
            self._draw_result_text(surface, cx)
            return

        # PHASE_IDLE
        draw_text(
            surface, fonts, "Duelo de gestos", 44, TEXT,
            (cx, hero.y + 56), bold=True, anchor="midtop",
        )
        subtitle = (
            "Hacé el gesto objetivo lo más rápido que puedas."
            if duel.num_players == 1
            else "El primero en hacer el gesto objetivo se lleva el punto."
        )
        draw_text(surface, fonts, subtitle, 22, MUTED, (cx, hero.y + 122), anchor="midtop")
        players = "1 jugador" if duel.num_players == 1 else f"{duel.num_players} jugadores"
        draw_text(
            surface, fonts,
            f"{players} · {duel.total_rounds or '∞'} rondas",
            16, DIM, (cx, hero.y + 156), anchor="midtop",
        )
        draw_text(
            surface, fonts, start_hint(self.gate), 18, AMBER,
            (cx, hero.y + 184), bold=True, anchor="midtop",
        )

    def _draw_result_text(self, surface: pygame.Surface, cx: int) -> None:
        fonts = self.app.fonts
        duel = self.duel
        hero = self.hero
        solo = duel.num_players == 1

        if duel.phase == PHASE_FINISHED:
            winner = duel.match_winner
            if solo:
                title = "Partida completada"
                color = GOOD
            else:
                title = "Empate en la partida" if winner is None else f"Gana el jugador {winner + 1}"
                color = MUTED if winner is None else PLAYER_COLORS[winner]
            draw_text(surface, fonts, title, 44, color, (cx, hero.y + 92), bold=True, anchor="midtop")

            if solo:
                player = duel.players[0]
                detail = f"{duel.rounds_played} rondas jugadas"
                if player.best_ms is not None:
                    detail += f" · mejor tiempo {format_ms(player.best_ms)}"
                draw_text(
                    surface, fonts, detail, 20, MUTED,
                    (cx, hero.y + 152), anchor="midtop",
                )
            else:
                score = " · ".join(
                    f"{points_label(p.points)}" for p in duel.players
                )
                draw_text(
                    surface, fonts, f"Partida terminada · {score}",
                    20, MUTED, (cx, hero.y + 152), anchor="midtop",
                )
            draw_text(
                surface, fonts, start_hint(self.gate),
                18, AMBER, (cx, hero.y + 186), bold=True, anchor="midtop",
            )
            return

        winner = duel.winner
        record = duel.current_round
        if winner is None:
            draw_text(
                surface, fonts, "Nadie lo logró", 44, MUTED,
                (cx, hero.y + 92), bold=True, anchor="midtop",
            )
            draw_text(
                surface, fonts,
                f"Se acabó el tiempo. {start_hint(self.gate)}.",
                20, MUTED, (cx, hero.y + 152), anchor="midtop",
            )
            return

        if solo:
            draw_text(
                surface, fonts, "Ronda superada", 44, GOOD,
                (cx, hero.y + 92), bold=True, anchor="midtop",
            )
            time_ms = record.times_ms.get(0) if record else None
            draw_text(
                surface, fonts,
                f"Tiempo: {format_ms(time_ms)}. {start_hint(self.gate)}.",
                20, MUTED, (cx, hero.y + 152), anchor="midtop",
            )
            return

        draw_text(
            surface, fonts, f"Gana el jugador {winner + 1}", 44,
            PLAYER_COLORS[winner], (cx, hero.y + 92), bold=True, anchor="midtop",
        )
        margin = record.margin_ms if record else None
        if margin is not None:
            subtitle = f"Diferencia de {format_ms(margin)}. {start_hint(self.gate)}."
        else:
            subtitle = f"Sin rival en el tiempo. {start_hint(self.gate)}."
        draw_text(surface, fonts, subtitle, 20, MUTED, (cx, hero.y + 152), anchor="midtop")

    def _draw_progress(self, surface: pygame.Surface) -> None:
        duel = self.duel
        bar = pygame.Rect(self.hero.x + 44, self.hero.bottom - 24, self.hero.w - 88, 6)
        progress = min(1.0, duel.play_elapsed(self.now) / duel.round_timeout)
        pygame.draw.rect(surface, CARD_SOFT, bar, border_radius=3)
        if progress > 0:
            filled = pygame.Rect(bar.x, bar.y, max(4, int(bar.w * progress)), bar.h)
            pygame.draw.rect(surface, AMBER, filled, border_radius=3)

    def _draw_panel(self, surface: pygame.Surface, index: int, rect: pygame.Rect) -> None:
        fonts = self.app.fonts
        duel = self.duel
        score = duel.players[index]

        winner_now = (
            duel.phase in (PHASE_RESULT, PHASE_FINISHED) and score.won_last
        )
        stats = draw_player_panel(
            surface, fonts, rect,
            index=index,
            num_players=duel.num_players,
            frame=self.frame,
            demo=self.app.demo,
            label=score.live_label,
            border=PLAYER_COLORS[index % len(PLAYER_COLORS)] if winner_now else BORDER,
        )

        elapsed = duel.elapsed_ms(index, self.now) if duel.phase == PHASE_PLAYING else score.last_ms
        if duel.phase == PHASE_IDLE:
            elapsed = 0
        draw_text(
            surface, fonts, self._status_text(index), 13, (196, 206, 230),
            (stats.x, stats.y),
        )
        timer_rect = draw_text(
            surface, fonts, format_ms(elapsed), 32,
            TEXT if elapsed else (206, 214, 234),
            (stats.x, stats.y + 16), bold=True,
        )
        if winner_now:
            draw_check(surface, (timer_rect.right + 18, timer_rect.centery), 16)
        draw_text(
            surface, fonts, points_label(score.points), 19, (214, 221, 240),
            (stats.right, stats.y + 20), anchor="topright",
        )

    def _status_text(self, index: int) -> str:
        duel = self.duel
        score = duel.players[index]
        solo = duel.num_players == 1
        if self._can_start():
            if self.gate.ready[index]:
                return "Pulgar arriba · listo"
            if self.gate.is_locked(index):
                return "Bajá el pulgar y volvé a levantarlo"
            return "Levantá el pulgar para empezar"
        if duel.phase == PHASE_PLAYING:
            if duel.times_ms.get(index) is not None:
                return "¡Acertado!" if solo else "¡Acertado! esperando al rival"
            if score.misses >= 3:
                return "Gestos detectados que no coinciden"
            return "Buscando el gesto objetivo"
        if duel.phase in (PHASE_RESULT, PHASE_FINISHED):
            return "Ronda cerrada"
        if duel.phase == PHASE_COUNTDOWN:
            return "Preparado"
        return "Acercá la mano a la cámara" if solo else "Acercá la mano a tu mitad"

    def _draw_bottom(self, surface: pygame.Surface) -> None:
        fonts = self.app.fonts
        duel = self.duel
        can_start = self._can_start()

        if duel.phase in (PHASE_PLAYING, PHASE_COUNTDOWN):
            label = "Ronda en curso"
        elif duel.phase == PHASE_FINISHED:
            label = "Jugar de nuevo"
        elif duel.phase == PHASE_RESULT:
            label = "Iniciar otra ronda"
        else:
            label = "Iniciar ronda"

        bg = (240, 242, 250) if can_start else CARD_SOFT
        fg = (16, 20, 38) if can_start else MUTED
        if self.hover_button:
            bg = (255, 255, 255)
        pygame.draw.rect(surface, bg, self.button_rect, border_radius=32)
        draw_text(
            surface, fonts, label, 22, fg, self.button_rect.center,
            bold=True, anchor="center",
        )

        draw_card(surface, self.note_rect, (20, 25, 48), radius=16, border=(36, 44, 76))
        if duel.num_players == 1:
            note = (
                "Contrarreloj con una sola persona: cada ronda muestra un gesto "
                "y registra tu tiempo."
            )
        else:
            note = (
                "Una cámara sin sincronizar: cada persona ocupa una mitad del "
                "encuadre."
            )
        draw_paragraph(
            surface, fonts, note,
            13, MUTED, (self.note_rect.x + 14, self.note_rect.y + 12),
            self.note_rect.w - 28, line_height=18,
        )
        if self.error:
            draw_text(
                surface, fonts, self.error[:80], 12, BAD,
                (self.note_rect.x + 14, self.note_rect.bottom - 22),
            )

    def _draw_history(self, surface: pygame.Surface) -> None:
        """Tabla de rondas con scroll (la rueda o las flechas recorren)."""
        fonts = self.app.fonts
        duel = self.duel
        rect = self.history_rect
        draw_card(surface, rect, (20, 25, 48), radius=16, border=(36, 44, 76))

        top = rect.y + 12
        draw_text(surface, fonts, "Rondas anteriores", 15, TEXT, (rect.x + 18, top), bold=True)
        if len(duel.history) > self._history_rows():
            draw_text(
                surface, fonts, "rueda o ↑ ↓ para recorrer", 13, MUTED,
                (rect.right - 18, top + 2), anchor="topright",
            )

        header_y = rect.y + 38
        draw_text(surface, fonts, "Ronda", 14, MUTED, (rect.x + 18, header_y))
        draw_text(surface, fonts, "Gesto", 14, MUTED, (rect.x + 96, header_y))

        first_x = rect.x + 330
        last_x = rect.right - 240
        columns = max(1, duel.num_players)
        spacing = (last_x - first_x) / columns if columns else 0
        for i in range(duel.num_players):
            draw_text(
                surface, fonts, f"Jugador {i + 1}", 14, MUTED,
                (int(first_x + i * spacing), header_y),
            )
        if duel.num_players > 1:
            draw_text(
                surface, fonts, "Gana", 14, MUTED,
                (rect.right - 18, header_y), anchor="topright",
            )
        pygame.draw.line(
            surface, (38, 46, 78),
            (rect.x + 18, header_y + 24), (rect.right - 18, header_y + 24), 1,
        )

        if not duel.history:
            draw_text(
                surface, fonts, "Todavía no hay rondas jugadas.", 15, DIM,
                (rect.x + 18, header_y + 36),
            )
            return

        visible = self._history_rows()
        self.history_offset = max(0, min(self._history_max_offset(), self.history_offset))
        rows = list(reversed(duel.history))[
            self.history_offset:self.history_offset + visible
        ]
        for row_index, record in enumerate(rows):
            y = header_y + 36 + row_index * 26
            draw_text(surface, fonts, str(record.index), 16, TEXT, (rect.x + 18, y))
            glyph = hand_glyph(record.gesture, 20)
            surface.blit(glyph, (rect.x + 94, y - 2))
            draw_text(
                surface, fonts, short_label(record.gesture), 16, TEXT, (rect.x + 120, y)
            )
            for player in range(duel.num_players):
                value = record.times_ms.get(player)
                draw_text(
                    surface, fonts, format_ms(value), 16,
                    TEXT if value is not None else DIM,
                    (int(first_x + player * spacing), y),
                )
            if duel.num_players > 1:
                if record.winner is None:
                    draw_text(
                        surface, fonts, "—", 16, DIM,
                        (rect.right - 18, y), anchor="topright",
                    )
                else:
                    draw_text(
                        surface, fonts, f"Jugador {record.winner + 1}", 16,
                        PLAYER_COLORS[record.winner % len(PLAYER_COLORS)],
                        (rect.right - 18, y), bold=True, anchor="topright",
                    )

        # Barra de scroll.
        if self._history_max_offset() > 0:
            track = pygame.Rect(rect.right - 10, header_y + 30, 4, rect.h - 48)
            pygame.draw.rect(surface, (36, 44, 76), track, border_radius=2)
            fraction = visible / len(duel.history)
            thumb_h = max(24, int(track.h * fraction))
            span = max(1, self._history_max_offset())
            thumb_y = track.y + int((track.h - thumb_h) * self.history_offset / span)
            pygame.draw.rect(
                surface, (110, 126, 176),
                pygame.Rect(track.x, thumb_y, track.w, thumb_h), border_radius=2,
            )


# ---------------------------------------------------------------------------
# Modo cadena
# ---------------------------------------------------------------------------
class ChainScreen(Screen):
    """Carrera por una cadena de gestos: la misma lista para los dos."""

    def __init__(self, app: "GestureRushApp") -> None:
        super().__init__(app)
        self.race: ChainRace = app.chain
        self.frame: np.ndarray | None = None
        self.readings = []
        self.error: str | None = None

        self.hero = pygame.Rect(MARGIN, 78, WIDTH - 2 * MARGIN, 226)
        self.panels = row_panels(self.race.num_players, 318, 348, max_width=620)
        self.button_rect = pygame.Rect(MARGIN, 704, 300, 64)
        self.note_rect = pygame.Rect(MARGIN + 316, 682, WIDTH - 2 * MARGIN - 316, 108)
        self.hover_button = False
        self.gate = ThumbsUpGate(self.race.num_players)
        self.confetti = Confetti()
        self._finished_seen = False

    # -- Eventos -------------------------------------------------------------
    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.app.show_menu()
            elif event.key in (pygame.K_SPACE, pygame.K_RETURN, pygame.K_KP_ENTER):
                self._start_race()
            elif event.key == pygame.K_r:
                self.race.reset()
                self.confetti.clear()
                self._finished_seen = False
            elif event.key in (pygame.K_1, pygame.K_2, pygame.K_3):
                player = event.key - pygame.K_1
                force = getattr(self.app.pipeline, "force", None)
                if force is not None and player < self.race.num_players:
                    force(player)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.button_rect.collidepoint(event.pos) and self._can_start():
                self._start_race()

    def _can_start(self) -> bool:
        return self.race.phase in (PHASE_IDLE, PHASE_RESULT, PHASE_FINISHED)

    def _start_race(self) -> None:
        if self._can_start():
            self.race.start(self.now)

    # -- Actualizacion -------------------------------------------------------
    def update(self, now: float) -> None:
        super().update(now)
        self.confetti.update(now)
        try:
            self.frame, self.readings = self.app.pipeline.read(now)
            self.error = None
        except Exception as exc:  # camara desconectada a mitad de carrera
            self.frame, self.readings = None, []
            self.error = str(exc)

        self.race.tick(now)
        labels: list[str | None] = [None] * self.race.num_players
        for reading in self.readings:
            score = self.race.players[reading.player]
            score.live_label = reading.label
            score.live_confidence = reading.confidence
            if 0 <= reading.player < len(labels):
                labels[reading.player] = reading.stable
            if reading.stable is not None:
                self.race.observe(reading.player, reading.stable, reading.confidence, now)

        # Pulgar arriba de todos = iniciar la carrera.
        if self.gate.update(labels, self._can_start()):
            self._start_race()

        self._celebrate_finish()

        mouse = pygame.mouse.get_pos()
        self.hover_button = self.button_rect.collidepoint(mouse) and self._can_start()

    def _celebrate_finish(self) -> None:
        """Lanza confeti la primera vez que la carrera termina con ganador."""
        if self.race.phase != PHASE_FINISHED:
            self._finished_seen = False
            return
        if self._finished_seen:
            return
        self._finished_seen = True
        if self.race.winner is not None:
            panel = self.panels[self.race.winner % len(self.panels)]
            self.confetti.burst((panel.centerx, panel.y + 110))
            self.confetti.burst((self.hero.centerx, self.hero.centery), count=80)

    # -- Dibujo --------------------------------------------------------------
    def draw(self) -> None:
        surface = self.app.screen
        surface.fill(BG)
        draw_arena_header(surface, self.app.fonts, "Cadena de gestos", self.race.num_players)
        self._draw_hero(surface)
        for index, panel in enumerate(self.panels):
            self._draw_panel(surface, index, panel)
        self._draw_bottom(surface)
        self._draw_scoreboard(surface)
        self.confetti.draw(surface)

    def _draw_hero(self, surface: pygame.Surface) -> None:
        fonts = self.app.fonts
        race = self.race
        hero = self.hero
        draw_card(surface, hero, CARD, radius=22)
        cx = hero.centerx

        if race.phase == PHASE_COUNTDOWN:
            draw_text(
                surface, fonts, str(race.countdown_value(self.now)), 120, AMBER,
                (cx, hero.centery - 34), bold=True, anchor="center",
            )
            draw_text(
                surface, fonts,
                "Prepará la cadena" if race.num_players == 1 else "Preparen la cadena",
                22, MUTED, (cx, hero.bottom - 36), anchor="midbottom",
            )
            return

        if not race.chain:
            draw_text(
                surface, fonts, "Cadena de gestos", 44, TEXT,
                (cx, hero.y + 52), bold=True, anchor="midtop",
            )
            subtitle = (
                "Completá la cadena en orden lo más rápido que puedas."
                if race.num_players == 1
                else "La misma secuencia para los dos: el primero en completarla gana."
            )
            draw_text(surface, fonts, subtitle, 22, MUTED, (cx, hero.y + 118), anchor="midtop")
            draw_text(
                surface, fonts, f"{race.chain_length} gestos en orden",
                16, DIM, (cx, hero.y + 156), anchor="midtop",
            )
            draw_text(
                surface, fonts, start_hint(self.gate), 18, AMBER,
                (cx, hero.y + 184), bold=True, anchor="midtop",
            )
            return

        self._draw_chain_strip(surface, hero.y + 14)
        if race.phase == PHASE_FINISHED:
            self._draw_winner(surface, cx)
        else:
            caption = (
                "Completá la cadena antes de que se acabe el tiempo"
                if race.num_players == 1
                else "El primero en completar la cadena gana"
            )
            draw_text(surface, fonts, caption, 19, MUTED, (cx, hero.y + 112), anchor="midtop")

    def _draw_chain_strip(self, surface: pygame.Surface, top: int) -> None:
        fonts = self.app.fonts
        race = self.race
        steps = max(1, race.num_steps)
        size = 58
        step = min(118, (self.hero.w - 180) // steps)
        x0 = self.hero.centerx - (step * (steps - 1)) // 2
        for i, gesture in enumerate(race.chain):
            x = x0 + i * step
            glyph = hand_glyph(gesture, size)
            surface.blit(glyph, glyph.get_rect(midtop=(x, top)))
            draw_text(
                surface, fonts, short_label(gesture), 13, MUTED,
                (x, top + size + 2), anchor="midtop",
            )

    def _draw_winner(self, surface: pygame.Surface, cx: int) -> None:
        fonts = self.app.fonts
        race = self.race
        hero = self.hero
        winner = race.winner

        if winner is None:
            draw_text(
                surface, fonts, "Nadie completó la cadena", 38, MUTED,
                (cx, hero.y + 102), bold=True, anchor="midtop",
            )
            draw_text(
                surface, fonts, "Se acabó el tiempo.", 19, MUTED,
                (cx, hero.y + 160), anchor="midtop",
            )
            draw_text(
                surface, fonts, start_hint(self.gate), 18, AMBER,
                (cx, hero.y + 194), bold=True, anchor="midtop",
            )
            return

        score = race.players[winner]
        if race.num_players == 1:
            draw_text(
                surface, fonts, "Cadena completada", 40, GOOD,
                (cx, hero.y + 102), bold=True, anchor="midtop",
            )
            detail = (
                f"{race.num_steps} gestos en {format_ms(score.finished_ms)}"
                f" · media {format_ms(race.average_step_ms(winner))}"
            )
            draw_text(surface, fonts, detail, 19, MUTED, (cx, hero.y + 158), anchor="midtop")
            draw_text(
                surface, fonts, start_hint(self.gate), 18, AMBER,
                (cx, hero.y + 192), bold=True, anchor="midtop",
            )
            return

        draw_text(
            surface, fonts, f"Gana el jugador {winner + 1}", 40,
            PLAYER_COLORS[winner % len(PLAYER_COLORS)],
            (cx, hero.y + 102), bold=True, anchor="midtop",
        )
        detail = f"{race.num_steps} gestos en {format_ms(score.finished_ms)}"
        rival = next((p for p in race.players if p.index != winner), None)
        if rival is not None:
            if rival.done and rival.finished_ms is not None:
                margin = abs(rival.finished_ms - (score.finished_ms or 0))
                detail += f" · diferencia de {format_ms(margin)}"
            else:
                detail += f" · el jugador {rival.index + 1} quedó en {rival.progress}/{race.num_steps}"
        draw_text(surface, fonts, detail, 19, MUTED, (cx, hero.y + 158), anchor="midtop")
        draw_text(
            surface, fonts, start_hint(self.gate), 18, AMBER,
            (cx, hero.y + 192), bold=True, anchor="midtop",
        )

    def _draw_panel(self, surface: pygame.Surface, index: int, rect: pygame.Rect) -> None:
        fonts = self.app.fonts
        race = self.race
        score = race.players[index]
        winner_now = race.phase == PHASE_FINISHED and race.winner == index

        stats = draw_player_panel(
            surface, fonts, rect,
            index=index,
            num_players=race.num_players,
            frame=self.frame,
            demo=self.app.demo,
            label=score.live_label,
            border=PLAYER_COLORS[index % len(PLAYER_COLORS)] if winner_now else BORDER,
        )

        elapsed = race.elapsed_ms(index, self.now)
        if race.phase == PHASE_IDLE:
            elapsed = 0
        timer_rect = draw_text(
            surface, fonts, format_ms(elapsed), 32,
            TEXT if elapsed else (206, 214, 234),
            (stats.x, stats.y + 12), bold=True,
        )
        if winner_now:
            draw_check(surface, (timer_rect.right + 18, timer_rect.centery), 16)
        draw_text(
            surface, fonts, f"{score.progress}/{race.num_steps or race.chain_length}",
            19, (214, 221, 240), (stats.right, stats.y + 16), anchor="topright",
        )
        self._draw_progress_dots(
            surface, timer_rect.right + 26, timer_rect.centery, score
        )

    def _draw_progress_dots(
        self, surface: pygame.Surface, x0: int, y: int, score
    ) -> None:
        race = self.race
        total = max(1, race.num_steps or race.chain_length)
        for i in range(total):
            x = x0 + i * 24
            if i < score.progress:
                pygame.draw.circle(surface, GOOD, (x, y), 9)
                draw_check(surface, (x, y), 10, (10, 18, 12), 2)
            elif i == score.progress and race.phase == PHASE_PLAYING:
                pygame.draw.circle(surface, AMBER, (x, y), 9, 2)
            else:
                pygame.draw.circle(surface, (52, 62, 100), (x, y), 8)

    def _draw_bottom(self, surface: pygame.Surface) -> None:
        fonts = self.app.fonts
        race = self.race
        can_start = self._can_start()

        if race.phase in (PHASE_PLAYING, PHASE_COUNTDOWN):
            label = "Carrera en curso"
        elif race.phase == PHASE_FINISHED:
            label = "Otra carrera"
        else:
            label = "Iniciar carrera"

        bg = (240, 242, 250) if can_start else CARD_SOFT
        fg = (16, 20, 38) if can_start else MUTED
        if self.hover_button:
            bg = (255, 255, 255)
        pygame.draw.rect(surface, bg, self.button_rect, border_radius=32)
        draw_text(
            surface, fonts, label, 22, fg, self.button_rect.center,
            bold=True, anchor="center",
        )

        draw_card(surface, self.note_rect, (20, 25, 48), radius=16, border=(36, 44, 76))
        if race.num_players == 1:
            note = (
                "Completá la cadena en orden: cada acierto avanza una posición y "
                "fallar solo cuesta tiempo. Si no llegás al final antes del "
                "tiempo límite, la carrera queda desierta."
            )
        else:
            note = (
                "Misma cadena para los dos. Cada acierto avanza una posición y fallar "
                "solo cuesta tiempo. Gana quien completa el último gesto antes; si "
                "llegan los dos, gana el tiempo total más bajo."
            )
        draw_paragraph(
            surface, fonts, note,
            15, MUTED, (self.note_rect.x + 18, self.note_rect.y + 16),
            min(self.note_rect.w - 36, 760), line_height=23,
        )
        if self.error:
            draw_text(
                surface, fonts, self.error[:110], 14, BAD,
                (self.note_rect.x + 18, self.note_rect.bottom - 26),
            )

    def _draw_scoreboard(self, surface: pygame.Surface) -> None:
        fonts = self.app.fonts
        race = self.race
        top = 806
        spread = WIDTH - 2 * MARGIN
        columns = (
            MARGIN + 10,
            MARGIN + int(spread * 0.26),
            MARGIN + int(spread * 0.50),
            MARGIN + int(spread * 0.74),
        )
        draw_text(surface, fonts, "Jugador", 15, MUTED, (columns[0], top))
        draw_text(surface, fonts, "Cadena", 15, MUTED, (columns[1], top))
        draw_text(surface, fonts, "Tiempo total", 15, MUTED, (columns[2], top))
        draw_text(surface, fonts, "Media por gesto", 15, MUTED, (columns[3], top))
        pygame.draw.line(
            surface, (38, 46, 78),
            (MARGIN, top + 24), (WIDTH - MARGIN, top + 24), 1,
        )

        if not race.chain:
            draw_text(
                surface, fonts, "La cadena se sortea al iniciar la carrera.", 16, DIM,
                (MARGIN + 10, top + 38),
            )
            return

        for row, score in enumerate(race.players):
            y = top + 40 + row * 26
            color = PLAYER_COLORS[score.index % len(PLAYER_COLORS)]
            draw_text(
                surface, fonts, f"Jugador {score.index + 1}", 17, color,
                (columns[0], y), bold=True,
            )
            draw_text(
                surface, fonts, f"{score.progress}/{race.num_steps}", 17, TEXT,
                (columns[1], y),
            )
            draw_text(
                surface, fonts, format_ms(race.elapsed_ms(score.index, self.now)),
                17, TEXT if score.done or race.phase == PHASE_PLAYING else DIM,
                (columns[2], y),
            )
            draw_text(
                surface, fonts, format_ms(race.average_step_ms(score.index)), 17, MUTED,
                (columns[3], y),
            )


# ---------------------------------------------------------------------------
# Modo prueba
# ---------------------------------------------------------------------------
class PracticeScreen(Screen):
    """Diagnostico del modelo: prediccion cruda, top 3 y voto temporal.

    Una sola vista de camara, sin reglas ni puntaje. Existe para explicar como
    decide el clasificador y como lo estabiliza el voto temporal antes de que el
    juego acepte un gesto.
    """

    def __init__(self, app: "GestureRushApp") -> None:
        super().__init__(app)
        self.practice: PracticeSession = app.practice
        self.frame: np.ndarray | None = None
        self.readings: list[PlayerReading] = []
        self.error: str | None = None
        self.threshold = float(config.CONFIDENCE_THRESHOLD)
        self.window = int(config.SMOOTHING_WINDOW)
        self.min_votes = int(config.SMOOTHING_MIN_VOTES)
        self.show_landmarks = True

        self.camera_rect = pygame.Rect(MARGIN, 78, 1000, 562)
        self.side_rect = pygame.Rect(MARGIN + 1016, 78, 536, 562)
        self.note_rect = pygame.Rect(MARGIN, 656, WIDTH - 2 * MARGIN, 228)

    # -- Eventos -------------------------------------------------------------
    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.app.show_menu()
            elif event.key == pygame.K_l:
                self.show_landmarks = not self.show_landmarks
            elif event.key == pygame.K_r:
                reset = getattr(self.app.pipeline, "reset", None)
                if callable(reset):
                    reset()

    # -- Actualizacion -------------------------------------------------------
    def update(self, now: float) -> None:
        super().update(now)
        try:
            self.frame, self.readings = self.app.pipeline.read(now, probe=True)
            self.error = None
        except Exception as exc:  # camara desconectada a mitad de la prueba
            self.frame, self.readings = None, []
            self.error = str(exc)

    @property
    def reading(self) -> PlayerReading:
        return self.readings[0] if self.readings else PlayerReading(player=0)

    # -- Dibujo --------------------------------------------------------------
    def draw(self) -> None:
        surface = self.app.screen
        surface.fill(BG)
        draw_arena_header(
            surface, self.app.fonts, "Prueba del modelo", 1,
            chip="1 cámara · diagnóstico",
        )
        self._draw_preview(surface)
        self._draw_metrics(surface)
        self._draw_notes(surface)

    def _draw_preview(self, surface: pygame.Surface) -> None:
        fonts = self.app.fonts
        rect = self.camera_rect
        reading = self.reading
        draw_card(surface, rect, CARD, radius=20)
        draw_text(
            surface, fonts, "Cámara única", 16, TEXT,
            (rect.x + 20, rect.y + 18), bold=True,
        )

        if not reading.present:
            hand, hand_color = "sin mano", MUTED
        elif reading.handedness:
            side = str(reading.handedness).lower()
            hand = "mano derecha" if side.startswith("r") else "mano izquierda"
            hand_color = AMBER
        else:
            hand, hand_color = "mano detectada", AMBER
        draw_pill(
            surface, fonts, hand, (rect.right - 20, rect.y + 16),
            CARD_SOFT, hand_color, size=14, anchor="topright",
        )

        viewport = pygame.Rect(rect.x + 20, rect.y + 50, rect.w - 40, 484)
        draw_camera_viewport(
            surface, fonts, viewport,
            index=0, num_players=1,
            frame=self.frame, demo=self.app.demo,
            color=(64, 76, 114), avatar_scale=1.9,
            landmarks=reading.landmarks if self.show_landmarks else None,
        )
        draw_text(
            surface, fonts, "Se clasifica la mano más centrada del encuadre.",
            13, MUTED, (rect.x + 20, viewport.bottom + 14),
        )
        hint = (
            "21 landmarks · L para ocultar"
            if self.show_landmarks
            else "landmarks ocultos · L para mostrar"
        )
        draw_text(
            surface, fonts, hint, 13, DIM,
            (rect.right - 20, viewport.bottom + 14), anchor="topright",
        )

    def _separator(self, surface: pygame.Surface, rect: pygame.Rect, offset: int) -> None:
        pygame.draw.line(
            surface, (38, 46, 78),
            (rect.x + 20, rect.y + offset), (rect.right - 20, rect.y + offset), 1,
        )

    def _draw_metrics(self, surface: pygame.Surface) -> None:
        fonts = self.app.fonts
        rect = self.side_rect
        reading = self.reading
        x, right = rect.x, rect.right
        draw_card(surface, rect, CARD, radius=20)

        # Prediccion cruda del frame.
        draw_text(surface, fonts, "PREDICCIÓN CRUDA", 13, MUTED, (x + 20, rect.y + 16))
        present = reading.present and reading.label is not None
        confidence = reading.confidence if present else 0.0
        conf_color = GOOD if confidence >= self.threshold else AMBER
        if present:
            glyph = hand_glyph(reading.label, 54)
            surface.blit(glyph, glyph.get_rect(center=(x + 50, rect.y + 62)))
            draw_text(
                surface, fonts, short_label(reading.label), 30, TEXT,
                (x + 84, rect.y + 38), bold=True,
            )
            draw_text(
                surface, fonts, f"{confidence * 100:.1f} %".replace(".", ","),
                26, conf_color, (right - 20, rect.y + 42),
                bold=True, anchor="topright",
            )
        else:
            draw_text(surface, fonts, "sin mano", 28, DIM, (x + 20, rect.y + 40), bold=True)
            draw_text(
                surface, fonts, "—", 28, DIM,
                (right - 20, rect.y + 42), bold=True, anchor="topright",
            )

        meter = pygame.Rect(x + 20, rect.y + 106, rect.w - 40, 10)
        draw_meter(surface, meter, confidence, conf_color if present else (52, 62, 100))
        tick = meter.x + int(meter.w * self.threshold)
        pygame.draw.line(surface, TEXT, (tick, meter.y - 5), (tick, meter.bottom + 5), 2)
        draw_text(
            surface, fonts,
            f"confianza · umbral {self.threshold * 100:.0f} % marcado",
            13, MUTED, (x + 20, meter.bottom + 8),
        )

        self._separator(surface, rect, 142)

        # Top 3: reparto de probabilidad entre clases.
        draw_text(
            surface, fonts, "TOP 3 DEL CLASIFICADOR", 13, MUTED,
            (x + 20, rect.y + 152),
        )
        top = reading.top or []
        if top:
            for row, (label, probability) in enumerate(top[:3]):
                y = rect.y + 174 + row * 32
                glyph = hand_glyph(label, 20)
                surface.blit(glyph, (x + 22, y + 2))
                draw_text(
                    surface, fonts, short_label(label), 15,
                    TEXT if row == 0 else MUTED, (x + 52, y),
                )
                bar = pygame.Rect(x + 150, y + 6, 170, 8)
                draw_meter(
                    surface, bar, probability,
                    AMBER if row == 0 else (70, 84, 130),
                )
                draw_text(
                    surface, fonts, f"{probability * 100:.0f} %", 14, MUTED,
                    (right - 20, y), anchor="topright",
                )
        else:
            draw_text(
                surface, fonts, "Sin predicción: no hay mano.", 15, DIM,
                (x + 22, rect.y + 178),
            )

        self._separator(surface, rect, 284)

        # Voto temporal: cuantos frames sostienen cada etiqueta.
        draw_text(surface, fonts, "VOTO TEMPORAL", 13, MUTED, (x + 20, rect.y + 294))
        if reading.vote_label is not None:
            draw_text(
                surface, fonts,
                f"{short_label(reading.vote_label)} · "
                f"{reading.vote_count} de {reading.vote_total} votos",
                16, TEXT, (x + 20, rect.y + 314),
            )
        else:
            draw_text(
                surface, fonts, "sin votos todavía", 16, DIM,
                (x + 20, rect.y + 314),
            )
        for slot in range(self.window):
            dx = x + 26 + slot * 22
            if slot < reading.vote_count:
                pygame.draw.circle(surface, GOOD, (dx, rect.y + 344), 7)
            elif slot < reading.vote_total:
                pygame.draw.circle(surface, MUTED, (dx, rect.y + 344), 6, 2)
            else:
                pygame.draw.circle(surface, (48, 58, 92), (dx, rect.y + 344), 5)
        draw_text(
            surface, fonts,
            f"{self.window} frames (1 voto cada uno) · confirma con {self.min_votes}",
            13, MUTED, (x + 20, rect.y + 358),
        )

        self._separator(surface, rect, 386)

        # Lo que el juego aceptaria con esta lectura.
        draw_text(
            surface, fonts, "EL JUEGO ACEPTARÍA", 13, MUTED,
            (x + 20, rect.y + 396),
        )
        if reading.stable is not None:
            glyph = hand_glyph(reading.stable, 38)
            surface.blit(glyph, glyph.get_rect(midleft=(x + 24, rect.y + 424)))
            draw_text(
                surface, fonts, short_label(reading.stable), 22, GOOD,
                (x + 72, rect.y + 412), bold=True,
            )
            draw_pill(
                surface, fonts, "confirmado", (right - 20, rect.y + 408),
                GOOD, (10, 20, 14), size=13, bold=True, anchor="topright",
            )
        else:
            draw_text(surface, fonts, "—", 22, DIM, (x + 20, rect.y + 412), bold=True)
            draw_pill(
                surface, fonts, "sin confirmar", (right - 20, rect.y + 408),
                CARD_SOFT, MUTED, size=13, anchor="topright",
            )

        self._separator(surface, rect, 462)

        # Metricas del pipeline.
        draw_text(surface, fonts, "RENDIMIENTO", 13, MUTED, (x + 20, rect.y + 472))
        fps = float(getattr(self.app.pipeline, "fps", 0.0) or 0.0)
        tracker = getattr(self.app.pipeline, "tracker", None)
        inference = float(getattr(tracker, "last_inference_ms", 0.0) or 0.0)
        fps_text = f"FPS {fps:.1f}".replace(".", ",") if fps > 0 else "FPS —"
        inference_text = (
            f"Inferencia {inference:.1f} ms".replace(".", ",")
            if inference > 0
            else "Inferencia —"
        )
        draw_text(surface, fonts, fps_text, 16, TEXT, (x + 20, rect.y + 492), bold=True)
        draw_text(
            surface, fonts, inference_text, 16, TEXT,
            (right - 20, rect.y + 492), bold=True, anchor="topright",
        )
        draw_text(
            surface, fonts,
            f"Ventana {self.window} · mín. {self.min_votes} votos · "
            f"umbral {self.threshold * 100:.0f} %",
            13, MUTED, (x + 20, rect.y + 518),
        )
        draw_text(
            surface, fonts, "Umbral alto: más estricto y más lento.",
            13, AMBER, (x + 20, rect.y + 538),
        )
        if self.error:
            draw_text(
                surface, fonts, self.error[:52], 12, BAD, (x + 20, rect.y + 556),
            )

    def _draw_notes(self, surface: pygame.Surface) -> None:
        fonts = self.app.fonts
        rect = self.note_rect
        draw_card(surface, rect, (20, 25, 48), radius=16, border=(36, 44, 76))
        draw_text(
            surface, fonts, "Cómo leer esta pantalla", 16, AMBER,
            (rect.x + 22, rect.y + 16), bold=True,
        )

        column_w = (rect.w - 44 - 40) // 2
        left_x = rect.x + 22
        right_x = left_x + column_w + 40
        draw_text(
            surface, fonts, "Del frame al gesto", 15, TEXT,
            (left_x, rect.y + 50), bold=True,
        )
        draw_paragraph(
            surface, fonts,
            "MediaPipe entrega 21 landmarks de la mano; se normalizan (muñeca al "
            "origen, escala por la palma) y el clasificador devuelve una "
            "probabilidad por clase. La predicción cruda es la más alta y el "
            "voto temporal solo la confirma si se repite 5 de 7 frames con "
            f"confianza ≥ {self.threshold * 100:.0f} %.",
            16, MUTED, (left_x, rect.y + 76), column_w, line_height=23,
        )
        draw_text(
            surface, fonts, "Qué mirar", 15, TEXT,
            (right_x, rect.y + 50), bold=True,
        )
        draw_paragraph(
            surface, fonts,
            "Top 3 repartido = el modelo duda entre gestos parecidos. Confianza "
            "bajo el umbral o votos incompletos = todavía no hay gesto "
            "confirmado. FPS mide la captura; la inferencia mide solo la "
            "clasificación. R reinicia el voto; ESC vuelve al menú.",
            16, MUTED, (right_x, rect.y + 76), column_w, line_height=23,
        )


# ---------------------------------------------------------------------------
# Volumen
# ---------------------------------------------------------------------------
class VolumeControl:
    """Altavoz de la esquina: al pulsarlo despliega el volumen de la musica.

    Se dibuja por encima de la pantalla activa y consume sus propios eventos,
    asi los clics del panel no disparan botones del juego. Tambien responde a
    ``V`` (abrir/cerrar), ``M`` (silencio) y ``+``/``-`` (subir/bajar).
    """

    BUTTON = 40

    def __init__(self, app: "GestureRushApp") -> None:
        self.app = app
        self.open = False
        self.dragging = False
        self.hover = False

        self.rect = pygame.Rect(
            WIDTH - MARGIN - self.BUTTON, 22, self.BUTTON, self.BUTTON
        )
        self.panel = pygame.Rect(
            WIDTH - MARGIN - 320, self.rect.bottom + 10, 320, 140
        )
        self.mute_rect = pygame.Rect(
            self.panel.right - 118, self.panel.y + 12, 94, 30
        )
        self.slider = pygame.Rect(
            self.panel.x + 24, self.panel.y + 72, self.panel.w - 48, 10
        )
        # Fondo oscurecido detras del panel: deja claro que el resto no responde.
        self._backdrop = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        self._backdrop.fill((4, 7, 18, 150))

    @property
    def music(self) -> Music:
        return self.app.music

    # -- Estado --------------------------------------------------------------
    def toggle(self) -> None:
        self.open = not self.open
        self.dragging = False

    def _slide_to(self, x: int) -> None:
        fraction = (x - self.slider.x) / max(1, self.slider.w)
        self.music.set_volume(fraction)

    # -- Eventos -------------------------------------------------------------
    def handle_event(self, event: pygame.event.Event) -> bool:
        """Devuelve ``True`` si el evento era del control y no de la pantalla."""
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_v:
                self.toggle()
                return True
            if event.key == pygame.K_m:
                self.music.toggle_mute()
                self.open = True
                return True
            if event.key in (pygame.K_PLUS, pygame.K_KP_PLUS, pygame.K_EQUALS):
                self.music.change_volume(0.05)
                self.open = True
                return True
            if event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                self.music.change_volume(-0.05)
                self.open = True
                return True
            if event.key == pygame.K_ESCAPE and self.open:
                self.open = False
                return True
            return False

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.toggle()
                return True
            if not self.open:
                return False
            if self.slider.inflate(0, 24).collidepoint(event.pos):
                self.dragging = True
                self._slide_to(event.pos[0])
                return True
            if self.mute_rect.collidepoint(event.pos):
                self.music.toggle_mute()
                return True
            if self.panel.collidepoint(event.pos):
                return True
            # Clic fuera: se cierra el panel y el clic sigue a la pantalla.
            self.open = False
            return False

        if event.type == pygame.MOUSEBUTTONUP and event.button == 1 and self.dragging:
            self.dragging = False
            return True

        if event.type == pygame.MOUSEMOTION and self.dragging:
            self._slide_to(event.pos[0])
            return True

        return False

    def update(self) -> None:
        self.hover = self.rect.collidepoint(pygame.mouse.get_pos())

    # -- Dibujo --------------------------------------------------------------
    def _draw_speaker(
        self,
        surface: pygame.Surface,
        center: tuple[int, int],
        color: tuple[int, int, int],
    ) -> None:
        x, y = center
        pygame.draw.polygon(
            surface, color,
            (
                (x - 12, y - 5), (x - 6, y - 5), (x + 2, y - 12),
                (x + 2, y + 12), (x - 6, y + 5), (x - 12, y + 5),
            ),
        )
        if self.music.muted:
            pygame.draw.line(surface, BAD, (x - 4, y - 11), (x + 14, y + 11), 3)
            return
        for radius in (9, 15):
            pygame.draw.arc(
                surface, color,
                pygame.Rect(x + 3, y - radius, radius * 2, radius * 2),
                -0.85, 0.85, 2,
            )

    def _draw_panel(self, surface: pygame.Surface) -> None:
        fonts = self.app.fonts
        music = self.music
        panel = self.panel

        surface.blit(self._backdrop, (0, 0))
        pygame.draw.rect(
            surface, (6, 9, 22), panel.move(0, 4), border_radius=16
        )
        draw_card(surface, panel, (26, 33, 62), radius=16, border=(58, 70, 116))
        draw_text(
            surface, fonts, "Música", 18, TEXT,
            (panel.x + 22, panel.y + 16), bold=True,
        )

        muted = music.muted
        pygame.draw.rect(
            surface, AMBER if muted else (38, 48, 86),
            self.mute_rect, border_radius=15
        )
        draw_text(
            surface, fonts, "Sonar" if muted else "Silencio", 13,
            (26, 22, 8) if muted else (196, 206, 230),
            self.mute_rect.center, bold=muted, anchor="center",
        )

        track = self.slider
        fraction = music.volume
        pygame.draw.rect(surface, (40, 50, 84), track, border_radius=5)
        color = DIM if muted else AMBER
        if fraction > 0:
            filled = pygame.Rect(
                track.x, track.y, max(track.h, int(track.w * fraction)), track.h
            )
            pygame.draw.rect(surface, color, filled, border_radius=5)
        knob = (track.x + int(track.w * fraction), track.centery)
        pygame.draw.circle(surface, (240, 244, 255), knob, 9)
        pygame.draw.circle(surface, color, knob, 9, 2)

        draw_text(
            surface, fonts, "Silenciado" if muted else f"{int(round(fraction * 100))} %",
            15, MUTED if muted else AMBER,
            (panel.x + 22, panel.y + 92), bold=True,
        )
        if music.available:
            draw_text(
                surface, fonts, "V cierra · M silencia · + / −", 12, DIM,
                (panel.right - 22, panel.y + 96), anchor="topright",
            )
        else:
            draw_text(
                surface, fonts, "sin dispositivo de audio", 12, BAD,
                (panel.right - 22, panel.y + 96), anchor="topright",
            )

    def draw(self, surface: pygame.Surface) -> None:
        if self.open:
            self._draw_panel(surface)

        bg = (36, 46, 86) if self.hover else CARD_SOFT
        if self.open:
            bg = AMBER
        fg = (26, 22, 8) if self.open else TEXT
        pygame.draw.rect(surface, bg, self.rect, border_radius=12)
        pygame.draw.rect(
            surface, AMBER_DEEP if self.open else BORDER,
            self.rect, width=2, border_radius=12,
        )
        self._draw_speaker(surface, self.rect.center, fg)


# ---------------------------------------------------------------------------
# Aplicacion
# ---------------------------------------------------------------------------
class GestureRushApp:
    """Ventana, bucle principal y navegacion entre pantallas."""

    def __init__(
        self,
        duel: Duel,
        chain: ChainRace,
        practice: PracticeSession,
        pipeline,
        *,
        demo: bool = False,
        fullscreen: bool = True,
    ) -> None:
        pygame.init()
        pygame.display.set_caption("Gesture Rush")
        self.fullscreen = bool(fullscreen)
        self._apply_mode()
        self.clock = pygame.time.Clock()
        self.fonts = Fonts()
        self.music = Music()
        self.duel = duel
        self.chain = chain
        self.practice = practice
        self.pipeline = pipeline
        self.demo = demo
        self.running = True
        self.players = duel.num_players
        self.rounds = duel.total_rounds

        self.screens: dict[str, Screen] = {
            "menu": MenuScreen(self),
            "duel": DuelScreen(self),
            "chain": ChainScreen(self),
            "practice": PracticeScreen(self),
        }
        self.volume = VolumeControl(self)
        self.active: Screen = self.screens["menu"]

    # -- Ventana -------------------------------------------------------------
    def _apply_mode(self) -> None:
        flags = pygame.SCALED
        flags |= pygame.FULLSCREEN if self.fullscreen else pygame.RESIZABLE
        try:
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT), flags)
            return
        except pygame.error:
            pass
        if self.fullscreen:
            # Si el sistema no puede con pantalla completa, sigue en ventana.
            self.fullscreen = False
            try:
                self.screen = pygame.display.set_mode(
                    (WIDTH, HEIGHT), pygame.SCALED | pygame.RESIZABLE
                )
            except pygame.error:
                pass  # se queda con la superficie anterior

    def toggle_fullscreen(self) -> None:
        self.fullscreen = not self.fullscreen
        self._apply_mode()

    # -- Navegacion ----------------------------------------------------------
    def show_menu(self) -> None:
        self.active = self.screens["menu"]

    def set_players(self, count: int) -> None:
        """Cambia entre 1 y 2 jugadores y rearma Duelo y Cadena.

        Los motores y las pantallas guardan el numero de franjas, asi que se
        reconstruyen con la misma configuracion (rondas, largo de cadena y
        vocabulario) y el menu sigue activo.
        """
        count = 1 if int(count) <= 1 else 2
        if count == self.players:
            return
        self.players = count
        previous = (self.screens["duel"], self.screens["chain"])
        self.duel = Duel(
            num_players=count,
            total_rounds=self.rounds,
            gestures=self.duel.gestures,
        )
        self.chain = ChainRace(
            num_players=count,
            chain_length=self.chain.chain_length,
            gestures=self.chain.gestures,
        )
        self.screens["duel"] = DuelScreen(self)
        self.screens["chain"] = ChainScreen(self)
        if self.active in previous:
            self.active = self.screens["menu"]

    def set_rounds(self, rounds: int) -> None:
        """Cambia las rondas del duelo y rearma su motor y su pantalla."""
        rounds = max(0, int(rounds))
        if rounds == self.rounds:
            return
        self.rounds = rounds
        previous = self.screens["duel"]
        self.duel = Duel(
            num_players=self.players,
            total_rounds=rounds,
            gestures=self.duel.gestures,
        )
        self.screens["duel"] = DuelScreen(self)
        if self.active is previous:
            self.active = self.screens["menu"]

    def start_game(self, key: str) -> None:
        """Entra al minijuego ``key`` con datos limpios."""
        games = {
            "duel": self.duel,
            "chain": self.chain,
            "practice": self.practice,
        }
        game = games[key]
        bind = getattr(self.pipeline, "bind", None)
        if callable(bind):
            bind(game)
        reset = getattr(self.pipeline, "reset", None)
        if callable(reset):
            reset()
        self.active = self.screens[key]

    # -- Bucle ---------------------------------------------------------------
    def run(self) -> None:
        try:
            while self.running:
                now = time.perf_counter()
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                    elif event.type == pygame.KEYDOWN and event.key == pygame.K_F11:
                        self.toggle_fullscreen()
                    elif self.volume.handle_event(event):
                        continue
                    else:
                        self.active.handle_event(event)
                self.active.update(now)
                self.volume.update()
                self.active.draw()
                self.volume.draw(self.screen)
                pygame.display.flip()
                self.clock.tick(FPS)
        finally:
            self.music.close()
            close = getattr(self.pipeline, "close", None)
            if callable(close):
                close()
            pygame.quit()
