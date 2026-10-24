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

El dibujo se hace sobre una superficie de 1240x900 que pygame escala a la
ventana real (``pygame.SCALED``), asi que las coordenadas son siempre las
mismas y la interfaz se adapta al monitor.
"""

from __future__ import annotations

import time

import cv2
import numpy as np
import pygame

from . import config
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
WIDTH = 1240
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


def _capsule(
    surface: pygame.Surface,
    color: tuple[int, int, int],
    start: tuple[int, int],
    end: tuple[int, int],
    width: int,
) -> None:
    """Segmento con extremos redondeados (dedo o falange)."""
    radius = max(1, int(width // 2))
    pygame.draw.line(surface, color, start, end, width)
    pygame.draw.circle(surface, color, start, radius)
    pygame.draw.circle(surface, color, end, radius)


def _finger_glyph(
    surface: pygame.Surface,
    x: int,
    width: int,
    length: int,
    splay: int,
    color: tuple[int, int, int] = AMBER,
) -> None:
    """Dedo extendido: dos falanges con una curva suave hacia la punta."""
    base = (x, 58)
    knuckle = (x + int(round(splay * 0.45)), 58 - int(round(length * 0.52)))
    tip = (x + splay, 58 - length)
    _capsule(surface, color, base, knuckle, width)
    _capsule(surface, color, knuckle, tip, max(7, width - 2))


def _knuckle_glyph(
    surface: pygame.Surface,
    x: int,
    width: int,
    color: tuple[int, int, int] = AMBER,
) -> None:
    """Dedo recogido: apenas asoma sobre la palma."""
    _capsule(surface, color, (x, 60), (x, 47), width)


def _palm_glyph(surface: pygame.Surface, color: tuple[int, int, int] = AMBER) -> None:
    pygame.draw.rect(surface, color, pygame.Rect(*PALM_RECT), border_radius=15)
    pygame.draw.rect(surface, color, pygame.Rect(*WRIST_RECT), border_radius=9)


def _thumb_out_glyph(surface: pygame.Surface, *, horizontal: bool = False) -> None:
    if horizontal:
        # Forma de "L": el pulgar sale de costado.
        _capsule(surface, AMBER, (49, 80), (29, 73), 14)
        return
    _capsule(surface, AMBER, (50, 84), (41, 68), 15)
    _capsule(surface, AMBER, (41, 68), (36, 51), 13)
    pygame.draw.circle(surface, AMBER, (50, 84), 8)


def _thumb_up_glyph(surface: pygame.Surface) -> None:
    # Sale por fuera de la palma para que se lea el pulgar por encima del puño.
    _capsule(surface, AMBER, (50, 86), (40, 64), 15)
    _capsule(surface, AMBER, (40, 64), (36, 38), 13)
    pygame.draw.circle(surface, AMBER, (50, 86), 8)


def _thumb_across_glyph(surface: pygame.Surface) -> None:
    """Pulgar plegado sobre la palma; tono mas oscuro para separarlo."""
    _capsule(surface, AMBER_DEEP, (53, 86), (71, 75), 13)


def _ok_glyph(surface: pygame.Surface) -> None:
    """OK: el indice y el pulgar forman un anillo, el resto estirado."""
    pygame.draw.circle(surface, AMBER, (54, 41), 11, 7)
    _capsule(surface, AMBER, (50, 84), (46, 62), 14)
    _capsule(surface, AMBER, (46, 62), (53, 51), 12)
    for x, width, length, splay in FINGER_SPECS[1:]:
        _finger_glyph(surface, x, width, length, splay)
    _palm_glyph(surface)


def hand_glyph(gesture: str | None, size: int) -> pygame.Surface:
    """Icono vectorial de la mano para el gesto pedido."""
    gesture = gesture or "fist"
    key = (gesture, size)
    cached = _GLYPH_CACHE.get(key)
    if cached is not None:
        return cached

    surf = pygame.Surface((120, 120), pygame.SRCALPHA)

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
    """Recorta la franja del jugador y la ajusta al recuadro sin deformar."""
    height, width = frame.shape[:2]
    part = max(1, width // num_players)
    crop = frame[:, player * part:(player + 1) * part]
    ch, cw = crop.shape[:2]
    scale = min(size[0] / cw, size[1] / ch)
    target = (max(1, int(cw * scale)), max(1, int(ch * scale)))
    resized = cv2.resize(crop, target, interpolation=cv2.INTER_AREA)
    rgb = np.ascontiguousarray(resized[:, :, ::-1])
    return pygame.image.frombuffer(rgb.tobytes(), target, "RGB"), target


def format_seconds(value: int | None) -> str:
    """Tiempo en milisegundos -> texto en segundos (``1,23 s``)."""
    if value is None:
        return "—"
    return f"{value / 1000.0:.2f}".replace(".", ",") + " s"


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
        (WIDTH - MARGIN - 214, 34), anchor="topright",
    )
    people = "1 persona" if num_players == 1 else f"{num_players} personas"
    draw_pill(
        surface, fonts,
        chip or f"1 cámara · {people}",
        (WIDTH - MARGIN, 22), CARD_SOFT, MUTED, size=16, anchor="topright",
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


def draw_camera_viewport(
    surface: pygame.Surface,
    fonts: Fonts,
    viewport: pygame.Rect,
    *,
    index: int,
    num_players: int,
    frame: np.ndarray | None,
    demo: bool,
    color: tuple[int, int, int],
    avatar_scale: float = 1.25,
    landmarks: np.ndarray | None = None,
) -> None:
    """Recuadro con la imagen del jugador (o el avatar si no hay camara)."""
    inner = PLAYER_INNER[index % len(PLAYER_INNER)]
    pygame.draw.rect(surface, inner, viewport, border_radius=14)

    if frame is not None:
        image, size = frame_surface(frame, index, num_players, viewport.size)
        masked = image.copy()
        masked.blit(rounded_mask(size, 12), (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
        surface.blit(masked, masked.get_rect(center=viewport.center))
        if landmarks is not None:
            draw_hand_landmarks(
                surface, viewport, size, landmarks,
                index=index, num_players=num_players,
            )
    else:
        draw_avatar(surface, (viewport.centerx, viewport.centery - 6), color, avatar_scale)
        caption = "sin cámara · modo demo" if demo else "sin señal de cámara"
        draw_text(
            surface, fonts, caption, 14, MUTED,
            (viewport.centerx, viewport.bottom - 12), anchor="midbottom",
        )

    pygame.draw.rect(surface, color, viewport, width=2, border_radius=14)


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
) -> pygame.Rect:
    """Fondo del panel, recuadro de camara, avatar y chip de prediccion."""
    color = PLAYER_COLORS[index % len(PLAYER_COLORS)]
    draw_card(surface, rect, CARD, radius=20, border=border or BORDER)

    side = "izquierda" if index == 0 else "derecha"
    zone = "encuadre completo" if num_players == 1 else f"mitad {side}"
    draw_text(
        surface, fonts, f"Cámara única · {zone}", 15, MUTED,
        (rect.x + 18, rect.y + 16),
    )

    viewport = pygame.Rect(rect.x + 18, rect.y + 46, rect.w - 36, 190)
    draw_camera_viewport(
        surface, fonts, viewport,
        index=index, num_players=num_players,
        frame=frame, demo=demo, color=color,
    )

    if label is not None:
        text = f"Jugador {index + 1} · {short_label(label)}"
        chip_bg, chip_fg = color, (14, 18, 34)
    else:
        text = f"Jugador {index + 1} · sin mano"
        chip_bg, chip_fg = CARD_SOFT, MUTED
    draw_pill(
        surface, fonts, text, (viewport.centerx, viewport.bottom),
        chip_bg, chip_fg, size=15, bold=True, anchor="midtop",
    )
    return viewport


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

        # Selector de jugadores para Duelo y Cadena.
        self.player_rects = [
            pygame.Rect(WIDTH - MARGIN - 128, 70, 128, 40),   # 2 jugadores
            pygame.Rect(WIDTH - MARGIN - 248, 70, 112, 40),   # 1 jugador
        ]
        self.players_label_right = self.player_rects[1].x - 14

    # -- Selector de jugadores -----------------------------------------------
    def _player_rect(self, count: int) -> pygame.Rect:
        return self.player_rects[0] if count == 2 else self.player_rects[1]

    def _choose_players(self, count: int) -> None:
        self.app.set_players(count)

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
                self._choose_players(1 if self.app.players == 2 else 2)
            elif event.key == pygame.K_ESCAPE:
                self.app.running = False
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for count, rect in ((1, self.player_rects[1]), (2, self.player_rects[0])):
                if rect.collidepoint(event.pos):
                    self._choose_players(count)
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

    def _draw_players_selector(self, surface: pygame.Surface) -> None:
        """Elige cuántas personas juegan Duelo y Cadena."""
        fonts = self.app.fonts
        draw_text(
            surface, fonts, "Jugadores", 15, MUTED,
            (self.players_label_right, 82), anchor="topright",
        )
        for count, text in ((1, "1 jugador"), (2, "2 jugadores")):
            rect = self._player_rect(count)
            selected = self.app.players == count
            bg = AMBER if selected else CARD_SOFT
            fg = (26, 22, 8) if selected else MUTED
            pygame.draw.rect(surface, bg, rect, border_radius=20)
            draw_text(
                surface, fonts, text, 15, fg, rect.center,
                bold=selected, anchor="center",
            )

    def draw(self) -> None:
        surface = self.app.screen
        fonts = self.app.fonts
        surface.fill(BG)

        draw_text(surface, fonts, "Gesture Rush", 46, TEXT, (MARGIN, 68), bold=True)
        draw_text(
            surface, fonts, "Elegí un minijuego", 22, MUTED, (MARGIN, 132)
        )
        self._draw_players_selector(surface)

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
            "1, 2, 3 eligen minijuego · P o clic cambia jugadores · ESC salir",
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


class DuelScreen(Screen):
    """Partida del modo clásico."""

    def __init__(self, app: "GestureRushApp") -> None:
        super().__init__(app)
        self.duel: Duel = app.duel
        self.frame: np.ndarray | None = None
        self.readings = []
        self.error: str | None = None

        self.hero = pygame.Rect(MARGIN, 78, WIDTH - 2 * MARGIN, 226)
        self.panels = row_panels(self.duel.num_players, 318, 348, max_width=760)
        self.button_rect = pygame.Rect(MARGIN, 704, 300, 64)
        self.note_rect = pygame.Rect(MARGIN + 316, 682, WIDTH - 2 * MARGIN - 316, 108)
        self.hover_button = False

    # -- Eventos -------------------------------------------------------------
    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.app.show_menu()
            elif event.key in (pygame.K_SPACE, pygame.K_RETURN, pygame.K_KP_ENTER):
                self._start_round()
            elif event.key == pygame.K_r:
                self.duel.reset()
            elif event.key in (pygame.K_1, pygame.K_2, pygame.K_3):
                player = event.key - pygame.K_1
                force = getattr(self.app.pipeline, "force", None)
                if force is not None and player < self.duel.num_players:
                    force(player)
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
        try:
            self.frame, self.readings = self.app.pipeline.read(now)
            self.error = None
        except Exception as exc:  # camara desconectada a mitad de partida
            self.frame, self.readings = None, []
            self.error = str(exc)

        self.duel.tick(now)
        for reading in self.readings:
            score = self.duel.players[reading.player]
            score.live_label = reading.label
            score.live_confidence = reading.confidence
            if reading.stable is not None:
                self.duel.observe(reading.player, reading.stable, reading.confidence, now)

        mouse = pygame.mouse.get_pos()
        self.hover_button = self.button_rect.collidepoint(mouse) and self._can_start()

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
                f"{remaining:0.1f} s".replace(".", ","),
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
            f"{players} · {duel.total_rounds or '∞'} rondas · pulsen iniciar",
            18, DIM, (cx, hero.y + 168), anchor="midtop",
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
                    detail += f" · mejor tiempo {format_seconds(player.best_ms)}"
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
                surface, fonts, "Pulsen iniciar para jugar otra partida",
                18, DIM, (cx, hero.y + 186), anchor="midtop",
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
                surface, fonts, "Se acabó el tiempo. Pulsen iniciar para otra ronda.",
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
                f"Tiempo: {format_seconds(time_ms)}. Pulsen iniciar para otra ronda.",
                20, MUTED, (cx, hero.y + 152), anchor="midtop",
            )
            return

        draw_text(
            surface, fonts, f"Gana el jugador {winner + 1}", 44,
            PLAYER_COLORS[winner], (cx, hero.y + 92), bold=True, anchor="midtop",
        )
        margin = record.margin_ms if record else None
        if margin is not None:
            subtitle = f"Diferencia de {format_seconds(margin)}. Pulsen iniciar para otra ronda."
        else:
            subtitle = "Sin rival en el tiempo. Pulsen iniciar para otra ronda."
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
        draw_player_panel(
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
            elapsed = None
        timer_rect = draw_text(
            surface, fonts, format_seconds(elapsed), 46,
            TEXT if elapsed is not None else DIM,
            (rect.x + 18, rect.y + 254), bold=True,
        )
        if winner_now:
            draw_check(surface, (timer_rect.right + 24, timer_rect.centery), 22)
        draw_text(
            surface, fonts, points_label(score.points), 22, MUTED,
            (rect.right - 18, rect.y + 272), anchor="topright",
        )

        status = self._status_text(index)
        draw_text(surface, fonts, status, 15, DIM, (rect.x + 18, rect.y + 316))

    def _status_text(self, index: int) -> str:
        duel = self.duel
        score = duel.players[index]
        solo = duel.num_players == 1
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
                "Una sola persona frente a la cámara: el modo clásico se juega "
                "contra el reloj. Cada ronda muestra un gesto y registra tu "
                "tiempo; si no lo lográs, la ronda queda desierta."
            )
        else:
            note = (
                "Una cámara: menos hardware y sin sincronizar. Cada persona ocupa "
                "una mitad del encuadre (izquierda o derecha). Riesgo: si se cruzan "
                "o se acercan, el modelo puede confundir quién es quién."
            )
        draw_paragraph(
            surface, fonts, note,
            15, MUTED, (self.note_rect.x + 18, self.note_rect.y + 16),
            self.note_rect.w - 36, line_height=23,
        )
        if self.error:
            draw_text(
                surface, fonts, self.error[:110], 14, BAD,
                (self.note_rect.x + 18, self.note_rect.bottom - 26),
            )

    def _draw_history(self, surface: pygame.Surface) -> None:
        fonts = self.app.fonts
        duel = self.duel
        top = 806
        draw_text(surface, fonts, "Ronda", 15, MUTED, (MARGIN + 10, top))
        draw_text(surface, fonts, "Gesto", 15, MUTED, (MARGIN + 118, top))

        first_x = MARGIN + 360
        last_x = WIDTH - MARGIN - 120
        columns = max(1, duel.num_players)
        spacing = (last_x - first_x) / columns if columns else 0
        for i in range(duel.num_players):
            draw_text(
                surface, fonts, f"Jugador {i + 1}", 15, MUTED,
                (int(first_x + i * spacing), top),
            )
        if duel.num_players > 1:
            draw_text(surface, fonts, "Gana", 15, MUTED, (last_x + 40, top))
        pygame.draw.line(
            surface, (38, 46, 78),
            (MARGIN, top + 24), (WIDTH - MARGIN, top + 24), 1,
        )

        if not duel.history:
            draw_text(
                surface, fonts, "Todavía no hay rondas jugadas.", 16, DIM,
                (MARGIN + 10, top + 38),
            )
            return

        rows = list(reversed(duel.history))[:2]
        for row_index, record in enumerate(rows):
            y = top + 40 + row_index * 26
            draw_text(surface, fonts, str(record.index), 17, TEXT, (MARGIN + 10, y))
            glyph = hand_glyph(record.gesture, 20)
            surface.blit(glyph, (MARGIN + 116, y - 2))
            draw_text(
                surface, fonts, short_label(record.gesture), 17, TEXT, (MARGIN + 142, y)
            )
            for player in range(duel.num_players):
                value = record.times_ms.get(player)
                draw_text(
                    surface, fonts, format_seconds(value), 17,
                    TEXT if value is not None else DIM,
                    (int(first_x + player * spacing), y),
                )
            if duel.num_players > 1:
                if record.winner is None:
                    draw_text(surface, fonts, "—", 17, DIM, (last_x + 40, y))
                else:
                    draw_text(
                        surface, fonts, f"Jugador {record.winner + 1}", 17,
                        PLAYER_COLORS[record.winner % len(PLAYER_COLORS)],
                        (last_x + 40, y), bold=True,
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
        self.panels = row_panels(self.race.num_players, 318, 348, max_width=760)
        self.button_rect = pygame.Rect(MARGIN, 704, 300, 64)
        self.note_rect = pygame.Rect(MARGIN + 316, 682, WIDTH - 2 * MARGIN - 316, 108)
        self.hover_button = False

    # -- Eventos -------------------------------------------------------------
    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.app.show_menu()
            elif event.key in (pygame.K_SPACE, pygame.K_RETURN, pygame.K_KP_ENTER):
                self._start_race()
            elif event.key == pygame.K_r:
                self.race.reset()
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
        try:
            self.frame, self.readings = self.app.pipeline.read(now)
            self.error = None
        except Exception as exc:  # camara desconectada a mitad de carrera
            self.frame, self.readings = None, []
            self.error = str(exc)

        self.race.tick(now)
        for reading in self.readings:
            score = self.race.players[reading.player]
            score.live_label = reading.label
            score.live_confidence = reading.confidence
            if reading.stable is not None:
                self.race.observe(reading.player, reading.stable, reading.confidence, now)

        mouse = pygame.mouse.get_pos()
        self.hover_button = self.button_rect.collidepoint(mouse) and self._can_start()

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
                surface, fonts,
                f"{race.chain_length} gestos en orden · pulsen iniciar para sortear la cadena",
                18, DIM, (cx, hero.y + 164), anchor="midtop",
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
                surface, fonts, "Pulsen iniciar para otra carrera.", 19, MUTED,
                (cx, hero.y + 160), anchor="midtop",
            )
            return

        score = race.players[winner]
        if race.num_players == 1:
            draw_text(
                surface, fonts, "Cadena completada", 40, GOOD,
                (cx, hero.y + 102), bold=True, anchor="midtop",
            )
            detail = (
                f"{race.num_steps} gestos en {format_seconds(score.finished_ms)}"
                f" · media {format_seconds(race.average_step_ms(winner))}"
            )
            draw_text(surface, fonts, detail, 19, MUTED, (cx, hero.y + 160), anchor="midtop")
            return

        draw_text(
            surface, fonts, f"Gana el jugador {winner + 1}", 40,
            PLAYER_COLORS[winner % len(PLAYER_COLORS)],
            (cx, hero.y + 102), bold=True, anchor="midtop",
        )
        detail = f"{race.num_steps} gestos en {format_seconds(score.finished_ms)}"
        rival = next((p for p in race.players if p.index != winner), None)
        if rival is not None:
            if rival.done and rival.finished_ms is not None:
                margin = abs(rival.finished_ms - (score.finished_ms or 0))
                detail += f" · diferencia de {format_seconds(margin)}"
            else:
                detail += f" · el jugador {rival.index + 1} quedó en {rival.progress}/{race.num_steps}"
        draw_text(surface, fonts, detail, 19, MUTED, (cx, hero.y + 160), anchor="midtop")

    def _draw_panel(self, surface: pygame.Surface, index: int, rect: pygame.Rect) -> None:
        fonts = self.app.fonts
        race = self.race
        score = race.players[index]
        winner_now = race.phase == PHASE_FINISHED and race.winner == index

        draw_player_panel(
            surface, fonts, rect,
            index=index,
            num_players=race.num_players,
            frame=self.frame,
            demo=self.app.demo,
            label=score.live_label,
            border=PLAYER_COLORS[index % len(PLAYER_COLORS)] if winner_now else BORDER,
        )

        elapsed = race.elapsed_ms(index, self.now)
        timer_rect = draw_text(
            surface, fonts, format_seconds(elapsed), 44,
            TEXT if elapsed is not None else DIM,
            (rect.x + 18, rect.y + 250), bold=True,
        )
        if winner_now:
            draw_check(surface, (timer_rect.right + 24, timer_rect.centery), 22)
        draw_text(
            surface, fonts, f"{score.progress}/{race.num_steps}", 22, MUTED,
            (rect.right - 18, rect.y + 268), anchor="topright",
        )
        self._draw_progress_dots(surface, rect, score)

    def _draw_progress_dots(self, surface: pygame.Surface, rect: pygame.Rect, score) -> None:
        race = self.race
        total = max(1, race.num_steps)
        y = rect.y + 324
        for i in range(total):
            x = rect.x + 24 + i * 30
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
            self.note_rect.w - 36, line_height=23,
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
        draw_text(surface, fonts, "Jugador", 15, MUTED, (MARGIN + 10, top))
        draw_text(surface, fonts, "Cadena", 15, MUTED, (MARGIN + 300, top))
        draw_text(surface, fonts, "Tiempo total", 15, MUTED, (MARGIN + 560, top))
        draw_text(surface, fonts, "Media por gesto", 15, MUTED, (MARGIN + 820, top))
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
                (MARGIN + 10, y), bold=True,
            )
            draw_text(
                surface, fonts, f"{score.progress}/{race.num_steps}", 17, TEXT,
                (MARGIN + 300, y),
            )
            draw_text(
                surface, fonts, format_seconds(race.elapsed_ms(score.index, self.now)),
                17, TEXT if score.done or race.phase == PHASE_PLAYING else DIM,
                (MARGIN + 560, y),
            )
            draw_text(
                surface, fonts, format_seconds(race.average_step_ms(score.index)), 17, MUTED,
                (MARGIN + 820, y),
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

        self.camera_rect = pygame.Rect(MARGIN, 78, 780, 562)
        self.side_rect = pygame.Rect(820, 78, 396, 562)
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
    ) -> None:
        pygame.init()
        pygame.display.set_caption("Gesture Rush")
        self.screen = pygame.display.set_mode(
            (WIDTH, HEIGHT), pygame.SCALED | pygame.RESIZABLE
        )
        self.clock = pygame.time.Clock()
        self.fonts = Fonts()
        self.duel = duel
        self.chain = chain
        self.practice = practice
        self.pipeline = pipeline
        self.demo = demo
        self.running = True
        self.players = duel.num_players

        self.screens: dict[str, Screen] = {
            "menu": MenuScreen(self),
            "duel": DuelScreen(self),
            "chain": ChainScreen(self),
            "practice": PracticeScreen(self),
        }
        self.active: Screen = self.screens["menu"]

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
            total_rounds=self.duel.total_rounds,
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
                    else:
                        self.active.handle_event(event)
                self.active.update(now)
                self.active.draw()
                pygame.display.flip()
                self.clock.tick(FPS)
        finally:
            close = getattr(self.pipeline, "close", None)
            if callable(close):
                close()
            pygame.quit()
