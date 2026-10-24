"""Interfaz grafica del minijuego "Duelo de gestos" (Pygame).

Pantallas
---------
- ``MenuScreen``: eleccion de minijuego. Hoy solo "Duelo de gestos" esta
  habilitado; los demas aparecen como proximamente.
- ``DuelScreen``: partida clasica con una sola camara. Cada jugador ocupa una
  mitad del encuadre y compite por hacer el gesto objetivo primero.

El dibujo se hace sobre una superficie de 1240x900 que pygame escala a la
ventana real (``pygame.SCALED``), asi que las coordenadas son siempre las
mismas y la interfaz se adapta al monitor.
"""

from __future__ import annotations

import time

import cv2
import numpy as np
import pygame

from .game import (
    PHASE_COUNTDOWN,
    PHASE_FINISHED,
    PHASE_IDLE,
    PHASE_PLAYING,
    PHASE_RESULT,
    ChainRace,
    Duel,
)

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


def hand_glyph(gesture: str | None, size: int) -> pygame.Surface:
    """Icono vectorial de la mano para el gesto pedido."""
    gesture = gesture or "fist"
    key = (gesture, size)
    cached = _GLYPH_CACHE.get(key)
    if cached is not None:
        return cached

    fingers, thumb = HAND_POSES.get(gesture, HAND_POSES["fist"])
    surf = pygame.Surface((120, 120), pygame.SRCALPHA)

    for i, extended in enumerate(fingers):
        height = 46 if extended else 15
        rect = pygame.Rect(37 + i * 13, 62 - height, 12, height)
        pygame.draw.rect(surf, AMBER, rect, border_radius=6)

    # El pulgar se dibuja antes que la palma para que quede "pegado" a ella.
    thumb_rect = (
        pygame.Rect(24, 40, 14, 40) if thumb else pygame.Rect(27, 68, 14, 24)
    )
    pygame.draw.rect(surf, AMBER, thumb_rect, border_radius=7)

    pygame.draw.rect(surf, AMBER, pygame.Rect(36, 58, 52, 48), border_radius=15)
    pygame.draw.rect(surf, AMBER, pygame.Rect(42, 98, 40, 14), border_radius=7)

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


def format_ms(value: int | None) -> str:
    return "—" if value is None else f"{value} ms"


def points_label(points: int) -> str:
    return f"{points} pt" if points == 1 else f"{points} pts"


def draw_arena_header(
    surface: pygame.Surface,
    fonts: Fonts,
    title: str,
    num_players: int,
) -> None:
    """Cabecera comun a las pantallas de juego."""
    draw_text(surface, fonts, title, 30, TEXT, (MARGIN, 26), bold=True)
    draw_text(
        surface, fonts, "ESC · menú", 15, MUTED,
        (WIDTH - MARGIN - 214, 34), anchor="topright",
    )
    draw_pill(
        surface, fonts,
        f"1 cámara · {num_players} personas",
        (WIDTH - MARGIN, 22), CARD_SOFT, MUTED, size=16, anchor="topright",
    )


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
    confidence: float,
    border: tuple[int, int, int] | None = None,
) -> pygame.Rect:
    """Fondo del panel, recuadro de camara, avatar y chip de prediccion."""
    color = PLAYER_COLORS[index % len(PLAYER_COLORS)]
    inner = PLAYER_INNER[index % len(PLAYER_INNER)]
    draw_card(surface, rect, CARD, radius=20, border=border or BORDER)

    side = "izquierda" if index == 0 else ("derecha" if index == num_players - 1 else "centro")
    draw_text(
        surface, fonts, f"Cámara única · mitad {side}", 15, MUTED,
        (rect.x + 18, rect.y + 16),
    )

    viewport = pygame.Rect(rect.x + 18, rect.y + 46, rect.w - 36, 190)
    pygame.draw.rect(surface, inner, viewport, border_radius=14)

    if frame is not None:
        image, size = frame_surface(frame, index, num_players, viewport.size)
        masked = image.copy()
        masked.blit(rounded_mask(size, 12), (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
        surface.blit(masked, masked.get_rect(center=viewport.center))
    else:
        draw_avatar(surface, (viewport.centerx, viewport.centery - 6), color, 1.25)
        caption = "sin cámara · modo demo" if demo else "sin señal de cámara"
        draw_text(
            surface, fonts, caption, 14, MUTED,
            (viewport.centerx, viewport.bottom - 12), anchor="midbottom",
        )

    pygame.draw.rect(surface, color, viewport, width=2, border_radius=14)

    if label is not None:
        text = (
            f"Jugador {index + 1} · {short_label(label)} {confidence * 100:.0f}%"
        )
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
            "desc": "El mismo gesto para los dos: el primero en hacerlo se lleva el punto.",
            "tag": "1 cámara · 2 jugadores",
            "glyph": "fist",
            "enabled": True,
        },
        {
            "key": "chain",
            "title": "Cadena de gestos",
            "desc": "La misma secuencia para los dos. El primero en completarla gana la carrera.",
            "tag": "1 cámara · 2 jugadores",
            "glyph": "victory",
            "enabled": True,
        },
        {
            "key": "cup",
            "title": "Torneo",
            "desc": "Llaves eliminatorias de cuatro jugadores por rondas.",
            "tag": "Próximamente",
            "glyph": "thumbs_up",
            "enabled": False,
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

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE):
                self.app.start_game("duel")
            elif event.key == pygame.K_1:
                self.app.start_game("duel")
            elif event.key == pygame.K_2:
                self.app.start_game("chain")
            elif event.key == pygame.K_ESCAPE:
                self.app.running = False
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
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

    def draw(self) -> None:
        surface = self.app.screen
        fonts = self.app.fonts
        surface.fill(BG)

        draw_text(surface, fonts, "Gesture Rush", 46, TEXT, (MARGIN, 68), bold=True)
        draw_text(
            surface, fonts, "Elegí un minijuego", 22, MUTED, (MARGIN, 132)
        )

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
            "ENTER jugar · ESC salir · una sola cámara, dos jugadores por mitades",
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
            "Elegí el minijuego.",
            "Cada jugador se para frente a una mitad del encuadre.",
            "El primero en hacer el gesto objetivo se lleva el punto.",
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
        gap = 16
        panel_w = (WIDTH - 2 * MARGIN - gap) // 2
        self.panels = [
            pygame.Rect(MARGIN + i * (panel_w + gap), 318, panel_w, 348)
            for i in range(self.duel.num_players)
        ]
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
        draw_text(
            surface, fonts, "El primero en hacer el gesto objetivo se lleva el punto.",
            22, MUTED, (cx, hero.y + 122), anchor="midtop",
        )
        draw_text(
            surface, fonts,
            f"{duel.num_players} jugadores · {duel.total_rounds or '∞'} rondas · pulsen iniciar",
            18, DIM, (cx, hero.y + 168), anchor="midtop",
        )

    def _draw_result_text(self, surface: pygame.Surface, cx: int) -> None:
        fonts = self.app.fonts
        duel = self.duel
        hero = self.hero

        if duel.phase == PHASE_FINISHED:
            winner = duel.match_winner
            title = "Empate en la partida" if winner is None else f"Gana el jugador {winner + 1}"
            color = MUTED if winner is None else PLAYER_COLORS[winner]
            score = " · ".join(
                f"{points_label(p.points)}" for p in duel.players
            )
            draw_text(surface, fonts, title, 44, color, (cx, hero.y + 92), bold=True, anchor="midtop")
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

        draw_text(
            surface, fonts, f"Gana el jugador {winner + 1}", 44,
            PLAYER_COLORS[winner], (cx, hero.y + 92), bold=True, anchor="midtop",
        )
        margin = record.margin_ms if record else None
        if margin is not None:
            subtitle = f"Diferencia de {margin} ms. Pulsen iniciar para otra ronda."
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
            confidence=score.live_confidence,
            border=PLAYER_COLORS[index % len(PLAYER_COLORS)] if winner_now else BORDER,
        )

        elapsed = duel.elapsed_ms(index, self.now) if duel.phase == PHASE_PLAYING else score.last_ms
        if duel.phase == PHASE_IDLE:
            elapsed = None
        timer_rect = draw_text(
            surface, fonts, format_ms(elapsed), 46,
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
        if duel.phase == PHASE_PLAYING:
            if duel.times_ms.get(index) is not None:
                return "¡Acertado! esperando al rival"
            if score.misses >= 3:
                return "Gestos detectados que no coinciden"
            return "Buscando el gesto objetivo"
        if duel.phase in (PHASE_RESULT, PHASE_FINISHED):
            return "Ronda cerrada"
        if duel.phase == PHASE_COUNTDOWN:
            return "Preparado"
        return "Acercá la mano a tu mitad"

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
        draw_paragraph(
            surface, fonts,
            "Una cámara: menos hardware y sin sincronizar. Separen a las dos "
            "personas, una en cada mitad del encuadre. Riesgo: si se cruzan o se "
            "acercan, el modelo puede confundir quién es quién.",
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
                    surface, fonts, format_ms(value), 17,
                    TEXT if value is not None else DIM,
                    (int(first_x + player * spacing), y),
                )
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
        gap = 16
        panel_w = (WIDTH - 2 * MARGIN - gap) // 2
        self.panels = [
            pygame.Rect(MARGIN + i * (panel_w + gap), 318, panel_w, 348)
            for i in range(self.race.num_players)
        ]
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
                surface, fonts, "Preparen la cadena", 22, MUTED,
                (cx, hero.bottom - 36), anchor="midbottom",
            )
            return

        if not race.chain:
            draw_text(
                surface, fonts, "Cadena de gestos", 44, TEXT,
                (cx, hero.y + 52), bold=True, anchor="midtop",
            )
            draw_text(
                surface, fonts, "La misma secuencia para los dos: el primero en completarla gana.",
                22, MUTED, (cx, hero.y + 118), anchor="midtop",
            )
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
            draw_text(
                surface, fonts, "El primero en completar la cadena gana", 19, MUTED,
                (cx, hero.y + 112), anchor="midtop",
            )

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
                detail += f" · diferencia de {margin} ms"
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
            confidence=score.live_confidence,
            border=PLAYER_COLORS[index % len(PLAYER_COLORS)] if winner_now else BORDER,
        )

        elapsed = race.elapsed_ms(index, self.now)
        timer_rect = draw_text(
            surface, fonts, format_ms(elapsed), 44,
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
        draw_paragraph(
            surface, fonts,
            "Misma cadena para los dos. Cada acierto avanza una posición y fallar "
            "solo cuesta tiempo. Gana quien completa el último gesto antes; si "
            "llegan los dos, gana el tiempo total más bajo.",
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
                surface, fonts, format_ms(race.elapsed_ms(score.index, self.now)),
                17, TEXT if score.done or race.phase == PHASE_PLAYING else DIM,
                (MARGIN + 560, y),
            )
            draw_text(
                surface, fonts, format_ms(race.average_step_ms(score.index)), 17, MUTED,
                (MARGIN + 820, y),
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
        self.pipeline = pipeline
        self.demo = demo
        self.running = True

        self.screens: dict[str, Screen] = {
            "menu": MenuScreen(self),
            "duel": DuelScreen(self),
            "chain": ChainScreen(self),
        }
        self.active: Screen = self.screens["menu"]

    # -- Navegacion ----------------------------------------------------------
    def show_menu(self) -> None:
        self.active = self.screens["menu"]

    def start_game(self, key: str) -> None:
        """Entra al minijuego ``key`` con datos limpios."""
        game = self.duel if key == "duel" else self.chain
        bind = getattr(self.pipeline, "bind", None)
        if callable(bind):
            bind(game)
        reset = getattr(getattr(self.pipeline, "tracker", None), "reset", None)
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
