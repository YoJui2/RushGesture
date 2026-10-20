"""Captura de webcam con control de FPS y gestion de errores.

No depende de MediaPipe, de modo que pueda probarse y reutilizarse sin
`mediapipe` instalado.
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from . import config


class CameraError(RuntimeError):
    """Error al abrir o leer de la camara."""


class Camera:
    """Envoltorio simple sobre ``cv2.VideoCapture``.

    Ejemplo
    -------
    >>> with Camera() as cam:            # doctest: +SKIP
    ...     ok, frame = cam.read()
    """

    def __init__(
        self,
        index: int = config.CAMERA_INDEX,
        width: int = config.FRAME_WIDTH,
        height: int = config.FRAME_HEIGHT,
        target_fps: int = config.TARGET_FPS,
    ) -> None:
        self.index = index
        self.width = width
        self.height = height
        self.target_fps = target_fps
        self._cap: cv2.VideoCapture | None = None
        self._last_times: list[float] = []

    # -- Ciclo de vida -------------------------------------------------------
    def open(self) -> "Camera":
        if self._cap is not None and self._cap.isOpened():
            return self
        for backend in _candidate_backends():
            try:
                cap = cv2.VideoCapture(self.index, backend)
            except cv2.error:
                # Algunos drivers lanzan una excepcion C++ en lugar de
                # devolver un VideoCapture vacio.
                continue
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                cap.set(cv2.CAP_PROP_FPS, self.target_fps)
                self._cap = cap
                return self
            cap.release()
        raise CameraError(
            f"No se pudo abrir la camara en el indice {self.index}. "
            "Verifica que este conectada y no en uso por otra aplicacion. "
            "Para elegir otra entrada: "
            "set GESTURE_CAMERA_INDEX=<indice> (o usa scripts/list_cameras.py)."
        )

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "Camera":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.release()

    # -- Lectura -------------------------------------------------------------
    @property
    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def read(self) -> tuple[bool, np.ndarray | None]:
        """Lee un frame. Devuelve ``(ok, frame_bgr)``.

        Si falla la lectura, intenta reconectar una vez antes de devolver
        ``(False, None)``.
        """
        if self._cap is None:
            self.open()
        assert self._cap is not None
        ok, frame = self._cap.read()
        if not ok or frame is None:
            self._reconnect()
            if self._cap is not None:
                ok, frame = self._cap.read()
        if ok and frame is not None:
            self._record_time()
        return bool(ok and frame is not None), frame

    def _reconnect(self) -> None:
        self.release()
        try:
            self.open()
        except CameraError:
            self._cap = None

    # -- Rendimiento ---------------------------------------------------------
    def _record_time(self) -> None:
        now = time.perf_counter()
        self._last_times.append(now)
        if len(self._last_times) > 30:
            self._last_times.pop(0)

    @property
    def fps(self) -> float:
        """FPS estimado a partir de los ultimos frames."""
        if len(self._last_times) < 2:
            return 0.0
        span = self._last_times[-1] - self._last_times[0]
        if span <= 0:
            return 0.0
        return (len(self._last_times) - 1) / span


def _is_windows() -> bool:
    import sys

    return sys.platform.startswith("win")


def _candidate_backends() -> list[int]:
    """Backends a probar en orden segun la configuracion.

    En Windows ``auto`` prueba Media Foundation (``CAP_MSMF``), luego
    DirectShow y por ultimo cualquiera. Algunos drivers solo responden a uno
    de los dos backends, y DirectShow puede devolver fotogramas negros a
    resoluciones altas.
    """
    named = {
        "dshow": cv2.CAP_DSHOW,
        "msmf": cv2.CAP_MSMF,
        "any": cv2.CAP_ANY,
    }
    requested = getattr(config, "CAMERA_BACKEND", "auto")
    if requested in named:
        return [named[requested]]
    if not _is_windows():
        return [cv2.CAP_ANY]
    return [cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY]
