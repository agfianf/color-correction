"""Configuration constants for color checker card grid layout.

This module defines the standard layout for the X-Rite ColorChecker Classic
24-patch card, which has a 6x4 grid (6 columns, 4 rows = 24 patches total).
"""

from typing import Final

# ============================================================================
# Grid Dimensions
# ============================================================================

GRID_ROWS: Final[int] = 4
"""Number of rows in the color checker grid."""

GRID_COLS: Final[int] = 6
"""Number of columns in the color checker grid."""

TOTAL_PATCHES: Final[int] = GRID_ROWS * GRID_COLS  # 24
"""Total number of patches in the color checker card."""


# ============================================================================
# Grid Position Indices
# ============================================================================

ROW_END_INDICES: Final[frozenset[int]] = frozenset([5, 11, 17, 23])
"""Indices of patches at the end of each row (rightmost column)."""

ROW_START_INDICES: Final[frozenset[int]] = frozenset([0, 6, 12, 18])
"""Indices of patches at the start of each row (leftmost column)."""

COL_END_INDICES: Final[frozenset[int]] = frozenset(range(18, 24))
"""Indices of patches in the last row (bottom row)."""

COL_START_INDICES: Final[frozenset[int]] = frozenset(range(0, 6))
"""Indices of patches in the first row (top row)."""


# ============================================================================
# Neighbor Offsets
# ============================================================================

NEIGHBOR_RIGHT_OFFSET: Final[int] = 1
"""Index offset to get right neighbor patch."""

NEIGHBOR_LEFT_OFFSET: Final[int] = -1
"""Index offset to get left neighbor patch."""

NEIGHBOR_BOTTOM_OFFSET: Final[int] = GRID_COLS  # 6
"""Index offset to get bottom neighbor patch (next row)."""

NEIGHBOR_TOP_OFFSET: Final[int] = -GRID_COLS  # -6
"""Index offset to get top neighbor patch (previous row)."""


# ============================================================================
# Visualization Defaults
# ============================================================================

DEFAULT_GRID_FIGSIZE_WIDTH: Final[int] = 15
"""Default figure width for grid visualizations."""

DEFAULT_GRID_FIGSIZE_HEIGHT_PER_ROW: Final[int] = 4
"""Default figure height per row for grid visualizations."""


# ============================================================================
# Detection Defaults
# ============================================================================

MIN_PATCHES_REQUIRED: Final[int] = 24
"""Minimum number of patches required for valid detection."""

DEFAULT_CONFIDENCE_THRESHOLD: Final[float] = 0.25
"""Default confidence threshold for card detection."""

DEFAULT_IOU_THRESHOLD: Final[float] = 0.7
"""Default Intersection over Union threshold for NMS."""


# ============================================================================
# Helper Functions
# ============================================================================


def is_row_end(index: int) -> bool:
    """Check if patch index is at row end (rightmost column).

    Parameters
    ----------
    index : int
        Patch index (0-23)

    Returns
    -------
    bool
        True if patch is at row end
    """
    return index in ROW_END_INDICES


def is_row_start(index: int) -> bool:
    """Check if patch index is at row start (leftmost column).

    Parameters
    ----------
    index : int
        Patch index (0-23)

    Returns
    -------
    bool
        True if patch is at row start
    """
    return index in ROW_START_INDICES


def get_row_number(index: int) -> int:
    """Get row number (0-3) for a given patch index.

    Parameters
    ----------
    index : int
        Patch index (0-23)

    Returns
    -------
    int
        Row number (0 for top row, 3 for bottom row)
    """
    return index // GRID_COLS


def get_col_number(index: int) -> int:
    """Get column number (0-5) for a given patch index.

    Parameters
    ----------
    index : int
        Patch index (0-23)

    Returns
    -------
    int
        Column number (0 for leftmost, 5 for rightmost)
    """
    return index % GRID_COLS
