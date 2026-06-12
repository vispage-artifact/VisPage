"""Actual visual page renderer with unit-tile composition.

Rendering must preserve a stable unit-to-token scale. Each memory unit is first
rendered into one fixed-width tile whose height is determined by the unit text.
A visual page is then composed by vertically stacking those unit tiles. Baseline
pages, residual pages, and reusable memory pages all use this same path.
"""

from __future__ import annotations

import math
import textwrap
from pathlib import Path

from visual_memory_system.schema import MemoryUnit, VisualPage


class PillowPageRenderer:
    def __init__(
        self,
        *,
        output_dir: str | Path,
        unit_width: int = 1024,
        width_scale: float = 1.0,
        post_render_scale: float = 1.0,
        font_size: int = 24,
        line_height: int = 34,
        padding: int = 24,
        chars_per_line: int = 72,
        tile_gap: int = 0,
        run_tag: str | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.unit_width = unit_width
        self.width_scale = width_scale
        self.render_width = math.ceil(unit_width * width_scale)
        self.post_render_scale = post_render_scale
        self.font_size = font_size
        self.line_height = line_height
        self.padding = padding
        self.chars_per_line = chars_per_line
        self.tile_gap = tile_gap
        self.run_tag = run_tag
        if (
            unit_width <= 0
            or width_scale < 1.0
            or post_render_scale <= 0
            or post_render_scale > 1.0
            or font_size <= 0
            or line_height <= 0
            or padding < 0
            or chars_per_line <= 0
            or tile_gap < 0
        ):
            raise ValueError("renderer dimensions must be positive")

    def render_page(self, page: VisualPage, memory_by_id: dict[str, MemoryUnit]) -> VisualPage:
        try:
            from PIL import Image, ImageDraw, ImageFont
        except ImportError as exc:
            raise RuntimeError("Pillow is required for actual visual page rendering") from exc

        missing = [unit_id for unit_id in page.unit_ids if unit_id not in memory_by_id]
        if missing:
            raise ValueError(f"page {page.page_key} contains unknown unit ids: {missing}")

        font = ImageFont.load_default(size=self.font_size)
        tiles = []
        if self.run_tag:
            tiles.append(self._render_run_tag_tile(font))
        tiles.extend(
            self._render_unit_tile(memory_by_id[unit_id], font)
            for unit_id in page.unit_ids
        )
        if not tiles:
            raise ValueError(f"page {page.page_key} has no rendered tiles")
        height = sum(tile.height for tile in tiles) + self.tile_gap * (len(tiles) - 1)
        image = Image.new("RGB", (self.render_width, height), "white")
        y = 0
        for tile in tiles:
            image.paste(tile, (0, y))
            y += tile.height + self.tile_gap
        output_width = self.render_width
        output_height = height
        if self.post_render_scale != 1.0:
            output_width = max(1, math.ceil(self.render_width * self.post_render_scale))
            output_height = max(1, math.ceil(height * self.post_render_scale))
            image = image.resize((output_width, output_height), Image.Resampling.LANCZOS)

        image_path = self.output_dir / f"{_safe_filename(page.page_key)}.png"
        image.save(image_path)
        estimated_tokens = math.ceil((output_width / 32) * (output_height / 32))
        return VisualPage(
            page_key=page.page_key,
            unit_ids=page.unit_ids,
            image_path=str(image_path),
            prompt_tokens_estimate=estimated_tokens,
            metadata={
                **page.metadata,
                "renderer": "pillow_unit_tiles",
                "width": output_width,
                "layout_width": self.unit_width,
                "width_scale": self.width_scale,
                "post_render_scale": self.post_render_scale,
                "height": output_height,
                "layout_height": height,
                "unit_count": len(page.unit_ids),
                "tile_gap": self.tile_gap,
                "run_tag": self.run_tag,
            },
        )

    def estimate_page_tokens(
        self,
        unit_ids: tuple[str, ...] | list[str],
        memory_by_id: dict[str, MemoryUnit],
    ) -> int:
        height = self.estimate_page_height(unit_ids, memory_by_id)
        output_width = max(1, math.ceil(self.render_width * self.post_render_scale))
        output_height = max(1, math.ceil(height * self.post_render_scale))
        return math.ceil((output_width / 32) * (output_height / 32))

    def estimate_page_height(
        self,
        unit_ids: tuple[str, ...] | list[str],
        memory_by_id: dict[str, MemoryUnit],
    ) -> int:
        missing = [unit_id for unit_id in unit_ids if unit_id not in memory_by_id]
        if missing:
            raise ValueError(f"page estimate contains unknown unit ids: {missing}")
        heights = []
        if self.run_tag:
            lines = textwrap.wrap(f"run_tag: {self.run_tag}", width=self.chars_per_line)
            heights.append(self.padding * 2 + len(lines) * self.line_height)
        for unit_id in unit_ids:
            line_count = len(self._unit_lines(memory_by_id[unit_id]))
            heights.append(self.padding * 2 + line_count * self.line_height)
        if not heights:
            raise ValueError("cannot estimate empty page")
        return sum(heights) + self.tile_gap * (len(heights) - 1)

    def _unit_lines(self, unit: MemoryUnit) -> list[str]:
        header = f"[{unit.unit_id}]"
        wrapped = textwrap.wrap(unit.text, width=self.chars_per_line) or [""]
        return [header, *wrapped]

    def _render_run_tag_tile(self, font):
        from PIL import Image, ImageDraw

        lines = textwrap.wrap(f"run_tag: {self.run_tag}", width=self.chars_per_line)
        height = self.padding * 2 + len(lines) * self.line_height
        image = Image.new("RGB", (self.unit_width, height), "white")
        draw = ImageDraw.Draw(image)
        y = self.padding
        for line in lines:
            draw.text((self.padding, y), line, fill="black", font=font)
            y += self.line_height
        return image

    def _render_unit_tile(self, unit: MemoryUnit, font):
        from PIL import Image, ImageDraw

        lines = self._unit_lines(unit)
        height = self.padding * 2 + len(lines) * self.line_height
        image = Image.new("RGB", (self.unit_width, height), "white")
        draw = ImageDraw.Draw(image)
        y = self.padding
        for line in lines:
            draw.text((self.padding, y), line, fill="black", font=font)
            y += self.line_height
        return image


def _safe_filename(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in value)
