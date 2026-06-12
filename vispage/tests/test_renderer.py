from visual_memory_system.page.renderer import PillowPageRenderer
from visual_memory_system.schema import MemoryUnit, VisualPage


def test_renderer_stacks_unit_tiles_without_compression(tmp_path) -> None:
    renderer = PillowPageRenderer(
        output_dir=tmp_path,
        unit_width=320,
        font_size=12,
        line_height=16,
        padding=8,
        chars_per_line=20,
        tile_gap=0,
    )
    memory = {
        "u1": MemoryUnit(unit_id="u1", text="short text"),
        "u2": MemoryUnit(unit_id="u2", text="another short text"),
    }
    page1 = renderer.render_page(VisualPage(page_key="p1", unit_ids=("u1",)), memory)
    page2 = renderer.render_page(VisualPage(page_key="p2", unit_ids=("u2",)), memory)
    combined = renderer.render_page(VisualPage(page_key="p12", unit_ids=("u1", "u2")), memory)

    assert combined.metadata["width"] == page1.metadata["width"] == page2.metadata["width"]
    assert combined.metadata["height"] == page1.metadata["height"] + page2.metadata["height"]
    assert combined.metadata["unit_count"] == 2


def test_renderer_adds_run_tag_to_page_start(tmp_path) -> None:
    memory = {"u1": MemoryUnit(unit_id="u1", text="short text")}
    plain = PillowPageRenderer(
        output_dir=tmp_path / "plain",
        unit_width=320,
        font_size=12,
        line_height=16,
        padding=8,
        chars_per_line=20,
    ).render_page(VisualPage(page_key="p1", unit_ids=("u1",)), memory)
    tagged = PillowPageRenderer(
        output_dir=tmp_path / "tagged",
        unit_width=320,
        font_size=12,
        line_height=16,
        padding=8,
        chars_per_line=20,
        run_tag="exp1__20260607_153012",
    ).render_page(VisualPage(page_key="p1", unit_ids=("u1",)), memory)

    assert tagged.metadata["run_tag"] == "exp1__20260607_153012"
    assert tagged.metadata["height"] > plain.metadata["height"]


def test_renderer_width_scale_adds_right_padding_without_relayout(tmp_path) -> None:
    memory = {"u1": MemoryUnit(unit_id="u1", text="short text")}
    base = PillowPageRenderer(
        output_dir=tmp_path / "base",
        unit_width=320,
        width_scale=1.0,
        font_size=12,
        line_height=16,
        padding=8,
        chars_per_line=20,
    ).render_page(VisualPage(page_key="p1", unit_ids=("u1",)), memory)
    wide = PillowPageRenderer(
        output_dir=tmp_path / "wide",
        unit_width=320,
        width_scale=2.0,
        font_size=12,
        line_height=16,
        padding=8,
        chars_per_line=20,
    ).render_page(VisualPage(page_key="p1", unit_ids=("u1",)), memory)

    assert wide.metadata["layout_width"] == 320
    assert wide.metadata["width"] == 640
    assert wide.metadata["height"] == base.metadata["height"]
    assert wide.prompt_tokens_estimate > base.prompt_tokens_estimate
