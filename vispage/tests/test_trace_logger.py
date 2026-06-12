import json

from experiments.run_system import apply_run_tag
from visual_memory_system.config import (
    DatasetConfig,
    LocalityConfig,
    PageConfig,
    RendererConfig,
    RetrievalConfig,
    RunConfig,
    RuntimeConfig,
)
from visual_memory_system.experiments.factory import _effective_page_max_units
from visual_memory_system.runner.system_runner import RunResult
from visual_memory_system.runner.trace_logger import (
    TraceLogger,
    append_run_index,
    build_run_manifest,
    build_run_output_dir,
)
from visual_memory_system.schema import RuntimeTraceRow


def test_trace_logger_writes_required_files(tmp_path) -> None:
    config = RunConfig(
        run_name="test",
        dataset=DatasetConfig(name="d", memory_path="m.jsonl", query_path="q.jsonl"),
        retrieval=RetrievalConfig(mode="provided_access_units", topk=2),
        locality=LocalityConfig(mode="random_anchor", max_units=2),
        renderer=RendererConfig(output_dir="images"),
        runtime=RuntimeConfig(mode="vllm", base_url="http://127.0.0.1:8000", model="model"),
    )
    result = RunResult(
        rows=[
            RuntimeTraceRow(
                query_id="q1",
                query_index=1,
                mode="baseline_cold",
                retrieved_unit_count=2,
                selected_page_key=None,
                coverage=0.0,
                residual_count=2,
                submitted_page_keys=("p1",),
                engine_ttft_ms=100.0,
                client_ttft_ms=110.0,
                prompt_tokens=200,
            )
        ],
        registered_pages={},
    )

    TraceLogger(tmp_path).write_run(
        config=config,
        result=result,
        manifest={"run_id": "test-run"},
    )
    assert (tmp_path / "run_config.json").exists()
    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "trace.jsonl").exists()
    assert (tmp_path / "pages.jsonl").exists()
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["queries"] == 1
    assert summary["baseline_cold_queries"] == 1


def test_page_amplification_bounds_effective_max_units() -> None:
    config = RunConfig(
        run_name="test",
        dataset=DatasetConfig(name="d", memory_path="m.jsonl", query_path="q.jsonl"),
        retrieval=RetrievalConfig(mode="provided_access_units", topk=10),
        locality=LocalityConfig(mode="random_anchor", max_units=200),
        page=PageConfig(max_units=120, max_amplification=8.0),
        renderer=RendererConfig(output_dir="images"),
        runtime=RuntimeConfig(mode="vllm", base_url="http://127.0.0.1:8000", model="model"),
    )

    assert _effective_page_max_units(config) == 80


def test_run_tag_updates_renderer_and_page_key_prefix() -> None:
    config = RunConfig(
        run_name="exp1",
        dataset=DatasetConfig(name="d", memory_path="m.jsonl", query_path="q.jsonl"),
        retrieval=RetrievalConfig(mode="provided_access_units", topk=2),
        locality=LocalityConfig(mode="random_anchor", max_units=2),
        page=PageConfig(key_prefix="exp1"),
        renderer=RendererConfig(output_dir="images"),
        runtime=RuntimeConfig(mode="vllm", base_url="http://127.0.0.1:8000", model="model"),
    )

    tagged = apply_run_tag(config, "20260607_153012")

    assert tagged.renderer.run_tag == "exp1__20260607_153012"
    assert tagged.page.key_prefix == "exp1:exp1__20260607_153012"


def test_run_artifact_layout_records_key_parameters(tmp_path) -> None:
    config = RunConfig(
        run_name="exp1",
        dataset=DatasetConfig(
            name="standard_jsonl_dataset",
            memory_path="dataset/locomo/processed/memory_units.jsonl",
            query_path="dataset/locomo/processed/query_records.jsonl",
        ),
        retrieval=RetrievalConfig(
            mode="embedding_retrieve",
            topk=10,
            embedding_cache_path="dataset/locomo/processed/embeddings/qwen3-embed-4b.json",
        ),
        locality=LocalityConfig(mode="embedding_ball", radius_percentile=5.0, max_units=100),
        page=PageConfig(key_prefix="exp1", max_units=100, max_amplification=10.0),
        renderer=RendererConfig(output_dir="images"),
        runtime=RuntimeConfig(
            mode="vllm",
            base_url="http://127.0.0.1:8000",
            model="qwen3-8b-vl",
            foreground_max_tokens=64,
            background_chunk_tokens=1024,
        ),
    )

    output_dir = build_run_output_dir(tmp_path, config, "20260607_153012")

    assert output_dir == (
        tmp_path
        / "exp1"
        / "locomo"
        / "embed_qwen3-embed-4b_topk10"
        / "ball_p5_max100"
        / "page100_amp10_fgcov0.5_precov0.5_overlapnone_inflightnone_wscale1"
        / "qwen3-8b-vl_fg64_bgchunk1024"
        / "20260607_153012"
    )


def test_append_run_index_writes_queryable_row(tmp_path) -> None:
    config = RunConfig(
        run_name="exp1",
        dataset=DatasetConfig(name="locomo", memory_path="m.jsonl", query_path="q.jsonl"),
        retrieval=RetrievalConfig(mode="provided_access_units", topk=2),
        locality=LocalityConfig(mode="random_anchor", max_units=2),
        renderer=RendererConfig(output_dir="images"),
        runtime=RuntimeConfig(mode="vllm", base_url="http://127.0.0.1:8000", model="model"),
    )
    output_dir = build_run_output_dir(tmp_path, config, "20260607_153012")
    manifest = build_run_manifest(
        config=config,
        config_path="configs/exp1.json",
        output_root=tmp_path,
        output_dir=output_dir,
        timestamp="20260607_153012",
        query_limit=20,
        git_commit="abc123",
    )

    append_run_index(tmp_path, manifest=manifest, summary={"queries": 20})

    rows = (tmp_path / "index.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1
    row = json.loads(rows[0])
    assert row["run_id"] == manifest["run_id"]
    assert row["query_limit"] == 20
    assert row["segments"]["dataset"] == "locomo"
    assert row["summary"]["queries"] == 20
    assert row["artifacts"]["trace"].endswith("trace.jsonl")
