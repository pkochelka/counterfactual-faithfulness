from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
import html
import threading
import traceback
import uuid

import pandas as pd
import streamlit as st
import os

from webnlg_utils import (
    DEFAULT_JUDGE_MODEL,
    DEFAULT_JUDGE_MAX_TOKENS,
    DEFAULT_OUTPUT_DIR,
    JudgeRequestError,
    SourceSpec,
    adhoc_output_path,
    annotation_lookup_key,
    build_annotation_summary,
    build_judge_prompt,
    enrich_sentences,
    infer_default_flat_csv_path,
    infer_default_xml_path,
    judge_output_path,
    judge_row,
    latest_annotation_map,
    load_adhoc_annotations,
    load_annotations_for_sources,
    load_entry_table,
    load_entry_table_from_flat_csv,
    load_env_defaults,
    load_output_sources,
    parse_source_specs,
    sanitize_identifier,
    write_judge_records,
)
from runtime_state import BATCH_RUNTIME


load_env_defaults()

st.set_page_config(page_title="LLM Judge Workspace", layout="wide")
st.markdown(
    """
    <style>
    .sentence-box {
        border: 6px solid rgba(49, 51, 63, 0.7);
        border-radius: 0.75rem;
        padding: 0.9rem 1rem;
        margin: 0.35rem 0 0.85rem 0;
        font-size: 1.05rem;
        line-height: 1.6;
    }
    .source-label {
        font-size: 0.9rem;
        font-weight: 500;
        color: rgba(49, 51, 63, 0.75);
        margin-bottom: 0.35rem;
    }
    .meta-card {
        border: 1px solid rgba(49, 51, 63, 0.16);
        border-radius: 0.75rem;
        padding: 0.7rem 0.85rem;
        min-height: 5.5rem;
    }
    .meta-label {
        font-size: 0.78rem;
        font-weight: 600;
        color: rgba(49, 51, 63, 0.7);
        margin-bottom: 0.35rem;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }
    .meta-value {
        font-size: 1rem;
        line-height: 1.35;
        overflow-wrap: anywhere;
        word-break: break-word;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def streamlit_secret(name: str, default: str = "") -> str:
    try:
        return str(st.secrets.get(name, default))
    except FileNotFoundError:
        return default


def parse_int_setting(raw_value: str, *, default: int, min_value: int, max_value: int) -> int:
    try:
        parsed = int(str(raw_value).strip())
    except (TypeError, ValueError):
        return default
    return max(min_value, min(max_value, parsed))


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_output_text() -> str:
    default_csv = repo_root() / "sentences_webnlg_cf_qwen3.5-122b.csv"
    return str(default_csv)


def default_fa_output_text() -> str:
    default_csv = repo_root() / "sentences_webnlg_fa_qwen3.5-122b.csv"
    return str(default_csv)


def dataset_presets() -> dict[str, dict[str, str]]:
    data_root = repo_root() / "data"
    xml_root = data_root / "GEM-v2-D2T-SharedTask"
    return {
        "CF preset": {
            "dataset_path": str(xml_root / "D2T-1-CFA_WebNLG_CounterFactual.xml"),
            "outputs_text": default_output_text(),
        },
        "FA preset": {
            "dataset_path": str(xml_root / "D2T-1-FA_WebNLG_Factual.xml"),
            "outputs_text": default_fa_output_text(),
        },
        "CSV CF preset": {
            "dataset_path": str(data_root / "webnlg_cf.csv"),
            "outputs_text": default_output_text(),
        },
        "CSV FA preset": {
            "dataset_path": str(data_root / "webnlg_fa.csv"),
            "outputs_text": default_fa_output_text(),
        },
        "Custom": {
            "dataset_path": str(infer_default_xml_path()),
            "outputs_text": default_output_text(),
        },
    }


def source_cache_signature(raw_sources: str) -> tuple[tuple[str, float], ...]:
    signature: list[tuple[str, float]] = []
    for spec in parse_source_specs(raw_sources):
        try:
            signature.append((str(spec.csv_path), spec.csv_path.stat().st_mtime))
        except FileNotFoundError:
            signature.append((str(spec.csv_path), -1.0))
    return tuple(signature)


def file_signature(path_text: str) -> tuple[str, float]:
    path = Path(path_text).expanduser()
    try:
        return str(path.resolve()), path.stat().st_mtime
    except FileNotFoundError:
        return str(path), -1.0


def annotation_cache_signature(
    output_dir: str,
    source_specs: tuple[tuple[str, str, str], ...],
    dataset_name: str,
) -> tuple[tuple[str, float], ...]:
    output_root = Path(output_dir).expanduser()
    signature: list[tuple[str, float]] = []

    for _label, csv_path, _source_id in source_specs:
        stem = Path(csv_path).stem
        for path in sorted(output_root.glob(f"judge_{stem}_*.jsonl")):
            try:
                signature.append((str(path.resolve()), path.stat().st_mtime))
            except FileNotFoundError:
                continue

    safe_dataset = sanitize_identifier(dataset_name)
    for path in sorted(output_root.glob(f"judge_adhoc_{safe_dataset}_*.jsonl")):
        try:
            signature.append((str(path.resolve()), path.stat().st_mtime))
        except FileNotFoundError:
            continue

    return tuple(signature)


@st.cache_data(show_spinner=False)
def cached_entry_table(dataset_path: str, _dataset_signature: tuple[str, float]) -> pd.DataFrame:
    suffix = Path(dataset_path).suffix.lower()
    if suffix == ".xml":
        return load_entry_table(dataset_path)
    if suffix == ".csv":
        return load_entry_table_from_flat_csv(dataset_path)
    raise ValueError(f"Unsupported dataset source type for {dataset_path!r}. Use .xml or .csv.")


@st.cache_data(show_spinner=False)
def cached_outputs(
    raw_sources: str,
    _source_signature: tuple[tuple[str, float], ...],
) -> tuple[list[SourceSpec], pd.DataFrame, list[str]]:
    specs = parse_source_specs(raw_sources)
    errors: list[str] = []
    valid_specs: list[SourceSpec] = []
    frames: list[pd.DataFrame] = []

    for spec in specs:
        try:
            if not spec.csv_path.exists():
                raise FileNotFoundError(spec.csv_path)
            frames.append(load_output_sources([spec]))
            valid_specs.append(spec)
        except Exception as exc:
            errors.append(f"{spec.label}: {exc}")

    if not frames:
        return valid_specs, pd.DataFrame(), errors
    return valid_specs, pd.concat(frames, ignore_index=True), errors


@st.cache_data(show_spinner=False)
def cached_enriched_source(
    csv_path: str,
    xml_path: str,
    _csv_signature: tuple[str, float],
    _xml_signature: tuple[str, float],
) -> pd.DataFrame:
    return enrich_sentences(csv_path, xml_path)


@st.cache_data(show_spinner=False)
def cached_annotations(
    output_dir: str,
    source_specs: tuple[tuple[str, str, str], ...],
    _annotation_signature: tuple[tuple[str, float], ...],
) -> pd.DataFrame:
    specs = [SourceSpec(label=label, csv_path=Path(csv_path), source_id=source_id) for label, csv_path, source_id in source_specs]
    return load_annotations_for_sources(output_dir=output_dir, source_specs=specs, judge_model=None)


@st.cache_data(show_spinner=False)
def cached_adhoc_annotations(
    output_dir: str,
    dataset_name: str,
    _annotation_signature: tuple[tuple[str, float], ...],
) -> pd.DataFrame:
    return load_adhoc_annotations(output_dir=output_dir, dataset_name=dataset_name, judge_model=None)


def render_annotation(record: dict[str, Any] | None) -> None:
    if not record:
        st.caption("No annotation saved for this output yet.")
        return

    score = record.get("faithfulness_score")
    if score is None:
        score = record.get("score")
    stamp = record.get("timestamp") or "unknown time"
    incorrect_information = record.get("incorrect_information") or []
    judge_model_name = record.get("judge_model") or "unknown judge model"
    requested_judge_model = record.get("requested_judge_model")
    provider_name = record.get("provider")
    request_cost = record.get("request_cost")

    if score is not None:
        st.metric("Faithfulness score", score)
    else:
        st.metric("Incorrect items", len(incorrect_information))
    st.caption(f"Judge model: {judge_model_name}")
    if provider_name:
        st.caption(f"Provider: {provider_name}")
    if requested_judge_model and requested_judge_model != judge_model_name:
        st.caption(f"Requested alias: {requested_judge_model}")
    if request_cost is not None:
        st.caption(f"Request cost: ${request_cost}")
    st.caption(stamp)

    with st.expander("Incorrect information", expanded=True):
        if incorrect_information:
            for item in incorrect_information:
                if isinstance(item, dict):
                    info_used = item.get("info_used", "")
                    correct_info = item.get("correct_info", "")
                    comment = item.get("comment", "")
                    st.error(f"Used: {info_used}" if info_used else "Used: ")
                    if correct_info:
                        st.write(f"Correct: {correct_info}")
                    if comment:
                        st.caption(comment)
                else:
                    st.error(str(item))
        else:
            st.success("None")


def render_prompt(prompt: str, expanded: bool = False) -> None:
    preview_lines = prompt.splitlines()[:7]
    st.code("\n".join(preview_lines), language="text")
    with st.expander("Full judge prompt", expanded=expanded):
        st.code(prompt, language="text")


def joined_output_rows(entries_df: pd.DataFrame, outputs_df: pd.DataFrame) -> pd.DataFrame:
    if outputs_df.empty:
        return pd.DataFrame()
    return entries_df.merge(outputs_df, how="inner", on="eid")


def annotations_for_selected_model(frame: pd.DataFrame, judge_model: str) -> pd.DataFrame:
    if frame.empty:
        return frame

    requested = frame.get("requested_judge_model")
    actual = frame.get("judge_model")

    if requested is None and actual is None:
        return frame.iloc[0:0].copy()

    requested_values = requested.fillna("").astype(str) if requested is not None else pd.Series("", index=frame.index)
    actual_values = actual.fillna("").astype(str) if actual is not None else pd.Series("", index=frame.index)
    mask = (requested_values == judge_model) | ((requested_values == "") & (actual_values == judge_model))
    return frame[mask].copy()


def source_visibility_controls(source_specs: list[SourceSpec]) -> dict[str, bool]:
    visibility: dict[str, bool] = {}
    for spec in source_specs:
        key = f"show_source_{spec.source_id}"
        default = st.session_state.get(key, True)
        visibility[spec.source_id] = st.checkbox(spec.label, value=default, key=key)
    return visibility


def source_spec_map(source_specs: list[SourceSpec]) -> dict[str, SourceSpec]:
    return {spec.source_id: spec for spec in source_specs}


def clear_active_batch_job(job_id: str) -> None:
    with BATCH_RUNTIME["lock"]:
        BATCH_RUNTIME["jobs"].pop(job_id, None)


def entry_picker(filtered_entries: pd.DataFrame) -> str:
    if filtered_entries.empty:
        raise ValueError("entry_picker requires at least one entry.")

    if "selected_eid" not in st.session_state:
        st.session_state.selected_eid = filtered_entries["eid"].iloc[0]

    valid_eids = filtered_entries["eid"].astype(str).tolist()
    if st.session_state.selected_eid not in valid_eids:
        st.session_state.selected_eid = valid_eids[0]

    selected = st.selectbox(
        "Selected entry",
        options=valid_eids,
        index=valid_eids.index(st.session_state.selected_eid),
    )
    st.session_state.selected_eid = selected
    return selected


def entry_navigation(filtered_entries: pd.DataFrame) -> str:
    valid_eids = filtered_entries["eid"].astype(str).tolist()
    if not valid_eids:
        raise ValueError("entry_navigation requires at least one entry.")

    selected_eid = entry_picker(filtered_entries)
    current_index = valid_eids.index(selected_eid)
    total = len(valid_eids)

    nav_cols = st.columns([1, 1, 1.2], gap="small")
    with nav_cols[0]:
        if st.button("Previous", use_container_width=True, disabled=current_index == 0):
            selected_eid = valid_eids[max(0, current_index - 1)]
            st.session_state.selected_eid = selected_eid
            st.rerun()
    with nav_cols[1]:
        if st.button("Next", use_container_width=True, disabled=current_index >= total - 1):
            selected_eid = valid_eids[min(total - 1, current_index + 1)]
            st.session_state.selected_eid = selected_eid
            st.rerun()
    with nav_cols[2]:
        if st.session_state.selected_eid not in valid_eids:
            st.session_state.selected_eid = valid_eids[0]
        display_index = valid_eids.index(st.session_state.selected_eid)
        st.metric("Position", f"{display_index + 1} / {total}")

    jump_cols = st.columns([1.1, 1.3], gap="small")
    with jump_cols[0]:
        jump_index = st.number_input(
            "Jump to row",
            min_value=1,
            max_value=total,
            value=valid_eids.index(st.session_state.selected_eid) + 1,
            step=1,
            key="jump_to_row",
            help="Enter a 1-based row number within the current filtered set.",
        )
        safe_jump_index = min(total, max(1, int(jump_index)))
        if safe_jump_index != valid_eids.index(st.session_state.selected_eid) + 1:
            selected_eid = valid_eids[safe_jump_index - 1]
            st.session_state.selected_eid = selected_eid
    with jump_cols[1]:
        jump_eid = st.selectbox(
            "Jump to entry id",
            options=valid_eids,
            index=valid_eids.index(st.session_state.selected_eid),
            key="jump_to_eid",
            help="Quickly jump to any entry id in the current filtered set.",
        )
        if jump_eid != selected_eid:
            selected_eid = jump_eid
            st.session_state.selected_eid = selected_eid

    return st.session_state.selected_eid


def run_single_judge(
    *,
    row: dict[str, Any],
    source_label: str,
    source_path: str,
    source_id: str,
    judge_model: str,
    auth_token: str,
    max_tokens: int,
    output_path: Path,
    overwrite: bool,
) -> None:
    try:
        record = judge_row(
            row,
            source_label=source_label,
            source_path=source_path,
            source_id=source_id,
            judge_model=judge_model,
            auth_token=auth_token,
            max_tokens=max_tokens,
        )
        write_judge_records(path=output_path, records=[record], overwrite=overwrite)
    except JudgeRequestError as exc:
        st.error(str(exc))
        if exc.details:
            with st.expander("Judge error details", expanded=True):
                st.json(exc.details)
        return
    except Exception as exc:
        st.error(f"Unexpected judge error: {exc}")
        with st.expander("Unexpected error details", expanded=True):
            st.code(traceback.format_exc(), language="text")
        return

    st.cache_data.clear()
    st.success(f"Saved annotation to {output_path.name}")
    st.rerun()


def _batch_job_snapshot(job_id: str) -> dict[str, Any] | None:
    with BATCH_RUNTIME["lock"]:
        job = BATCH_RUNTIME["jobs"].get(job_id)
        if not job:
            return None
        snapshot = {key: value for key, value in job.items() if key != "cancel_event"}
        snapshot["failures"] = list(job.get("failures", []))
        return snapshot


def request_stop_batch_job(job_id: str) -> None:
    with BATCH_RUNTIME["lock"]:
        job = BATCH_RUNTIME["jobs"].get(job_id)
        if not job:
            return
        job["cancel_requested"] = True
        job["status_message"] = "Stop requested. Waiting for in-flight requests to finish."
        cancel_event = job.get("cancel_event")
    if cancel_event is not None:
        cancel_event.set()


def _execute_batch_job(
    *,
    job_id: str,
    rows: pd.DataFrame,
    source_specs: list[SourceSpec],
    visible_source_ids: set[str],
    existing_annotations: dict[tuple[str, str], dict[str, Any]],
    judge_model: str,
    auth_token: str,
    max_tokens: int,
    output_dir: str,
    overwrite: bool,
    max_workers: int,
) -> None:
    spec_by_id = source_spec_map(source_specs)
    candidate_rows = rows[rows["source_id"].isin(list(visible_source_ids))]
    queued_jobs: list[tuple[dict[str, Any], SourceSpec]] = []
    skipped = 0

    for _, row in candidate_rows.iterrows():
        source_id = row["source_id"]
        spec = spec_by_id[source_id]
        key = annotation_lookup_key(str(row["eid"]), source_id)
        if not overwrite and key in existing_annotations:
            skipped += 1
            continue
        queued_jobs.append((row.to_dict(), spec))

    with BATCH_RUNTIME["lock"]:
        job = BATCH_RUNTIME["jobs"][job_id]
        job["skipped"] = skipped
        job["total_jobs"] = len(queued_jobs)
        job["status_message"] = "Preparing batch job."

    if not queued_jobs:
        with BATCH_RUNTIME["lock"]:
            job = BATCH_RUNTIME["jobs"][job_id]
            job["status"] = "completed"
            job["status_message"] = "No pending rows to judge."
            job["finished"] = True
        return

    def update_job(**kwargs: Any) -> None:
        with BATCH_RUNTIME["lock"]:
            if job_id in BATCH_RUNTIME["jobs"]:
                BATCH_RUNTIME["jobs"][job_id].update(kwargs)

    def worker(payload: dict[str, Any], spec: SourceSpec) -> dict[str, Any]:
        return judge_row(
            payload,
            source_label=spec.label,
            source_path=str(spec.csv_path),
            source_id=spec.source_id,
            judge_model=judge_model,
            auth_token=auth_token,
            max_tokens=max_tokens,
        )

    cancel_event = BATCH_RUNTIME["jobs"][job_id]["cancel_event"]
    completed_jobs = 0
    written = 0
    failed = 0
    failure_messages: list[tuple[str, dict[str, Any] | None]] = []
    status_message = "Running batch."

    try:
        queued_iter = iter(queued_jobs)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_map: dict[Any, tuple[dict[str, Any], SourceSpec]] = {}

            while len(future_map) < max_workers:
                try:
                    payload, spec = next(queued_iter)
                except StopIteration:
                    break
                future_map[pool.submit(worker, payload, spec)] = (payload, spec)

            while future_map:
                for future in as_completed(list(future_map.keys())):
                    payload, spec = future_map.pop(future)
                    completed_jobs += 1
                    try:
                        record = future.result()
                        out_path = judge_output_path(spec.csv_path.stem, judge_model, output_dir)
                        write_judge_records(path=out_path, records=[record], overwrite=overwrite)
                        written += 1
                        status_message = f"Judged {payload['eid']} from {spec.label} ({completed_jobs}/{len(queued_jobs)})"
                    except JudgeRequestError as exc:
                        failed += 1
                        failure_messages.append((f"{payload['eid']} / {spec.label}: {exc}", exc.details or None))
                        status_message = f"Failed {payload['eid']} from {spec.label} ({completed_jobs}/{len(queued_jobs)})"
                    except Exception as exc:
                        failed += 1
                        failure_messages.append(
                            (
                                f"{payload['eid']} / {spec.label}: {exc}",
                                {"traceback": traceback.format_exc()},
                            )
                        )
                        status_message = f"Failed {payload['eid']} from {spec.label} ({completed_jobs}/{len(queued_jobs)})"

                    update_job(
                        completed_jobs=completed_jobs,
                        written=written,
                        failed=failed,
                        failures=failure_messages,
                        progress=(completed_jobs / len(queued_jobs)) if queued_jobs else 1.0,
                        status_message=status_message,
                    )

                    if cancel_event.is_set():
                        status_message = "Stop requested. Finishing in-flight requests only."
                    while not cancel_event.is_set() and len(future_map) < max_workers:
                        try:
                            next_payload, next_spec = next(queued_iter)
                        except StopIteration:
                            break
                        future_map[pool.submit(worker, next_payload, next_spec)] = (next_payload, next_spec)
                    break

        final_status = "stopped" if cancel_event.is_set() else "completed"
        final_message = "Batch stopped." if cancel_event.is_set() else "Batch finished."
        update_job(
            status=final_status,
            finished=True,
            status_message=final_message,
            written=written,
            failed=failed,
            failures=failure_messages,
            progress=1.0 if not queued_jobs else (completed_jobs / len(queued_jobs)),
        )
    except Exception as exc:
        failure_messages.append((f"Batch worker error: {exc}", {"traceback": traceback.format_exc()}))
        update_job(
            status="failed",
            finished=True,
            failed=failed + 1,
            failures=failure_messages,
            status_message="Batch crashed unexpectedly.",
        )


def start_batch_job(
    *,
    rows: pd.DataFrame,
    source_specs: list[SourceSpec],
    visible_source_ids: set[str],
    existing_annotations: dict[tuple[str, str], dict[str, Any]],
    judge_model: str,
    auth_token: str,
    max_tokens: int,
    output_dir: str,
    overwrite: bool,
    max_workers: int,
) -> str:
    job_id = uuid.uuid4().hex
    with BATCH_RUNTIME["lock"]:
        BATCH_RUNTIME["jobs"][job_id] = {
            "job_id": job_id,
            "status": "running",
            "finished": False,
            "progress": 0.0,
            "completed_jobs": 0,
            "total_jobs": 0,
            "written": 0,
            "skipped": 0,
            "failed": 0,
            "failures": [],
            "judge_model": judge_model,
            "status_message": "Starting batch job.",
            "cancel_requested": False,
            "cancel_event": threading.Event(),
        }

    thread = threading.Thread(
        target=_execute_batch_job,
        kwargs={
            "job_id": job_id,
            "rows": rows.copy(),
            "source_specs": source_specs,
            "visible_source_ids": set(visible_source_ids),
            "existing_annotations": dict(existing_annotations),
            "judge_model": judge_model,
            "auth_token": auth_token,
            "max_tokens": max_tokens,
            "output_dir": output_dir,
            "overwrite": overwrite,
            "max_workers": max_workers,
        },
        daemon=True,
    )
    thread.start()
    return job_id


@st.fragment(run_every="2s")
def render_active_batch_panel() -> None:
    job_id = st.session_state.get("active_batch_job_id")
    if not job_id:
        return

    job = _batch_job_snapshot(job_id)
    if not job:
        st.warning("Batch status is temporarily unavailable.")
        return

    if job.get("finished"):
        st.cache_data.clear()
        st.session_state["last_batch_report"] = {
            "written": job.get("written", 0),
            "skipped": job.get("skipped", 0),
            "failed": job.get("failed", 0),
            "failures": job.get("failures", []),
            "judge_model": job.get("judge_model"),
        }
        st.session_state["active_batch_job_id"] = None
        clear_active_batch_job(job_id)
        if job.get("status") == "stopped":
            st.warning(
                f"Batch stopped. Wrote {job.get('written', 0)}, skipped {job.get('skipped', 0)}, failed {job.get('failed', 0)}."
            )
        elif job.get("failed"):
            st.warning(
                f"Batch finished. Wrote {job.get('written', 0)}, skipped {job.get('skipped', 0)}, failed {job.get('failed', 0)}."
            )
        else:
            st.success(
                f"Batch finished. Wrote {job.get('written', 0)}, skipped {job.get('skipped', 0)}."
            )
        st.rerun()
        return

    total_jobs = int(job.get("total_jobs", 0))
    completed_jobs = int(job.get("completed_jobs", 0))
    written = int(job.get("written", 0))
    skipped = int(job.get("skipped", 0))
    failed = int(job.get("failed", 0))

    st.info(f"Active batch judge: {job.get('judge_model', 'unknown model')}")
    live_cols = st.columns(4, gap="small")
    live_cols[0].metric("Completed", f"{completed_jobs}/{total_jobs}" if total_jobs else "0/0")
    live_cols[1].metric("Written", written)
    live_cols[2].metric("Skipped", skipped)
    live_cols[3].metric("Failed", failed)
    st.progress(float(job.get("progress", 0.0)))
    st.caption(job.get("status_message", "Batch running."))

    running_cols = st.columns(2, gap="small")
    with running_cols[0]:
        if st.button("Refresh batch status", use_container_width=True):
            st.rerun()
    with running_cols[1]:
        if st.button("Stop batch", use_container_width=True):
            request_stop_batch_job(job["job_id"])
            st.rerun()


def compute_batch_counts(
    *,
    rows: pd.DataFrame,
    visible_source_ids: set[str],
    existing_annotations: dict[tuple[str, str], dict[str, Any]],
    overwrite: bool,
) -> tuple[int, int, int]:
    candidate_rows = rows[rows["source_id"].isin(list(visible_source_ids))]
    total = len(candidate_rows)
    existing = 0
    if not overwrite:
        for _, row in candidate_rows.iterrows():
            key = annotation_lookup_key(str(row["eid"]), row["source_id"])
            if key in existing_annotations:
                existing += 1
    pending = total if overwrite else total - existing
    return total, existing, pending


def select_batch_rows_for_run(
    *,
    rows: pd.DataFrame,
    visible_source_ids: set[str],
    existing_annotations: dict[tuple[str, str], dict[str, Any]],
    overwrite: bool,
    run_limit: int,
) -> pd.DataFrame:
    candidate_rows = rows[rows["source_id"].isin(list(visible_source_ids))]
    if run_limit <= 0:
        return candidate_rows

    selected_indexes: list[Any] = []
    pending_count = 0
    for index, row in candidate_rows.iterrows():
        selected_indexes.append(index)
        key = annotation_lookup_key(str(row["eid"]), row["source_id"])
        is_existing = (not overwrite) and key in existing_annotations
        if not is_existing:
            pending_count += 1
            if pending_count >= run_limit:
                break

    return candidate_rows.loc[selected_indexes]


def annotation_spend_total(
    *,
    frame: pd.DataFrame,
    eids: set[str] | None = None,
    source_ids: set[str] | None = None,
) -> float:
    if frame.empty:
        return 0.0

    scoped = frame.copy()
    if eids is not None:
        scoped = scoped[scoped["eid"].astype(str).isin(eids)]
    if source_ids is not None and "source_id" in scoped.columns:
        scoped = scoped[scoped["source_id"].astype(str).isin(source_ids)]
    if scoped.empty or "request_cost" not in scoped.columns:
        return 0.0

    costs = pd.to_numeric(scoped["request_cost"], errors="coerce").fillna(0.0)
    return float(costs.sum())


def entry_score_map(
    *,
    annotation_map: dict[tuple[str, str], dict[str, Any]],
    visible_source_ids: set[str],
) -> dict[str, float]:
    per_eid: dict[str, list[float]] = {}
    for (eid, source_id), record in annotation_map.items():
        if source_id not in visible_source_ids:
            continue
        score = record.get("faithfulness_score")
        if score is None:
            score = record.get("score")
        if score is None:
            continue
        per_eid.setdefault(str(eid), []).append(float(score))

    return {
        eid: sum(scores) / len(scores)
        for eid, scores in per_eid.items()
        if scores
    }


st.title("LLM Judge Workspace")
st.caption("Browse modified triples, compare model outputs, and manage judge annotations in one place.")

if "batch_row_limit" not in st.session_state:
    st.session_state["batch_row_limit"] = 0

default_xml = str(infer_default_xml_path())
default_flat_csv = str(infer_default_flat_csv_path())
default_output_dir = str(repo_root() / DEFAULT_OUTPUT_DIR)
presets = dataset_presets()
dataset_preset_options = list(presets.keys())
stored_dataset_preset = st.session_state.get("dataset_preset", "CF preset")
if stored_dataset_preset not in dataset_preset_options:
    legacy_preset_map = {
        "CF flat CSV preset": "CSV CF preset",
        "FA flat CSV preset": "CSV FA preset",
    }
    stored_dataset_preset = legacy_preset_map.get(stored_dataset_preset, "CF preset")

with st.sidebar:
    st.header("Sources")
    selected_dataset_preset = st.selectbox(
        "Dataset preset",
        options=dataset_preset_options,
        index=dataset_preset_options.index(stored_dataset_preset),
    )
    st.session_state["dataset_preset"] = selected_dataset_preset
    if st.button("Use preset paths", use_container_width=True):
        st.session_state["dataset_path"] = presets[selected_dataset_preset]["dataset_path"]
        st.session_state["raw_sources"] = presets[selected_dataset_preset]["outputs_text"]
        st.rerun()

    dataset_path = st.text_input(
        "Dataset XML or CSV path",
        value=st.session_state.get("dataset_path", presets[selected_dataset_preset]["dataset_path"] if selected_dataset_preset in presets else default_xml),
        help="Supports either the original XML benchmark file or the CSV produced from it, such as data/webnlg_cf.csv.",
    )
    st.session_state["dataset_path"] = dataset_path
    st.caption("CSV can be created with `xml2csv.py`. Using the XML or the corresponding CSV should give the same dataset content in this app.")
    raw_sources = st.text_area(
        "Model output CSVs",
        value=st.session_state.get("raw_sources", presets[selected_dataset_preset]["outputs_text"] if selected_dataset_preset in presets else default_output_text()),
        height=120,
        help="Use one path per line. Optional label syntax: My model :: /path/to/file.csv",
    )
    st.session_state["raw_sources"] = raw_sources
    if st.button("Refresh loaded files", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.header("Judge Settings")
    judge_model = st.text_input("Judge model", value=st.session_state.get("judge_model", DEFAULT_JUDGE_MODEL))
    judge_max_tokens = st.number_input(
        "Judge max tokens",
        min_value=256,
        max_value=12000,
        value=int(st.session_state.get("judge_max_tokens", DEFAULT_JUDGE_MAX_TOKENS)),
        step=256,
        help="Maximum completion tokens for each judge request. Thinking models may need more, but non-thinking models are usually a better fit for this task.",
    )
    auth_token = st.text_input(
        "API key / token",
        value=st.session_state.get("judge_auth_token", os.getenv("AUTH_TOKEN") or streamlit_secret("AUTH_TOKEN", "")) or "",
        type="password",
    )
    output_dir = st.text_input("Annotation output directory", value=st.session_state.get("output_dir", default_output_dir))
    overwrite_existing = st.checkbox(
        "Rerun and overwrite existing annotations",
        value=bool(st.session_state.get("overwrite_existing", False)),
    )
    batch_concurrency = st.number_input(
        "Batch concurrency",
        min_value=1,
        max_value=20,
        value=int(st.session_state.get("batch_concurrency", 4)),
        step=1,
        help="How many judge requests to run at the same time during batch mode.",
    )
    skip_selected_model_existing = st.checkbox(
        "Skip rows already judged by selected model",
        value=bool(st.session_state.get("skip_selected_model_existing", True)),
        help="When enabled, batch skip checks only look at annotations created with the current Judge model setting. Results from other judge models do not count as existing.",
    )
    st.session_state["judge_model"] = judge_model
    st.session_state["judge_max_tokens"] = int(judge_max_tokens)
    st.session_state["judge_auth_token"] = auth_token
    st.session_state["output_dir"] = output_dir
    st.session_state["overwrite_existing"] = overwrite_existing
    st.session_state["batch_concurrency"] = int(batch_concurrency)
    st.session_state["skip_selected_model_existing"] = skip_selected_model_existing

try:
    entries_df = cached_entry_table(dataset_path, file_signature(dataset_path))
except Exception as exc:
    st.error(f"Unable to load dataset source: {exc}")
    st.stop()

source_specs, outputs_df, output_errors = cached_outputs(raw_sources, source_cache_signature(raw_sources))
if output_errors:
    for error in output_errors:
        st.warning(error)

visible_source_ids: set[str] = set()
with st.sidebar:
    st.header("Filters")
    categories = sorted(entries_df["category"].dropna().unique().tolist())
    if "selected_categories" not in st.session_state:
        st.session_state.selected_categories = categories
    else:
        st.session_state.selected_categories = [
            category for category in st.session_state.selected_categories if category in categories
        ]
        if not st.session_state.selected_categories and categories:
            st.session_state.selected_categories = []

    category_action_cols = st.columns(2, gap="small")
    with category_action_cols[0]:
        if st.button("Clear all categories", use_container_width=True):
            st.session_state.selected_categories = []
            st.rerun()
    with category_action_cols[1]:
        if st.button("Select all categories", use_container_width=True):
            st.session_state.selected_categories = categories
            st.rerun()

    with st.expander("Add categories", expanded=False):
        st.caption("Click a category to add it to the active filter.")
        for category in categories:
            already_selected = category in st.session_state.selected_categories
            if st.button(
                category,
                key=f"add_category_{category}",
                use_container_width=True,
                disabled=already_selected,
            ):
                st.session_state.selected_categories = st.session_state.selected_categories + [category]
                st.rerun()

    selected_categories = st.multiselect(
        "Category",
        options=categories,
        key="selected_categories",
        help="Use the buttons above to quickly add categories back after clearing.",
    )

    shape_types = sorted(entries_df["shape_type"].dropna().unique().tolist())
    selected_shape_types = st.multiselect("Shape type", options=shape_types, default=shape_types)
    min_size = int(entries_df["size"].min()) if not entries_df.empty else 0
    max_size = int(entries_df["size"].max()) if not entries_df.empty else 0
    selected_size = st.slider("Triple count", min_value=min_size, max_value=max_size, value=(min_size, max_size))
    eid_filter = st.text_input("Entry id contains", value="")

    st.header("Visible Outputs")
    visibility = source_visibility_controls(source_specs) if source_specs else {}
    visible_source_ids = {source_id for source_id, shown in visibility.items() if shown}
    browse_scope = st.radio(
        "Browse scope",
        options=[
            "All filtered XML entries",
            "Only entries with visible outputs",
            "Only entries with annotations",
        ],
        index=0,
        help="Choose whether the browser should show all filtered XML entries, only those present in visible model outputs, or only those that already have any saved annotations.",
    )

filtered_entries = entries_df.copy()
if categories and not selected_categories:
    filtered_entries = filtered_entries.iloc[0:0]
elif selected_categories:
    filtered_entries = filtered_entries[filtered_entries["category"].isin(selected_categories)]
if selected_shape_types:
    filtered_entries = filtered_entries[filtered_entries["shape_type"].isin(selected_shape_types)]
filtered_entries = filtered_entries[
    filtered_entries["size"].between(selected_size[0], selected_size[1])
]
if eid_filter:
    filtered_entries = filtered_entries[
        filtered_entries["eid"].astype(str).str.contains(eid_filter, case=False, na=False)
    ]

if filtered_entries.empty:
    st.warning("No entries match the current filters.")
    st.stop()

active_source_specs = [spec for spec in source_specs if spec.source_id in visible_source_ids]
source_spec_tuple = tuple((spec.label, str(spec.csv_path), spec.source_id) for spec in source_specs)
adhoc_dataset_name = Path(dataset_path).stem
annotation_signature = annotation_cache_signature(output_dir, source_spec_tuple, adhoc_dataset_name)
annotations_df = cached_annotations(output_dir, source_spec_tuple, annotation_signature)
adhoc_annotations_df = cached_adhoc_annotations(output_dir, adhoc_dataset_name, annotation_signature)
annotation_map = latest_annotation_map(annotations_df)
adhoc_map = latest_annotation_map(adhoc_annotations_df)
selected_model_annotation_map = latest_annotation_map(annotations_for_selected_model(annotations_df, judge_model))

merged_outputs = joined_output_rows(entries_df, outputs_df)
visible_outputs = merged_outputs[merged_outputs["source_id"].isin(list(visible_source_ids))] if not merged_outputs.empty else merged_outputs
filtered_batch_rows = visible_outputs[visible_outputs["eid"].isin(filtered_entries["eid"].tolist())] if not visible_outputs.empty else visible_outputs
batch_row_limit = int(st.session_state.get("batch_row_limit", 0))
batch_existing_lookup = selected_model_annotation_map if skip_selected_model_existing else {}
batch_rows_for_run = select_batch_rows_for_run(
    rows=filtered_batch_rows,
    visible_source_ids=visible_source_ids,
    existing_annotations=batch_existing_lookup,
    overwrite=overwrite_existing,
    run_limit=batch_row_limit,
)
filtered_output_eids = set(filtered_batch_rows["eid"].astype(str).tolist()) if not filtered_batch_rows.empty else set()
filtered_annotated_eids = {
    str(eid)
    for (eid, source_id), _record in annotation_map.items()
    if source_id in visible_source_ids and str(eid) in set(filtered_entries["eid"].astype(str).tolist())
}

if browse_scope == "Only entries with visible outputs":
    browser_entries = filtered_entries[filtered_entries["eid"].astype(str).isin(filtered_output_eids)]
elif browse_scope == "Only entries with annotations":
    browser_entries = filtered_entries[filtered_entries["eid"].astype(str).isin(filtered_annotated_eids)]
else:
    browser_entries = filtered_entries

sort_mode = st.session_state.get("browser_sort_mode", "Entry id")
score_map = entry_score_map(annotation_map=annotation_map, visible_source_ids=visible_source_ids)
browser_entries = browser_entries.copy()
browser_entries["annotation_score"] = browser_entries["eid"].astype(str).map(score_map)

if sort_mode == "Annotation score ascending":
    browser_entries = browser_entries.sort_values(
        by=["annotation_score", "eid"],
        ascending=[True, True],
        na_position="last",
    ).reset_index(drop=True)
elif sort_mode == "Annotation score descending":
    browser_entries = browser_entries.sort_values(
        by=["annotation_score", "eid"],
        ascending=[False, True],
        na_position="last",
    ).reset_index(drop=True)
else:
    browser_entries = browser_entries.sort_values("eid").reset_index(drop=True)

batch_total, batch_existing, batch_pending = compute_batch_counts(
    rows=batch_rows_for_run if batch_rows_for_run is not None else pd.DataFrame(),
    visible_source_ids=visible_source_ids,
    existing_annotations=batch_existing_lookup,
    overwrite=overwrite_existing,
)

active_batch_job_id = st.session_state.get("active_batch_job_id")
active_batch_job = _batch_job_snapshot(active_batch_job_id) if active_batch_job_id else None
if active_batch_job_id and active_batch_job is None:
    st.session_state["active_batch_job_id"] = None
    active_batch_job_id = None
if active_batch_job and active_batch_job.get("finished"):
    st.cache_data.clear()
    st.session_state["last_batch_report"] = {
        "written": active_batch_job.get("written", 0),
        "skipped": active_batch_job.get("skipped", 0),
        "failed": active_batch_job.get("failed", 0),
        "failures": active_batch_job.get("failures", []),
        "judge_model": active_batch_job.get("judge_model"),
    }
    st.session_state["active_batch_job_id"] = None
    clear_active_batch_job(active_batch_job_id)
    active_batch_job_id = None

with st.sidebar:
    st.header("Batch Actions")
    last_batch_report = st.session_state.get("last_batch_report")
    if last_batch_report:
        if last_batch_report.get("failed"):
            st.warning(
                f"Last batch: wrote {last_batch_report['written']}, skipped {last_batch_report['skipped']}, failed {last_batch_report['failed']}."
            )
            with st.expander("Batch failure details", expanded=True):
                for message, details in last_batch_report.get("failures", [])[:20]:
                    st.error(message)
                    if details:
                        st.json(details)
                if len(last_batch_report.get("failures", [])) > 20:
                    st.caption(f"Showing first 20 of {len(last_batch_report['failures'])} failures.")
        else:
            st.success(
                f"Last batch: wrote {last_batch_report['written']}, skipped {last_batch_report['skipped']}."
            )

    batch_preview_cols = st.columns(3, gap="small")
    batch_preview_cols[0].metric("Batch rows", batch_total)
    batch_preview_cols[1].metric("Existing", batch_existing if not overwrite_existing else 0)
    batch_preview_cols[2].metric("Will run", batch_pending)
    st.caption(
        "Batch uses the current filtered entries and only the currently visible outputs."
    )
    if int(batch_row_limit) > 0:
        st.caption(f"Batch row limit is active: up to {int(batch_row_limit)} non-skipped rows will be judged.")
    if skip_selected_model_existing:
        st.caption("Existing counts are checked only against annotations from the currently selected judge model.")
    else:
        st.caption("Skip checks are disabled. To replace same-model annotations cleanly, also enable overwrite.")
    if active_batch_job_id:
        render_active_batch_panel()
    else:
        batch_row_limit = st.number_input(
            "Batch row limit",
            min_value=0,
            max_value=100000,
            step=1,
            key="batch_row_limit",
            help="Maximum number of non-skipped rows to judge from the current filtered set. Use 0 for no limit.",
        )
        if st.button("Judge filtered visible outputs", use_container_width=True):
            if visible_outputs.empty:
                st.warning("No visible model outputs are loaded.")
            elif not auth_token:
                st.warning("Add an API key or token in the judge settings first.")
            else:
                job_id = start_batch_job(
                    rows=batch_rows_for_run,
                    source_specs=active_source_specs,
                    visible_source_ids=visible_source_ids,
                    existing_annotations=batch_existing_lookup,
                    judge_model=judge_model,
                    auth_token=auth_token,
                    max_tokens=int(judge_max_tokens),
                    output_dir=output_dir,
                    overwrite=overwrite_existing,
                    max_workers=int(batch_concurrency),
                )
                st.session_state["active_batch_job_id"] = job_id
                st.rerun()

summary_df = build_annotation_summary(
    outputs_frame=visible_outputs if not visible_outputs.empty else outputs_df[outputs_df["source_id"].isin(list(visible_source_ids))],
    annotations_frame=annotations_df,
    filtered_eids=browser_entries["eid"].astype(str).tolist(),
)

browser_eids = set(browser_entries["eid"].astype(str).tolist())
overall_spend = annotation_spend_total(frame=annotations_df) + annotation_spend_total(frame=adhoc_annotations_df)
browser_spend = annotation_spend_total(
    frame=annotations_df,
    eids=browser_eids,
    source_ids=visible_source_ids,
) + annotation_spend_total(
    frame=adhoc_annotations_df,
    eids=browser_eids,
)

metric_cols = st.columns(4)
metric_cols[0].metric("Filtered XML entries", len(filtered_entries))
metric_cols[1].metric("With visible outputs", len(filtered_output_eids))
metric_cols[2].metric("With annotations", len(filtered_annotated_eids))
metric_cols[3].metric("Browser entries", len(browser_entries))

context_cols = st.columns(5)
context_cols[0].metric("Loaded outputs", len(source_specs))
context_cols[1].metric("Visible outputs", len(visible_source_ids))
context_cols[2].metric("Judge model", judge_model)
context_cols[3].metric("Overall spend", f"${overall_spend:.4f}")
context_cols[4].metric("Browser spend", f"${browser_spend:.4f}")

if not summary_df.empty:
    st.subheader("Annotation Summary")
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

left, right = st.columns([1.0, 2.0], gap="large")

with left:
    st.subheader("Entry Browser")
    if browse_scope == "Only entries with visible outputs":
        st.caption("Showing only filtered entries that exist in the currently visible model outputs.")
    elif browse_scope == "Only entries with annotations":
        st.caption("Showing only filtered entries that already have saved annotations.")
    else:
        st.caption("Showing all entries from the filtered XML dataset, even if no output or annotation exists yet.")

    if browser_entries.empty:
        st.warning("No entries match the current browse scope.")
        st.stop()

    st.selectbox(
        "Order entries by",
        options=[
            "Entry id",
            "Annotation score ascending",
            "Annotation score descending",
        ],
        key="browser_sort_mode",
        help="Sort the browser list by entry id or by the average visible annotation score for that entry.",
    )
    selected_eid = entry_navigation(browser_entries)
    preview_columns = ["eid", "annotation_score", "category", "shape_type", "size", "num_modified_triples"]
    st.dataframe(browser_entries[preview_columns], use_container_width=True, hide_index=True, height=420)

selected_entry = browser_entries[browser_entries["eid"].astype(str) == str(st.session_state.selected_eid)].iloc[0].to_dict()

with right:
    st.subheader(f"Entry {selected_entry['eid']}")
    meta_cols = st.columns(4)
    meta_items = [
        ("Category", selected_entry["category"] or "Unknown"),
        ("Shape type", selected_entry["shape_type"] or "Unknown"),
        ("Triples", str(int(selected_entry["num_modified_triples"]))),
        ("Shape", selected_entry["shape"] or "Unknown"),
    ]
    for col, (label, value) in zip(meta_cols, meta_items):
        safe_label = html.escape(str(label))
        safe_value = html.escape(str(value))
        with col:
            st.markdown(
                (
                    "<div class='meta-card'>"
                    f"<div class='meta-label'>{safe_label}</div>"
                    f"<div class='meta-value'>{safe_value}</div>"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )

    st.markdown("**Modified triples**")
    triples = selected_entry.get("modified_triples_json") or []
    if triples:
        for triple in triples:
            st.code(triple, language="text")
    else:
        st.caption("No triples found for this entry.")

    st.subheader("Model Output")
    if not source_specs:
        st.info("Load one or more output CSVs in the sidebar to compare model generations.")
    else:
        visible_specs = [spec for spec in source_specs if spec.source_id in visible_source_ids]
        if not visible_specs:
            st.info("All loaded outputs are hidden. Enable at least one output in the sidebar.")
        else:
            columns = st.columns(min(len(visible_specs), 3))
            for index, spec in enumerate(visible_specs):
                column = columns[index % len(columns)]
                with column:
                    safe_source_label = html.escape(spec.label)
                    st.markdown(f"<div class='source-label'>{safe_source_label}</div>", unsafe_allow_html=True)
                    row_match = outputs_df[
                        (outputs_df["source_id"] == spec.source_id)
                        & (outputs_df["eid"].astype(str) == str(selected_entry["eid"]))
                    ]
                    if row_match.empty:
                        st.warning("No output for this entry in this CSV.")
                        continue

                    output_row = row_match.iloc[0].to_dict()
                    detail_row = {
                        "eid": selected_entry["eid"],
                        "category": selected_entry["category"],
                        "sentence": output_row["sentence"],
                        "modified_triples": selected_entry["modified_triples"],
                    }

                    st.markdown(
                        f"<div class='sentence-box'>{html.escape(str(output_row['sentence']))}</div>",
                        unsafe_allow_html=True,
                    )

                    record = annotation_map.get(annotation_lookup_key(selected_entry["eid"], spec.source_id))
                    render_annotation(record)
                    render_prompt(build_judge_prompt(detail_row))

                    if st.button("Judge this output", key=f"judge_{spec.source_id}_{selected_entry['eid']}", use_container_width=True):
                        if not auth_token:
                            st.warning("Add an API key or token first.")
                        else:
                            run_single_judge(
                                row=detail_row,
                                source_label=spec.label,
                source_path=str(spec.csv_path),
                source_id=spec.source_id,
                judge_model=judge_model,
                auth_token=auth_token,
                max_tokens=int(judge_max_tokens),
                output_path=judge_output_path(spec.csv_path.stem, judge_model, output_dir),
                overwrite=overwrite_existing,
            )

    st.subheader("Ad Hoc Judge")
    adhoc_key = f"adhoc_sentence_{selected_entry['eid']}"
    st.text_area(
        "Custom sentence for this entry",
        key=adhoc_key,
        height=120,
        placeholder="Paste or type a sentence to judge against the modified triples.",
    )
    adhoc_sentence = st.session_state.get(adhoc_key, "").strip()
    if adhoc_sentence:
        adhoc_source_id = sanitize_identifier("adhoc")
        adhoc_row = {
            "eid": selected_entry["eid"],
            "category": selected_entry["category"],
            "sentence": adhoc_sentence,
            "modified_triples": selected_entry["modified_triples"],
        }
        render_prompt(build_judge_prompt(adhoc_row))
        adhoc_record = adhoc_map.get(annotation_lookup_key(selected_entry["eid"], adhoc_source_id))
        render_annotation(adhoc_record)

        if st.button("Judge ad hoc sentence", key=f"judge_adhoc_{selected_entry['eid']}", use_container_width=False):
            if not auth_token:
                st.warning("Add an API key or token first.")
            else:
                run_single_judge(
                    row=adhoc_row,
                    source_label="Ad hoc",
                    source_path="adhoc",
                    source_id=adhoc_source_id,
                    judge_model=judge_model,
                    auth_token=auth_token,
                    max_tokens=int(judge_max_tokens),
                    output_path=adhoc_output_path(adhoc_dataset_name, judge_model, output_dir),
                    overwrite=overwrite_existing,
                )
    else:
        st.caption("Add a custom sentence to judge it for this entry.")
