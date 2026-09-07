"""Append-only persistence layer.

Requirement 11/12: facts are never overwritten.  ``save_fact`` looks up the
highest stored version of a ``fact_id``; if the incoming content differs it
writes ``version + 1`` and marks the old row ``superseded_by`` the new version
key.  Identical content is a no-op (duplicate detection).
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from typing import Any

from ..schemas.agent_io import AgentRunRecord, RunContext
from ..schemas.evaluation import (
    CatalystEvent,
    KillGateResult,
    Scenario,
    ScoreCard,
    Verdict,
)
from ..schemas.fact import Contradiction, Fact, Source
from ..schemas.validation import validate_fact, validate_source

_CONTENT_FIELDS = (
    "claim",
    "value",
    "evidence_class",
    "verified_status",
    "confidence",
    "materiality",
    "independent_confirmation",
    "contradicting_evidence",
    "source_url",
    "event_date",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Repository:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    # -- companies ---------------------------------------------------------
    def upsert_company(
        self,
        ticker: str,
        company_name: str = "UNKNOWN",
        *,
        cik: str = "UNKNOWN",
        exchange: str = "UNKNOWN",
        country: str = "UNKNOWN",
        sector: str = "UNKNOWN",
        aliases: Sequence[str] = (),
    ) -> None:
        now = _now()
        self.conn.execute(
            """
            INSERT INTO companies(ticker, company_name, cik, exchange, country, sector,
                                  aliases, first_seen, last_seen)
            VALUES(?,?,?,?,?,?,?,?,?)
            ON CONFLICT(ticker) DO UPDATE SET
                company_name=excluded.company_name,
                cik=excluded.cik,
                exchange=excluded.exchange,
                country=excluded.country,
                sector=excluded.sector,
                aliases=excluded.aliases,
                last_seen=excluded.last_seen
            """,
            (
                ticker.upper(),
                company_name,
                cik,
                exchange,
                country,
                sector,
                ",".join(aliases),
                now,
                now,
            ),
        )
        self.conn.commit()

    def get_company(self, ticker: str) -> sqlite3.Row | None:
        cur = self.conn.execute("SELECT * FROM companies WHERE ticker = ?", (ticker.upper(),))
        return cur.fetchone()

    # -- runs --------------------------------------------------------------
    def start_run(self, ctx: RunContext) -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO runs(run_id, ticker, mode, status, offline,
                                           used_fixtures, started_at, finished_at, notes)
               VALUES(?,?,?,?,?,?,?,?,?)""",
            (
                ctx.run_id,
                ctx.ticker,
                ctx.mode,
                str(ctx.status),
                int(ctx.offline),
                int(ctx.use_fixtures),
                ctx.started_at or _now(),
                "",
                "\n".join(ctx.notes),
            ),
        )
        self.conn.commit()

    def finish_run(self, ctx: RunContext) -> None:
        self.conn.execute(
            "UPDATE runs SET status = ?, finished_at = ?, notes = ? WHERE run_id = ?",
            (str(ctx.status), _now(), "\n".join(ctx.notes), ctx.run_id),
        )
        self.conn.commit()

    def previous_run(self, ticker: str, exclude_run_id: str) -> sqlite3.Row | None:
        cur = self.conn.execute(
            """SELECT * FROM runs WHERE ticker = ? AND run_id != ?
               ORDER BY started_at DESC LIMIT 1""",
            (ticker.upper(), exclude_run_id),
        )
        return cur.fetchone()

    # -- sources -----------------------------------------------------------
    def save_source(self, source: Source) -> None:
        validate_source(source)
        row = source.to_row()
        cols = ",".join(row)
        marks = ",".join("?" * len(row))
        self.conn.execute(
            f"INSERT OR REPLACE INTO sources({cols}) VALUES({marks})", tuple(row.values())
        )

    def save_sources(self, sources: Iterable[Source]) -> None:
        for s in sources:
            self.save_source(s)
        self.conn.commit()

    def find_source_by_hash(self, content_hash: str) -> sqlite3.Row | None:
        if not content_hash or content_hash == "UNKNOWN":
            return None
        cur = self.conn.execute(
            "SELECT * FROM sources WHERE content_hash = ? LIMIT 1", (content_hash,)
        )
        return cur.fetchone()

    # -- facts -------------------------------------------------------------
    def latest_fact_row(self, fact_id: str) -> sqlite3.Row | None:
        cur = self.conn.execute(
            "SELECT * FROM facts WHERE fact_id = ? ORDER BY version DESC LIMIT 1", (fact_id,)
        )
        return cur.fetchone()

    def save_fact(self, fact: Fact) -> tuple[str, int]:
        """Persist a fact append-only.  Returns (fact_id, stored_version)."""
        validate_fact(fact)
        existing = self.latest_fact_row(fact.fact_id)
        row = fact.to_row()
        if existing is not None:
            unchanged = all(str(existing[f]) == str(row[f]) for f in _CONTENT_FIELDS)
            if unchanged:
                return fact.fact_id, int(existing["version"])  # duplicate: no-op
            new_version = int(existing["version"]) + 1
            row["version"] = new_version
            # mark predecessor superseded (allowed: not a content column)
            self.conn.execute(
                "UPDATE facts SET superseded_by = ? WHERE fact_id = ? AND version = ?",
                (f"{fact.fact_id}#v{new_version}", fact.fact_id, existing["version"]),
            )
        row["created_at"] = _now()
        cols = ",".join(row)
        marks = ",".join("?" * len(row))
        self.conn.execute(f"INSERT INTO facts({cols}) VALUES({marks})", tuple(row.values()))
        self.conn.commit()
        return fact.fact_id, int(row["version"])

    def save_facts(self, facts: Iterable[Fact]) -> int:
        count = 0
        for f in facts:
            self.save_fact(f)
            count += 1
        return count

    def latest_facts(self, ticker: str) -> list[sqlite3.Row]:
        cur = self.conn.execute(
            """SELECT f.* FROM facts f
               JOIN (SELECT fact_id, MAX(version) AS v FROM facts WHERE ticker = ?
                     GROUP BY fact_id) m
               ON f.fact_id = m.fact_id AND f.version = m.v
               ORDER BY f.category, f.fact_id""",
            (ticker.upper(),),
        )
        return list(cur.fetchall())

    def fact_versions(self, fact_id: str) -> list[sqlite3.Row]:
        cur = self.conn.execute(
            "SELECT * FROM facts WHERE fact_id = ? ORDER BY version", (fact_id,)
        )
        return list(cur.fetchall())

    def facts_for_run(self, run_id: str) -> list[sqlite3.Row]:
        cur = self.conn.execute("SELECT * FROM facts WHERE run_id = ?", (run_id,))
        return list(cur.fetchall())

    # -- contradictions ----------------------------------------------------
    def save_contradictions(self, items: Iterable[Contradiction]) -> int:
        n = 0
        for c in items:
            row = c.to_row()
            row["created_at"] = _now()
            cols = ",".join(row)
            marks = ",".join("?" * len(row))
            self.conn.execute(
                f"INSERT OR REPLACE INTO contradictions({cols}) VALUES({marks})",
                tuple(row.values()),
            )
            n += 1
        self.conn.commit()
        return n

    # -- structured domain tables -----------------------------------------
    def save_capital_structure(self, run_id: str, ticker: str, data: dict[str, Any]) -> None:
        payload = {
            "run_id": run_id,
            "ticker": ticker.upper(),
            "created_at": _now(),
            **{k: v for k, v in data.items() if k not in ("run_id", "ticker", "created_at")},
        }
        cols = ",".join(payload)
        marks = ",".join("?" * len(payload))
        self.conn.execute(
            f"INSERT OR REPLACE INTO capital_structure({cols}) VALUES({marks})",
            tuple(payload.values()),
        )
        self.conn.commit()

    def save_clinical_trials(self, run_id: str, ticker: str, trials: Iterable[dict]) -> int:
        n = 0
        for t in trials:
            payload = {"run_id": run_id, "ticker": ticker.upper(), "created_at": _now(), **t}
            cols = ",".join(payload)
            marks = ",".join("?" * len(payload))
            self.conn.execute(
                f"INSERT OR REPLACE INTO clinical_trials({cols}) VALUES({marks})",
                tuple(payload.values()),
            )
            n += 1
        self.conn.commit()
        return n

    def save_regulatory_events(self, run_id: str, ticker: str, events: Iterable[dict]) -> int:
        n = 0
        for e in events:
            payload = {"run_id": run_id, "ticker": ticker.upper(), "created_at": _now(), **e}
            cols = ",".join(payload)
            marks = ",".join("?" * len(payload))
            self.conn.execute(
                f"INSERT OR REPLACE INTO regulatory_events({cols}) VALUES({marks})",
                tuple(payload.values()),
            )
            n += 1
        self.conn.commit()
        return n

    def save_insider_trades(self, run_id: str, ticker: str, trades: Iterable[dict]) -> int:
        n = 0
        for t in trades:
            payload = {"run_id": run_id, "ticker": ticker.upper(), "created_at": _now(), **t}
            cols = ",".join(payload)
            marks = ",".join("?" * len(payload))
            self.conn.execute(
                f"INSERT OR REPLACE INTO insider_trades({cols}) VALUES({marks})",
                tuple(payload.values()),
            )
            n += 1
        self.conn.commit()
        return n

    def save_catalysts(self, run_id: str, ticker: str, events: Iterable[CatalystEvent]) -> int:
        n = 0
        for e in events:
            payload = {
                "run_id": run_id,
                "ticker": ticker.upper(),
                "created_at": _now(),
                **e.to_row(),
            }
            cols = ",".join(payload)
            marks = ",".join("?" * len(payload))
            self.conn.execute(
                f"INSERT OR REPLACE INTO catalysts({cols}) VALUES({marks})",
                tuple(payload.values()),
            )
            n += 1
        self.conn.commit()
        return n

    # -- evaluation outputs ------------------------------------------------
    def save_scores(self, card: ScoreCard) -> int:
        n = 0
        capped = set(card.capped_by_kill_gate)
        for dim, value in card.scores.items():
            self.conn.execute(
                """INSERT OR REPLACE INTO scores(run_id, ticker, dimension, score,
                                                 confidence, rationale, capped_by_kill_gate,
                                                 created_at)
                   VALUES(?,?,?,?,?,?,?,?)""",
                (
                    card.run_id,
                    card.ticker,
                    dim,
                    value,
                    card.per_score_confidence.get(dim, card.evidence_confidence),
                    card.rationale.get(dim, ""),
                    int(dim in capped),
                    _now(),
                ),
            )
            n += 1
        self.conn.commit()
        return n

    def save_kill_gate(self, run_id: str, ticker: str, gate: KillGateResult) -> int:
        n = 0
        for a in gate.assessments:
            self.conn.execute(
                """INSERT OR REPLACE INTO kill_assessments(run_id, ticker, category, level,
                                                           rationale, evidence_confidence,
                                                           confirmation, created_at)
                   VALUES(?,?,?,?,?,?,?,?)""",
                (
                    run_id,
                    ticker.upper(),
                    str(a.category),
                    str(a.level),
                    a.rationale,
                    a.evidence_confidence,
                    str(a.confirmation),
                    _now(),
                ),
            )
            n += 1
        self.conn.commit()
        return n

    def save_evidence_sufficiency(self, run_id: str, ticker: str, matrix) -> int:
        """Persist the Evidence Sufficiency Matrix (requirement DG5/DG10)."""
        n = 0
        for domain, entry in matrix.domains.items():
            self.conn.execute(
                """INSERT OR REPLACE INTO evidence_sufficiency(run_id, ticker, domain,
                       search_status, evidence_sufficiency_status, decision_grade_fact_count,
                       total_fact_count, reason, created_at)
                   VALUES(?,?,?,?,?,?,?,?,?)""",
                (
                    run_id,
                    ticker.upper(),
                    str(domain),
                    str(entry.search_status),
                    str(entry.evidence_sufficiency_status),
                    entry.decision_grade_fact_count,
                    entry.total_fact_count,
                    entry.reason,
                    _now(),
                ),
            )
            n += 1
        self.conn.commit()
        return n

    def save_scenarios(self, run_id: str, ticker: str, scenarios: Iterable[Scenario]) -> int:
        n = 0
        for s in scenarios:
            payload = {
                "run_id": run_id,
                "ticker": ticker.upper(),
                "created_at": _now(),
                **s.to_row(),
            }
            cols = ",".join(payload)
            marks = ",".join("?" * len(payload))
            self.conn.execute(
                f"INSERT OR REPLACE INTO scenarios({cols}) VALUES({marks})",
                tuple(payload.values()),
            )
            n += 1
        self.conn.commit()
        return n

    def save_agent_run(self, record: AgentRunRecord) -> None:
        row = record.to_row()
        cols = ",".join(row)
        marks = ",".join("?" * len(row))
        self.conn.execute(
            f"INSERT OR REPLACE INTO agent_runs({cols}) VALUES({marks})", tuple(row.values())
        )
        self.conn.commit()

    # -- phase 2: checkpoints, coverage, llm calls, escalations -----------
    def save_checkpoint(self, checkpoint) -> None:
        from ..orchestrator.resume import serialize_payload

        self.conn.execute(
            """INSERT OR REPLACE INTO run_checkpoints(run_id, stage, stage_index, status,
                                                      payload, fact_count, created_at)
               VALUES(?,?,?,?,?,?,?)""",
            (
                checkpoint.run_id,
                checkpoint.stage,
                checkpoint.stage_index,
                checkpoint.status,
                serialize_payload(checkpoint.payload),
                checkpoint.fact_count,
                _now(),
            ),
        )
        self.conn.commit()

    def checkpoints(self, run_id: str) -> list:
        from ..orchestrator.resume import Checkpoint

        rows = self.conn.execute(
            "SELECT * FROM run_checkpoints WHERE run_id = ? ORDER BY stage_index", (run_id,)
        ).fetchall()
        out = []
        for row in rows:
            try:
                payload = json.loads(row["payload"] or "{}")
            except json.JSONDecodeError:
                payload = {}
            out.append(
                Checkpoint(
                    run_id=row["run_id"],
                    stage=row["stage"],
                    stage_index=int(row["stage_index"]),
                    status=row["status"],
                    payload=payload if isinstance(payload, dict) else {},
                    fact_count=int(row["fact_count"]),
                )
            )
        return out

    def facts_for_resume(self, run_id: str) -> list:
        """Rehydrate the latest version of every fact recorded for a run."""
        from ..schemas.enums import (
            EvidenceClass,
            FactCategory,
            Materiality,
            Provenance,
            SourceTier,
            VerifiedStatus,
        )
        from ..schemas.fact import Fact

        rows = self.conn.execute(
            """SELECT f.* FROM facts f
               JOIN (SELECT fact_id, MAX(version) AS v FROM facts WHERE run_id = ?
                     GROUP BY fact_id) m
               ON f.fact_id = m.fact_id AND f.version = m.v""",
            (run_id,),
        ).fetchall()
        facts = []
        for row in rows:
            facts.append(
                Fact(
                    fact_id=row["fact_id"],
                    ticker=row["ticker"],
                    category=FactCategory(row["category"]),
                    claim=row["claim"],
                    evidence_class=EvidenceClass(row["evidence_class"]),
                    source_id=row["source_id"],
                    source_url=row["source_url"],
                    source_title=row["source_title"] or "",
                    source_tier=SourceTier(row["source_tier"]),
                    publication_date=row["publication_date"],
                    event_date=row["event_date"],
                    effective_date=row["effective_date"],
                    filing_date=row["filing_date"],
                    verified_status=VerifiedStatus(row["verified_status"]),
                    confidence=float(row["confidence"]),
                    company_claim=bool(row["company_claim"]),
                    independent_confirmation=bool(row["independent_confirmation"]),
                    materiality=Materiality(row["materiality"]),
                    value=row["value"],
                    unit=row["unit"],
                    provenance=Provenance(row["provenance"]),
                    stale=bool(row["stale"]),
                    version=int(row["version"]),
                    run_id=row["run_id"],
                    notes=row["notes"] or "",
                )
            )
        return facts

    def save_research_coverage(self, run_id: str, ticker: str, coverage) -> int:
        n = 0
        for domain, entry in coverage.items():
            self.conn.execute(
                """INSERT OR REPLACE INTO research_coverage(run_id, ticker, domain, status,
                       queries_attempted, queries_executed, documents_found, paths, detail,
                       created_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?)""",
                (
                    run_id,
                    ticker.upper(),
                    str(domain),
                    str(entry.status),
                    entry.queries_attempted,
                    entry.queries_executed,
                    entry.documents_found,
                    ",".join(str(p) for p in entry.paths),
                    entry.detail,
                    _now(),
                ),
            )
            n += 1
        self.conn.commit()
        return n

    def save_llm_calls(self, run_id: str, calls) -> int:
        n = 0
        for call in calls:
            self.conn.execute(
                """INSERT INTO llm_calls(run_id, agent_id, model, input_tokens, output_tokens,
                       cache_read_tokens, duration_ms, ok, stop_reason, error, created_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    run_id,
                    call.agent_id,
                    call.model,
                    call.input_tokens,
                    call.output_tokens,
                    call.cache_read_tokens,
                    call.duration_ms,
                    int(call.ok),
                    call.stop_reason,
                    call.error[:500],
                    _now(),
                ),
            )
            n += 1
        self.conn.commit()
        return n

    def save_escalations(self, run_id: str, attempts) -> int:
        n = 0
        for attempt in attempts:
            self.conn.execute(
                """INSERT OR REPLACE INTO escalations(run_id, fact_id, reason, searched,
                       confirmed, confirming_url, queries, note, created_at)
                   VALUES(?,?,?,?,?,?,?,?,?)""",
                (
                    run_id,
                    attempt.fact_id,
                    attempt.reason,
                    int(attempt.searched),
                    int(attempt.confirmed),
                    attempt.confirming_url,
                    " | ".join(attempt.queries)[:1000],
                    attempt.note,
                    _now(),
                ),
            )
            n += 1
        self.conn.commit()
        return n

    # -- search discovery log (requirements M2/M3) ------------------------
    def save_discovery_log(self, discovery) -> tuple[int, int]:
        """Persist a DiscoveryLog: queries and hits, kept apart from facts."""
        n_queries = 0
        for query in discovery.queries:
            row = query.to_row()
            cols = ",".join(row)
            marks = ",".join("?" * len(row))
            self.conn.execute(
                f"INSERT OR REPLACE INTO search_queries({cols}) VALUES({marks})",
                tuple(row.values()),
            )
            n_queries += 1
        n_hits = 0
        for hit in discovery.hits:
            row = hit.to_row()
            cols = ",".join(row)
            marks = ",".join("?" * len(row))
            self.conn.execute(
                f"INSERT OR REPLACE INTO search_hits({cols}) VALUES({marks})",
                tuple(row.values()),
            )
            n_hits += 1
        self.conn.commit()
        return n_queries, n_hits

    def discovery_queries(self, run_id: str, purposes: tuple[str, ...] | None = None):
        if purposes:
            placeholders = ",".join("?" * len(purposes))
            return self.conn.execute(
                f"SELECT * FROM search_queries WHERE run_id = ? "
                f"AND query_purpose IN ({placeholders}) ORDER BY created_at",
                (run_id, *purposes),
            ).fetchall()
        return self.conn.execute(
            "SELECT * FROM search_queries WHERE run_id = ? ORDER BY created_at", (run_id,)
        ).fetchall()

    def discovery_hits(self, run_id: str, purposes: tuple[str, ...] | None = None):
        if purposes:
            placeholders = ",".join("?" * len(purposes))
            return self.conn.execute(
                f"SELECT * FROM search_hits WHERE run_id = ? "
                f"AND query_purpose IN ({placeholders}) ORDER BY rank",
                (run_id, *purposes),
            ).fetchall()
        return self.conn.execute(
            "SELECT * FROM search_hits WHERE run_id = ? ORDER BY rank", (run_id,)
        ).fetchall()

    def log_fetch(
        self,
        run_id: str,
        url: str,
        collector: str,
        outcome: str,
        http_status: int | None = None,
        attempts: int = 1,
        detail: str = "",
    ) -> None:
        self.conn.execute(
            """INSERT INTO fetch_log(run_id, url, collector, outcome, http_status,
                                     attempts, detail, created_at)
               VALUES(?,?,?,?,?,?,?,?)""",
            (run_id, url, collector, outcome, http_status, attempts, detail[:500], _now()),
        )
        self.conn.commit()

    # -- thesis versioning -------------------------------------------------
    def latest_thesis(self, ticker: str) -> sqlite3.Row | None:
        cur = self.conn.execute(
            "SELECT * FROM thesis_versions WHERE ticker = ? ORDER BY version DESC LIMIT 1",
            (ticker.upper(),),
        )
        return cur.fetchone()

    def save_thesis_version(
        self,
        ticker: str,
        run_id: str,
        verdict: Verdict,
        card: ScoreCard,
        diff: dict[str, Any],
        snapshot: dict[str, Any],
    ) -> int:
        prev = self.latest_thesis(ticker)
        version = 1 if prev is None else int(prev["version"]) + 1
        self.conn.execute(
            """INSERT INTO thesis_versions(ticker, version, run_id, action, evidence_confidence,
                                           headline, max_kill_level, what_changed, why_changed,
                                           new_facts, removed_assumptions, score_change,
                                           snapshot, research_status,
                                           blocking_verification_required, created_at)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                ticker.upper(),
                version,
                run_id,
                str(verdict.action),
                verdict.evidence_confidence,
                verdict.headline,
                str(verdict.kill_gate.max_level),
                json.dumps(diff.get("WHAT_CHANGED", []), ensure_ascii=False),
                json.dumps(diff.get("WHY_CHANGED", []), ensure_ascii=False),
                json.dumps(diff.get("NEW_FACT", []), ensure_ascii=False),
                json.dumps(diff.get("REMOVED_ASSUMPTION", []), ensure_ascii=False),
                json.dumps(diff.get("SCORE_CHANGE", {}), ensure_ascii=False),
                json.dumps(snapshot, ensure_ascii=False, default=str),
                str(getattr(verdict, "research_status", "COMPLETE")),
                json.dumps(
                    list(getattr(verdict, "blocking_verification_required", ())),
                    ensure_ascii=False,
                ),
                _now(),
            ),
        )
        self.conn.commit()
        return version
