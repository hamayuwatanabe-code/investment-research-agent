-- investment-research-agent persistent store (requirement 11).
-- Facts are APPEND-ONLY and versioned: a revised fact never overwrites its
-- predecessor.  The primary key is (fact_id, version).

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS companies (
    ticker              TEXT PRIMARY KEY,
    company_name        TEXT NOT NULL DEFAULT 'UNKNOWN',
    cik                 TEXT DEFAULT 'UNKNOWN',
    exchange            TEXT DEFAULT 'UNKNOWN',
    country             TEXT DEFAULT 'UNKNOWN',
    sector              TEXT DEFAULT 'UNKNOWN',
    aliases             TEXT DEFAULT '',
    first_seen          TEXT NOT NULL,
    last_seen           TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sources (
    source_id           TEXT NOT NULL,
    url                 TEXT NOT NULL,
    title               TEXT NOT NULL DEFAULT '',
    tier                TEXT NOT NULL,
    publisher           TEXT DEFAULT 'UNKNOWN',
    published_date      TEXT DEFAULT 'UNKNOWN',
    event_date          TEXT DEFAULT 'UNKNOWN',
    effective_date      TEXT DEFAULT 'UNKNOWN',
    filing_date         TEXT DEFAULT 'UNKNOWN',
    accession           TEXT DEFAULT 'UNKNOWN',
    retrieved_at        TEXT NOT NULL,
    provenance          TEXT NOT NULL DEFAULT 'LIVE',
    content_hash        TEXT DEFAULT 'UNKNOWN',
    syndicated_from     TEXT,
    excerpt             TEXT DEFAULT '',
    PRIMARY KEY (source_id)
);
CREATE INDEX IF NOT EXISTS idx_sources_hash ON sources(content_hash);

CREATE TABLE IF NOT EXISTS facts (
    fact_id                     TEXT NOT NULL,
    version                     INTEGER NOT NULL DEFAULT 1,
    ticker                      TEXT NOT NULL,
    category                    TEXT NOT NULL,
    claim                       TEXT NOT NULL,
    evidence_class              TEXT NOT NULL,
    source_id                   TEXT NOT NULL,
    source_url                  TEXT NOT NULL,
    source_title                TEXT DEFAULT '',
    source_tier                 TEXT NOT NULL,
    publication_date            TEXT DEFAULT 'UNKNOWN',
    event_date                  TEXT DEFAULT 'UNKNOWN',
    effective_date              TEXT DEFAULT 'UNKNOWN',
    filing_date                 TEXT DEFAULT 'UNKNOWN',
    verified_status             TEXT NOT NULL,
    confidence                  REAL NOT NULL DEFAULT 0.0,
    company_claim               INTEGER NOT NULL DEFAULT 0,
    independent_confirmation    INTEGER NOT NULL DEFAULT 0,
    corroborating_source_ids    TEXT DEFAULT '',
    contradicting_evidence      TEXT DEFAULT '',
    materiality                 TEXT NOT NULL DEFAULT 'INFORMATIONAL',
    value                       TEXT DEFAULT 'UNKNOWN',
    unit                        TEXT DEFAULT 'UNKNOWN',
    provenance                  TEXT NOT NULL DEFAULT 'LIVE',
    stale                       INTEGER NOT NULL DEFAULT 0,
    superseded_by               TEXT,
    run_id                      TEXT NOT NULL,
    notes                       TEXT DEFAULT '',
    tags                        TEXT DEFAULT '',
    content_kind                TEXT NOT NULL DEFAULT 'FULL_DOCUMENT',
    primary_source_url          TEXT,
    -- Phase 2 addition: DocumentStore identity (Document.doc_id) this fact
    -- was extracted from, when known. Nullable and additive -- see
    -- storage/db.py's ADDITIVE_COLUMNS for the same column applied to an
    -- existing database via ALTER TABLE. There is no DocumentStore table
    -- yet, so this column alone does not let a stored Fact's Document be
    -- reloaded after the process that produced it ends (see
    -- research/document_store.py's module docstring).
    document_id                 TEXT,
    created_at                  TEXT NOT NULL,
    PRIMARY KEY (fact_id, version)
);
CREATE INDEX IF NOT EXISTS idx_facts_ticker ON facts(ticker);
CREATE INDEX IF NOT EXISTS idx_facts_run ON facts(run_id);
CREATE INDEX IF NOT EXISTS idx_facts_category ON facts(ticker, category);

-- Guard rail: an UPDATE to an existing fact row is a bug, not a feature.
CREATE TRIGGER IF NOT EXISTS facts_are_append_only
BEFORE UPDATE OF claim, value, evidence_class, verified_status ON facts
BEGIN
    SELECT RAISE(ABORT, 'facts are append-only: write a new version instead');
END;

CREATE TABLE IF NOT EXISTS contradictions (
    contradiction_id    TEXT PRIMARY KEY,
    ticker              TEXT NOT NULL,
    kind                TEXT NOT NULL,
    description         TEXT NOT NULL,
    left_fact_id        TEXT NOT NULL,
    right_fact_id       TEXT NOT NULL,
    left_summary        TEXT DEFAULT '',
    right_summary       TEXT DEFAULT '',
    severity            TEXT NOT NULL DEFAULT 'MEDIUM',
    resolved            INTEGER NOT NULL DEFAULT 0,
    resolution_note     TEXT DEFAULT '',
    run_id              TEXT NOT NULL,
    created_at          TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_contradictions_ticker ON contradictions(ticker);

CREATE TABLE IF NOT EXISTS regulatory_events (
    event_id            TEXT PRIMARY KEY,
    ticker              TEXT NOT NULL,
    regulator           TEXT NOT NULL DEFAULT 'UNKNOWN',
    event_type          TEXT NOT NULL,
    event_date          TEXT DEFAULT 'UNKNOWN',
    agreed              TEXT DEFAULT 'UNKNOWN',
    not_agreed          TEXT DEFAULT 'UNKNOWN',
    unresolved          TEXT DEFAULT 'UNKNOWN',
    company_framing     TEXT DEFAULT '',
    regulator_statement TEXT DEFAULT '',
    fact_ids            TEXT DEFAULT '',
    run_id              TEXT NOT NULL,
    created_at          TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS clinical_trials (
    nct_id              TEXT NOT NULL,
    ticker              TEXT NOT NULL,
    title               TEXT DEFAULT '',
    phase               TEXT DEFAULT 'UNKNOWN',
    status              TEXT DEFAULT 'UNKNOWN',
    enrollment          INTEGER,
    randomized          TEXT DEFAULT 'UNKNOWN',
    blinding            TEXT DEFAULT 'UNKNOWN',
    control_arm         TEXT DEFAULT 'UNKNOWN',
    primary_endpoint    TEXT DEFAULT 'UNKNOWN',
    secondary_endpoints TEXT DEFAULT '',
    primary_completion  TEXT DEFAULT 'UNKNOWN',
    sponsor             TEXT DEFAULT 'UNKNOWN',
    fact_ids            TEXT DEFAULT '',
    run_id              TEXT NOT NULL,
    created_at          TEXT NOT NULL,
    PRIMARY KEY (nct_id, run_id)
);

CREATE TABLE IF NOT EXISTS capital_structure (
    run_id                      TEXT NOT NULL,
    ticker                      TEXT NOT NULL,
    as_of                       TEXT DEFAULT 'UNKNOWN',
    basic_shares                REAL,
    fully_diluted_shares        REAL,
    options                     REAL,
    rsus                        REAL,
    public_warrants             REAL,
    private_warrants            REAL,
    prefunded_warrants          REAL,
    preferred                   REAL,
    convertible_debt            REAL,
    atm_capacity                REAL,
    shelf_capacity              REAL,
    cash                        REAL,
    debt                        REAL,
    quarterly_burn              REAL,
    runway_months               REAL,
    going_concern               TEXT DEFAULT 'UNKNOWN',
    reverse_split_history       TEXT DEFAULT 'UNKNOWN',
    listing_compliance          TEXT DEFAULT 'UNKNOWN',
    unknown_fields              TEXT DEFAULT '',
    fact_ids                    TEXT DEFAULT '',
    created_at                  TEXT NOT NULL,
    PRIMARY KEY (run_id, ticker)
);

CREATE TABLE IF NOT EXISTS insider_trades (
    trade_id            TEXT PRIMARY KEY,
    ticker              TEXT NOT NULL,
    insider             TEXT DEFAULT 'UNKNOWN',
    role                TEXT DEFAULT 'UNKNOWN',
    transaction_date    TEXT DEFAULT 'UNKNOWN',
    transaction_code    TEXT DEFAULT 'UNKNOWN',
    shares              REAL,
    price               REAL,
    is_10b5_1           TEXT DEFAULT 'UNKNOWN',
    source_url          TEXT DEFAULT '',
    run_id              TEXT NOT NULL,
    created_at          TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS catalysts (
    event_id            TEXT NOT NULL,
    ticker              TEXT NOT NULL,
    run_id              TEXT NOT NULL,
    horizon             TEXT NOT NULL,
    date_jst            TEXT DEFAULT 'UNKNOWN',
    date_confidence     TEXT DEFAULT 'UNKNOWN',
    event               TEXT NOT NULL,
    expected_outcome    TEXT DEFAULT 'UNKNOWN',
    bull_outcome        TEXT DEFAULT 'UNKNOWN',
    bear_outcome        TEXT DEFAULT 'UNKNOWN',
    market_pricing      TEXT DEFAULT 'UNKNOWN',
    information_source  TEXT DEFAULT 'UNKNOWN',
    fact_ids            TEXT DEFAULT '',
    created_at          TEXT NOT NULL,
    PRIMARY KEY (event_id, run_id)
);

CREATE TABLE IF NOT EXISTS scores (
    run_id              TEXT NOT NULL,
    ticker              TEXT NOT NULL,
    dimension           TEXT NOT NULL,
    score               REAL,
    confidence          REAL,
    rationale           TEXT DEFAULT '',
    capped_by_kill_gate INTEGER NOT NULL DEFAULT 0,
    created_at          TEXT NOT NULL,
    PRIMARY KEY (run_id, ticker, dimension)
);

CREATE TABLE IF NOT EXISTS kill_assessments (
    run_id              TEXT NOT NULL,
    ticker              TEXT NOT NULL,
    category            TEXT NOT NULL,
    level               TEXT NOT NULL,
    rationale           TEXT DEFAULT '',
    evidence_confidence REAL DEFAULT 0.0,
    -- Decision-Grade Evidence Gate (requirement DG2): PROVISIONAL or
    -- CONFIRMED, distinct from `level`. A K5 finding can be PROVISIONAL.
    confirmation        TEXT NOT NULL DEFAULT 'PROVISIONAL',
    created_at          TEXT NOT NULL,
    PRIMARY KEY (run_id, ticker, category)
);

-- Per-run, per-domain outcome of the Evidence Sufficiency Matrix (requirement
-- DG5). Kept apart from research_coverage: that table answers "did we search
-- this domain", this one answers "did what we found settle anything".
CREATE TABLE IF NOT EXISTS evidence_sufficiency (
    run_id                      TEXT NOT NULL,
    ticker                      TEXT NOT NULL,
    domain                      TEXT NOT NULL,
    search_status               TEXT NOT NULL,
    evidence_sufficiency_status TEXT NOT NULL,
    decision_grade_fact_count   INTEGER NOT NULL DEFAULT 0,
    total_fact_count            INTEGER NOT NULL DEFAULT 0,
    reason                      TEXT DEFAULT '',
    created_at                  TEXT NOT NULL,
    PRIMARY KEY (run_id, domain)
);

CREATE TABLE IF NOT EXISTS scenarios (
    run_id                      TEXT NOT NULL,
    ticker                      TEXT NOT NULL,
    name                        TEXT NOT NULL,
    prob_low                    REAL,
    prob_high                   REAL,
    price_low                   REAL,
    price_high                  REAL,
    market_cap                  REAL,
    fully_diluted_market_cap    REAL,
    time_horizon                TEXT DEFAULT 'UNKNOWN',
    required_conditions         TEXT DEFAULT '',
    failure_conditions          TEXT DEFAULT '',
    confidence                  TEXT DEFAULT 'LOW_CONFIDENCE',
    notes                       TEXT DEFAULT '',
    created_at                  TEXT NOT NULL,
    PRIMARY KEY (run_id, ticker, name)
);

CREATE TABLE IF NOT EXISTS thesis_versions (
    ticker              TEXT NOT NULL,
    version             INTEGER NOT NULL,
    run_id              TEXT NOT NULL,
    action              TEXT NOT NULL,
    evidence_confidence REAL NOT NULL,
    headline            TEXT DEFAULT '',
    max_kill_level      TEXT DEFAULT 'K0',
    what_changed        TEXT DEFAULT '',
    why_changed         TEXT DEFAULT '',
    new_facts           TEXT DEFAULT '',
    removed_assumptions TEXT DEFAULT '',
    score_change        TEXT DEFAULT '',
    snapshot            TEXT DEFAULT '',
    -- Decision-Grade Evidence Gate (requirement DG1): COMPLETE, INCOMPLETE or
    -- BLOCKED_PENDING_VERIFICATION. `action` above may be the string "None"
    -- exactly when this is not COMPLETE.
    research_status               TEXT NOT NULL DEFAULT 'COMPLETE',
    blocking_verification_required TEXT DEFAULT '',
    created_at          TEXT NOT NULL,
    PRIMARY KEY (ticker, version)
);

CREATE TABLE IF NOT EXISTS agent_runs (
    run_id              TEXT NOT NULL,
    agent_id            TEXT NOT NULL,
    status              TEXT NOT NULL,
    started_at          TEXT NOT NULL,
    finished_at         TEXT NOT NULL,
    duration_ms         INTEGER NOT NULL DEFAULT 0,
    fact_count          INTEGER NOT NULL DEFAULT 0,
    error_count         INTEGER NOT NULL DEFAULT 0,
    errors              TEXT DEFAULT '',
    inputs_hash         TEXT DEFAULT '',
    PRIMARY KEY (run_id, agent_id)
);

CREATE TABLE IF NOT EXISTS runs (
    run_id              TEXT PRIMARY KEY,
    ticker              TEXT NOT NULL,
    mode                TEXT NOT NULL DEFAULT 'standard',
    status              TEXT NOT NULL DEFAULT 'INCOMPLETE_RESEARCH',
    offline             INTEGER NOT NULL DEFAULT 1,
    used_fixtures       INTEGER NOT NULL DEFAULT 0,
    started_at          TEXT NOT NULL,
    finished_at         TEXT DEFAULT '',
    notes               TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS fetch_log (
    run_id              TEXT NOT NULL,
    url                 TEXT NOT NULL,
    collector           TEXT NOT NULL,
    outcome             TEXT NOT NULL,
    http_status         INTEGER,
    attempts            INTEGER NOT NULL DEFAULT 1,
    detail              TEXT DEFAULT '',
    created_at          TEXT NOT NULL
);

-- ---------------------------------------------------------------------------
-- Resume support (requirement P8). A run records a checkpoint after each stage
-- so an interrupted run restarts from the last successful stage rather than
-- from the beginning.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS run_checkpoints (
    run_id              TEXT NOT NULL,
    stage               TEXT NOT NULL,
    stage_index         INTEGER NOT NULL,
    status              TEXT NOT NULL,
    payload             TEXT DEFAULT '',
    fact_count          INTEGER NOT NULL DEFAULT 0,
    created_at          TEXT NOT NULL,
    PRIMARY KEY (run_id, stage)
);
CREATE INDEX IF NOT EXISTS idx_checkpoints_run ON run_checkpoints(run_id, stage_index);

-- Per-run research coverage, so the completeness gate is auditable after the
-- fact and a resumed run knows which domains it still owes.
CREATE TABLE IF NOT EXISTS research_coverage (
    run_id              TEXT NOT NULL,
    ticker              TEXT NOT NULL,
    domain              TEXT NOT NULL,
    status              TEXT NOT NULL,
    queries_attempted   INTEGER NOT NULL DEFAULT 0,
    queries_executed    INTEGER NOT NULL DEFAULT 0,
    documents_found     INTEGER NOT NULL DEFAULT 0,
    paths               TEXT DEFAULT '',
    detail              TEXT DEFAULT '',
    created_at          TEXT NOT NULL,
    PRIMARY KEY (run_id, domain)
);

-- Every LLM call, for cost accounting and for proving which agents were
-- model-backed in a given run.
CREATE TABLE IF NOT EXISTS llm_calls (
    run_id              TEXT NOT NULL,
    agent_id            TEXT NOT NULL,
    model               TEXT NOT NULL,
    input_tokens        INTEGER NOT NULL DEFAULT 0,
    output_tokens       INTEGER NOT NULL DEFAULT 0,
    cache_read_tokens   INTEGER NOT NULL DEFAULT 0,
    duration_ms         INTEGER NOT NULL DEFAULT 0,
    ok                  INTEGER NOT NULL DEFAULT 1,
    stop_reason         TEXT DEFAULT '',
    error               TEXT DEFAULT '',
    created_at          TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_llm_calls_run ON llm_calls(run_id);

-- Primary-source escalation attempts (requirement P4).
CREATE TABLE IF NOT EXISTS escalations (
    run_id              TEXT NOT NULL,
    fact_id             TEXT NOT NULL,
    reason              TEXT NOT NULL,
    searched            INTEGER NOT NULL DEFAULT 0,
    confirmed           INTEGER NOT NULL DEFAULT 0,
    confirming_url      TEXT DEFAULT '',
    queries             TEXT DEFAULT '',
    note                TEXT DEFAULT '',
    created_at          TEXT NOT NULL,
    PRIMARY KEY (run_id, fact_id)
);

-- ---------------------------------------------------------------------------
-- Search discovery log (requirements M2/M3). Search results are discovery
-- evidence, never facts, and are stored here so they can never accidentally
-- be mistaken for verified evidence. Every query is auditable and tagged with
-- its purpose (BULL/BEAR/NEUTRAL/CONTRADICTION/VERIFICATION); a Bear-purpose
-- hit must never be readable by an agent whose isolation policy grants it only
-- Bull-purpose access, and vice versa (see purposes_for_agent()).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS search_queries (
    query_id            TEXT NOT NULL,
    run_id              TEXT NOT NULL,
    ticker              TEXT NOT NULL,
    agent_id            TEXT NOT NULL,
    query_purpose       TEXT NOT NULL,
    query_text          TEXT NOT NULL,
    origin_fact_id      TEXT,
    created_at          TEXT NOT NULL,
    results_count       INTEGER NOT NULL DEFAULT 0,
    executed            INTEGER NOT NULL DEFAULT 1,
    provider            TEXT DEFAULT 'UNKNOWN',
    rationale           TEXT DEFAULT '',
    outcome             TEXT DEFAULT 'UNKNOWN',
    PRIMARY KEY (query_id)
);
CREATE INDEX IF NOT EXISTS idx_search_queries_run ON search_queries(run_id, query_purpose);

CREATE TABLE IF NOT EXISTS search_hits (
    hit_id              TEXT NOT NULL,
    query_id            TEXT NOT NULL,
    run_id              TEXT NOT NULL,
    ticker              TEXT NOT NULL,
    title               TEXT DEFAULT '',
    url                 TEXT NOT NULL,
    snippet             TEXT DEFAULT '',
    provider            TEXT DEFAULT 'UNKNOWN',
    research_path       TEXT DEFAULT 'NONE',
    query_purpose       TEXT NOT NULL,
    captured_at         TEXT NOT NULL,
    published_date      TEXT DEFAULT 'UNKNOWN',
    tier                TEXT DEFAULT 'UNKNOWN',
    rank                INTEGER NOT NULL DEFAULT 0,
    body_retrieved      INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (hit_id)
);
CREATE INDEX IF NOT EXISTS idx_search_hits_run ON search_hits(run_id, query_purpose);
CREATE INDEX IF NOT EXISTS idx_search_hits_query ON search_hits(query_id);
