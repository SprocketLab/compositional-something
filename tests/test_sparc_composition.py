"""Offline tests for the SParC composition primitives (no GPU, no network)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from self.coding.sparc_composition import (
    ExecResult,
    clause_atoms,
    execute_sql,
    extract_sql,
    guard_levels,
    has_order_by,
    load_schemas,
    monotone_edit,
    monotone_sequence,
    result_shape_ok,
    results_equal,
    serialize_schema,
    static_check,
    suite_agree,
    suite_correct,
    suite_paths,
)

TABLES_ENTRY = {
    "db_id": "flight_test",
    "table_names_original": ["flights", "airlines"],
    "table_names": ["flights", "airlines"],
    "column_names_original": [
        [-1, "*"], [0, "origin"], [0, "dest"], [0, "airline_id"],
        [1, "id"], [1, "name"],
    ],
    "column_names": [
        [-1, "*"], [0, "origin"], [0, "dest"], [0, "airline id"],
        [1, "id"], [1, "name"],
    ],
    "column_types": ["text", "text", "text", "number", "number", "text"],
    "primary_keys": [4],
    "foreign_keys": [[3, 4]],
}

SCHEMA = load_schemas([TABLES_ENTRY])["flight_test"]

ROWS_A = [("LAX", "JFK", 1), ("LAX", "AUS", 1), ("JFK", "AUS", 2)]
ROWS_B = [("LAX", "JFK", 2), ("SFO", "AUS", 1)]


def build_db(path: Path, flight_rows) -> Path:
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE flights (origin text, dest text, airline_id real)")
    conn.execute("CREATE TABLE airlines (id real primary key, name text)")
    conn.executemany("INSERT INTO flights VALUES (?, ?, ?)", flight_rows)
    conn.executemany("INSERT INTO airlines VALUES (?, ?)",
                     [(1, "United"), (2, "Delta")])
    conn.commit()
    conn.close()
    return path


@pytest.fixture()
def db_a(tmp_path):
    return build_db(tmp_path / "a.sqlite", ROWS_A)


@pytest.fixture()
def db_b(tmp_path):
    return build_db(tmp_path / "b.sqlite", ROWS_B)


# --- schema -----------------------------------------------------------------

def test_load_schemas_shapes():
    assert SCHEMA.tables["flights"]["airline_id"] == "number"
    assert SCHEMA.foreign_keys == (("flights", "airline_id", "airlines", "id"),)
    assert SCHEMA.primary_keys == {"airlines": ("id",)}


def test_serialize_schema_with_and_without_samples(db_a):
    with_samples = serialize_schema(SCHEMA, db_a, value_samples=2)
    assert "flights(origin TEXT ['JFK', 'LAX'], dest TEXT ['AUS', 'JFK'], "
    assert "origin TEXT ['JFK', 'LAX']" in with_samples
    assert "airline_id NUMBER" in with_samples          # number cols unsampled
    assert "FK: flights.airline_id -> airlines.id" in with_samples
    bare = serialize_schema(SCHEMA, db_a, value_samples=0)
    assert "[" not in bare.split("FK:")[0]
    assert bare.splitlines()[0] == "flights(origin TEXT, dest TEXT, airline_id NUMBER)"


# --- monotone AST checker ---------------------------------------------------

def test_where_addition_is_monotone():
    v = monotone_edit("SELECT origin FROM flights",
                      "SELECT origin FROM flights WHERE dest = 'JFK'", SCHEMA)
    assert v.monotone and "where" in v.added and not v.removed


def test_single_clause_rewrite_is_monotone():
    v = monotone_edit("SELECT * FROM flights",
                      "SELECT airline_id FROM flights WHERE origin = 'LAX'", SCHEMA)
    assert v.monotone and list(v.removed) == ["select"]


def test_rewrite_plus_drop_is_not_monotone():
    v = monotone_edit(
        "SELECT origin FROM flights WHERE origin = 'LAX' AND dest = 'JFK'",
        "SELECT dest FROM flights WHERE origin = 'LAX'", SCHEMA)
    assert not v.monotone
    assert v.reason.startswith("multi_clause_change")


def test_order_and_limit_addition_is_monotone():
    v = monotone_edit("SELECT origin FROM flights",
                      "SELECT origin FROM flights ORDER BY origin LIMIT 3", SCHEMA)
    assert v.monotone and not v.removed


def test_limit_change_is_single_clause_modification():
    v = monotone_edit("SELECT origin FROM flights ORDER BY origin LIMIT 3",
                      "SELECT origin FROM flights ORDER BY origin LIMIT 1", SCHEMA)
    assert v.monotone and list(v.removed) == ["limit"]


def test_setop_and_subquery_are_structure_changes():
    v = monotone_edit("SELECT origin FROM flights",
                      "SELECT origin FROM flights UNION SELECT dest FROM flights",
                      SCHEMA)
    assert not v.monotone and v.reason == "structure_change:setop"
    v = monotone_edit(
        "SELECT name FROM airlines",
        "SELECT name FROM airlines WHERE id IN (SELECT airline_id FROM flights)",
        SCHEMA)
    assert not v.monotone and v.reason == "structure_change:nested"


def test_turn_one_is_monotone_and_sequence_walks_states():
    verdicts = monotone_sequence(
        ["SELECT * FROM flights",
         "SELECT * FROM flights WHERE origin = 'LAX'",
         "SELECT dest FROM flights UNION SELECT origin FROM flights"], SCHEMA)
    assert [v.monotone for v in verdicts] == [True, True, False]


def test_alias_and_qualification_invariance():
    plain = clause_atoms("SELECT name FROM airlines", SCHEMA)
    aliased = clause_atoms("SELECT T1.name FROM airlines AS T1", SCHEMA)
    assert plain == aliased


def test_order_by_alias_expansion_is_edit_stable():
    prev = ("SELECT airline_id, count(*) FROM flights "
            "GROUP BY airline_id ORDER BY count(*) DESC")
    nxt = ("SELECT airline_id FROM flights "
           "GROUP BY airline_id ORDER BY count(*) DESC LIMIT 1")
    v = monotone_edit(prev, nxt, SCHEMA)
    assert v.monotone and "order_by" not in v.removed


# --- static guard -----------------------------------------------------------

def test_double_quoted_literals_are_strings_not_identifiers():
    # Spider/SParC gold quotes string values with double quotes; they must not
    # be resolved as columns, and must equal the single-quoted form.
    dq = clause_atoms('SELECT origin FROM flights WHERE dest = "JFK"', SCHEMA)
    sq = clause_atoms("SELECT origin FROM flights WHERE dest = 'JFK'", SCHEMA)
    assert dq == sq
    v = monotone_edit("SELECT origin FROM flights",
                      'SELECT origin FROM flights WHERE dest = "O\'Hare"', SCHEMA)
    assert v.monotone


def test_static_check_flags_unknown_column():
    out = static_check("SELECT bogus FROM flights", SCHEMA)
    assert out["parses"] and not out["schema_valid"]
    assert out["reason"].startswith("schema:")


def test_static_check_literal_types():
    ok = static_check("SELECT * FROM flights WHERE airline_id = 2", SCHEMA)
    assert ok["literals_typed"]
    ok = static_check("SELECT * FROM flights WHERE airline_id = '2'", SCHEMA)
    assert ok["literals_typed"]                     # numeric string is fine
    bad = static_check("SELECT * FROM flights WHERE airline_id = 'Delta'", SCHEMA)
    assert not bad["literals_typed"] and bad["reason"] == "literal_type"
    bad = static_check("SELECT * FROM flights WHERE origin = 3", SCHEMA)
    assert not bad["literals_typed"]


def test_static_check_includes_monotone_edge():
    out = static_check("SELECT dest FROM flights",
                       SCHEMA, prev_sql="SELECT origin FROM flights WHERE dest = 'JFK'")
    assert not out["monotone"]


# --- execution --------------------------------------------------------------

def test_execute_readonly_and_basic(db_a):
    res = execute_sql(db_a, "SELECT count(*) FROM flights")
    assert res.ok and res.rows == ((3,),) and res.n_columns == 1
    res = execute_sql(db_a, "INSERT INTO flights VALUES ('X', 'Y', 9)")
    assert not res.ok


def test_execute_timeout_interrupts(db_a):
    res = execute_sql(
        db_a,
        "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM c) "
        "SELECT count(*) FROM c",
        timeout_s=0.3)
    assert not res.ok


def test_execute_truncation_flag(db_a):
    res = execute_sql(
        db_a,
        "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM c LIMIT 100) "
        "SELECT x FROM c",
        max_rows=10)
    assert res.ok and res.truncated and len(res.rows) == 10


def test_execute_tolerates_non_utf8(tmp_path):
    path = tmp_path / "raw.sqlite"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE t (v text)")
    conn.execute("INSERT INTO t VALUES (CAST(X'FF41' AS TEXT))")
    conn.commit()
    conn.close()
    res = execute_sql(path, "SELECT v FROM t")
    assert res.ok and len(res.rows) == 1


# --- denotational comparison ------------------------------------------------

def mk(rows, n_cols) -> ExecResult:
    return ExecResult(True, tuple(rows), n_cols, None)


def test_multiset_vs_ordered_semantics():
    a, b = mk([(1,), (2,)], 1), mk([(2,), (1,)], 1)
    assert results_equal(a, b, order_sensitive=False)
    assert not results_equal(a, b, order_sensitive=True)


def test_float_tolerance_and_bool():
    assert results_equal(mk([(1,)], 1), mk([(1.0,)], 1), False)
    assert results_equal(mk([(True,)], 1), mk([(1,)], 1), False)
    assert not results_equal(mk([(1.0,)], 1), mk([(1.001,)], 1), False)


def test_column_permutation_only_when_scoring():
    a, b = mk([("x", 1)], 2), mk([(1, "x")], 2)
    assert not results_equal(a, b, False)
    assert results_equal(a, b, False, permute_columns=True)


def test_row_multiset_counts_matter():
    assert not results_equal(mk([(1,), (1,)], 1), mk([(1,), (2,)], 1), False)


# --- suites -----------------------------------------------------------------

def test_suite_paths_fallback(tmp_path, db_a):
    ts_root = tmp_path / "testsuite"
    db_root = tmp_path / "database"
    (ts_root / "flight_test").mkdir(parents=True)
    (db_root / "flight_test").mkdir(parents=True)
    build_db(ts_root / "flight_test" / "i1.sqlite", ROWS_A)
    build_db(ts_root / "flight_test" / "i2.sqlite", ROWS_B)
    assert len(suite_paths(ts_root, db_root, "flight_test")) == 2
    assert suite_paths(ts_root, db_root, "other_db") == [
        db_root / "other_db" / "other_db.sqlite"]


def test_suite_agree_requires_every_instance(db_a, db_b):
    paths = [db_a, db_b]
    q = "SELECT origin FROM flights WHERE dest = 'AUS'"
    assert suite_agree(paths, q, q)
    # equal on db_a (both 3 rows) but different on db_b (3 vs 2 rows)
    assert not suite_agree(paths, "SELECT origin FROM flights",
                           "SELECT origin FROM flights WHERE origin != 'SFO'")


def test_suite_correct_order_iff_gold_orders(db_a):
    gold_unordered = "SELECT origin FROM flights"
    pred_reordered = "SELECT origin FROM flights ORDER BY dest"
    assert suite_correct([db_a], pred_reordered, gold_unordered)
    gold_ordered = "SELECT origin FROM flights ORDER BY origin DESC"
    pred_wrong_order = "SELECT origin FROM flights ORDER BY origin ASC"
    assert not suite_correct([db_a], pred_wrong_order, gold_ordered)


def test_suite_correct_tolerates_column_order(db_a):
    assert suite_correct([db_a], "SELECT dest, origin FROM flights",
                         "SELECT origin, dest FROM flights")


# --- shape / guards ---------------------------------------------------------

def test_result_shape_count_questions():
    good = mk([(3,)], 1)
    assert result_shape_ok("How many flights are there?", good)["ok"]
    two_col = mk([("LAX", 3)], 2)
    assert not result_shape_ok("How many flights are there?", two_col)["ok"]
    texty = mk([("LAX",)], 1)
    assert not result_shape_ok("Count the number of flights.", texty)["ok"]
    assert result_shape_ok("Which airline runs the most flights?", texty)["ok"]


def test_guard_levels_assembly():
    passing = {"produced": True, "parses": True, "schema_valid": True,
               "literals_typed": True, "monotone": True, "exec_ok": True,
               "shape_ok": True, "nonempty_ok": True, "twoview_agree": True}
    assert guard_levels([passing, passing]) == {
        "L1": True, "L2": True, "L3": True, "L4": True}
    no_agree = passing | {"twoview_agree": False}
    assert guard_levels([passing, no_agree]) == {
        "L1": True, "L2": True, "L3": True, "L4": False}
    exec_fail = passing | {"exec_ok": False}
    assert guard_levels([exec_fail]) == {
        "L1": True, "L2": True, "L3": False, "L4": False}
    silent = passing | {"produced": False}
    assert guard_levels([silent])["L1"] is False


# --- extraction -------------------------------------------------------------

def test_extract_sql_variants():
    q = "SELECT origin FROM flights WHERE dest = 'JFK'"
    assert extract_sql(q) == q
    assert extract_sql(f"```sql\n{q}\n```") == q
    assert extract_sql(f"SQL: {q};") == q
    assert extract_sql(f"Here is the query:\n{q};\nIt filters by dest.") == q
    assert extract_sql(f"{q}\n\nExplanation: filters by dest.") == q
    assert extract_sql("SELECT origin\nFROM flights") == "SELECT origin FROM flights"
    assert extract_sql("I cannot answer that.") == ""


def test_has_order_by():
    assert has_order_by("SELECT a FROM t ORDER BY a")
    assert not has_order_by("SELECT a FROM t")
    assert not has_order_by("not sql at all")
