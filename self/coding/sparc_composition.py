"""SParC/Spider composition primitives: schema, monotone AST diff, execution.

Everything here is GPU-free and unit-tested offline (tests/test_sparc_composition.py),
following the bfcl_composition.py precedent: the SQL-AST and execution logic is
the part of the pipeline most likely to be silently wrong, and the cheapest to
test.  The GPU scripts under reports/composition_screen/ stay thin.

Plan references are to new_task_candidates/coding/SParC_Spider_CSI_Research_Plan.md:
  §3   monotone-edit definition (additions + single-clause modifications)
  §8   schema serialization (bounded value samples, deterministic)
  §9   the three prompt modes
  §11  guard levels: static / execution / denotational two-view agreement

Test-suite scoring semantics reproduce Zhong et al. 2020 (multiset rows, order
enforced only under ORDER BY, float tolerance, column permutation allowed when
scoring against gold).  The official repo's executor is not imported; its
process-forking harness fights this codebase, and the comparator is small
enough to verify directly -- sparc_data.py --audit-scorer cross-checks it on
gold queries before anything downstream runs.
"""

from __future__ import annotations

import re
import sqlite3
import threading
from collections import Counter
from dataclasses import dataclass, field
from itertools import permutations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import sqlglot
from sqlglot import exp
from sqlglot.errors import OptimizeError, ParseError
from sqlglot.optimizer.qualify import qualify

from self.coding.atomic_data import AtomicExample

DIALECT = "sqlite"


class SqlParseError(ValueError):
    """The query does not parse under the SQLite grammar."""


class SchemaResolutionError(ValueError):
    """The query references tables or columns absent from the schema."""


# ---------------------------------------------------------------------------
# schema
# ---------------------------------------------------------------------------

_TYPE_DISPLAY = {"text": "TEXT", "number": "NUMBER", "time": "TIME",
                 "boolean": "BOOL", "others": "OTHER"}
_TYPE_SQL = {"text": "TEXT", "number": "REAL", "time": "TEXT",
             "boolean": "BOOLEAN", "others": "TEXT"}


@dataclass(frozen=True)
class DbSchema:
    db_id: str
    tables: Dict[str, Dict[str, str]]                  # table -> {column: spider type}
    primary_keys: Dict[str, Tuple[str, ...]]
    foreign_keys: Tuple[Tuple[str, str, str, str], ...]  # (tab, col, ref_tab, ref_col)

    def qualify_mapping(self) -> Dict[str, Dict[str, str]]:
        """Lowercase schema mapping for sqlglot's qualify (sqlite folds case)."""
        return {t.lower(): {c.lower(): _TYPE_SQL[ty] for c, ty in cols.items()}
                for t, cols in self.tables.items()}

    def column_type(self, table: Optional[str], column: str) -> Optional[str]:
        """Spider type of a column; table may be None (search all, skip if ambiguous)."""
        col = column.lower()
        if table:
            cols = self.tables.get(self._table_case(table), {})
            for c, ty in cols.items():
                if c.lower() == col:
                    return ty
            return None
        hits = {ty for cols in self.tables.values()
                for c, ty in cols.items() if c.lower() == col}
        return hits.pop() if len(hits) == 1 else None

    def _table_case(self, name: str) -> str:
        for t in self.tables:
            if t.lower() == name.lower():
                return t
        return name


def load_schemas(tables_json: Iterable[dict]) -> Dict[str, DbSchema]:
    """Parse Spider/SParC tables.json entries (original-case names)."""
    out = {}
    for entry in tables_json:
        table_names = entry["table_names_original"]
        cols = entry["column_names_original"]        # [(table_idx, name)], idx -1 = '*'
        types = entry["column_types"]
        tables: Dict[str, Dict[str, str]] = {t: {} for t in table_names}
        for (t_idx, name), ty in zip(cols, types):
            if t_idx >= 0:
                tables[table_names[t_idx]][name] = ty
        pks: Dict[str, List[str]] = {}
        for c_idx in entry["primary_keys"]:
            t_idx, name = cols[c_idx]
            pks.setdefault(table_names[t_idx], []).append(name)
        fks = []
        for c_idx, ref_idx in entry["foreign_keys"]:
            t_idx, name = cols[c_idx]
            r_idx, r_name = cols[ref_idx]
            fks.append((table_names[t_idx], name, table_names[r_idx], r_name))
        out[entry["db_id"]] = DbSchema(
            db_id=entry["db_id"], tables=tables,
            primary_keys={t: tuple(v) for t, v in pks.items()},
            foreign_keys=tuple(fks))
    return out


def serialize_schema(schema: DbSchema, db_path: Optional[Path] = None, *,
                     value_samples: int = 2, sample_chars: int = 40) -> str:
    """Plan §8: table lines with inline bounded value samples, then FK lines.

    Samples come from SELECT DISTINCT ... ORDER BY ... LIMIT k -- deterministic,
    so `value_samples` itself is the pinned sampling parameter.  Text columns
    only: WHERE literals are the grounding problem (plan §19 Risk 6); numeric
    formats are implied by the type.  value_samples=0 is the §17.8 ablation.
    """
    samples: Dict[Tuple[str, str], List[str]] = {}
    if value_samples > 0 and db_path is not None and Path(db_path).exists():
        try:
            conn = sqlite3.connect(f"file:{db_path}?mode=ro&immutable=1", uri=True)
            conn.text_factory = lambda b: b.decode("utf-8", "backslashreplace")
            for t, cols in schema.tables.items():
                for c, ty in cols.items():
                    if ty != "text":
                        continue
                    try:
                        rows = conn.execute(
                            f'SELECT DISTINCT "{c}" FROM "{t}" WHERE "{c}" IS NOT NULL '
                            f'ORDER BY "{c}" LIMIT {int(value_samples)}').fetchall()
                        vals = [str(r[0])[:sample_chars] for r in rows]
                        if vals:
                            samples[(t, c)] = vals
                    except sqlite3.Error:
                        continue
            conn.close()
        except sqlite3.Error:
            pass

    lines = []
    for t, cols in schema.tables.items():
        parts = []
        for c, ty in cols.items():
            piece = f"{c} {_TYPE_DISPLAY[ty]}"
            if (t, c) in samples:
                piece += " [" + ", ".join(repr(v) for v in samples[(t, c)]) + "]"
            parts.append(piece)
        lines.append(f"{t}({', '.join(parts)})")
    for t, c, rt, rc in schema.foreign_keys:
        lines.append(f"FK: {t}.{c} -> {rt}.{rc}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# parsing, canonicalization, clause atoms
# ---------------------------------------------------------------------------

CATEGORIES = ("select", "from_tables", "join_conds", "where",
              "group_by", "having", "order_by", "limit")


@dataclass(frozen=True)
class ClauseAtoms:
    select: frozenset
    distinct: bool
    from_tables: frozenset
    join_conds: frozenset
    where: frozenset
    group_by: frozenset
    having: frozenset
    order_by: Tuple[str, ...]
    limit: Optional[str]
    structure: str                      # plain | setop | nested | self_join


@dataclass(frozen=True)
class MonotoneVerdict:
    monotone: bool
    reason: Optional[str]
    added: Dict[str, tuple] = field(default_factory=dict)
    removed: Dict[str, tuple] = field(default_factory=dict)


_DQUOTE_RE = re.compile(r'"([^"]*)"')


def normalize_quotes(sql: str) -> str:
    """Rewrite double-quoted spans as single-quoted string literals.

    Spider/SParC gold SQL quotes string values with double quotes
    ('WHERE name = "John Dorian"').  SQLite executes that via its legacy
    fallback, but under the SQL standard those are identifiers, so qualify
    would try to resolve 'john dorian' as a column.  Spider's official
    evaluation applies the same rewrite; identifiers never need double quotes
    in this corpus (original names contain no spaces).
    """
    return _DQUOTE_RE.sub(lambda m: "'" + m.group(1).replace("'", "''") + "'", sql)


def _parse(sql: str) -> exp.Expression:
    try:
        tree = sqlglot.parse_one(normalize_quotes(sql), read=DIALECT)
    except (ParseError, ValueError) as err:
        raise SqlParseError(str(err)) from err
    if tree is None:
        raise SqlParseError("empty parse")
    return tree


def _structure(tree: exp.Expression) -> str:
    if isinstance(tree, (exp.Union, exp.Except, exp.Intersect)):
        return "setop"
    if len(list(tree.find_all(exp.Select))) > 1 or tree.find(exp.Subquery):
        return "nested"
    names = [t.name.lower() for t in tree.find_all(exp.Table)]
    if len(names) != len(set(names)):
        return "self_join"
    return "plain"


def _and_conjuncts(node: Optional[exp.Expression]) -> List[exp.Expression]:
    if node is None:
        return []
    if isinstance(node, exp.And):
        return list(node.flatten())
    return [node]


def _dealias(tree: exp.Expression) -> exp.Expression:
    """Replace alias qualifiers with real table names and drop the aliases.

    Safe only for structure == plain (no self-joins), which the caller checks.
    """
    alias_map = {}
    for t in tree.find_all(exp.Table):
        if t.args.get("alias"):
            alias_map[t.alias.lower()] = t.name
            t.set("alias", None)
    for c in tree.find_all(exp.Column):
        if c.table and c.table.lower() in alias_map:
            c.set("table", exp.to_identifier(alias_map[c.table.lower()]))
    return tree


def _expand_output_aliases(tree: exp.Expression) -> exp.Expression:
    """Rewrite ORDER BY / GROUP BY / HAVING references to select-output aliases.

    qualify() contracts `ORDER BY COUNT(*)` into `ORDER BY "_col_0"`; a later
    select-list edit would then rename the alias and fabricate an ORDER BY
    diff.  Substituting the source expression makes atoms edit-stable.
    """
    out_map = {}
    for e in tree.expressions:
        if isinstance(e, exp.Alias):
            out_map[e.alias.lower()] = e.this
    if not out_map:
        return tree
    for key in ("order", "group", "having"):
        node = tree.args.get(key)
        if node is None:
            continue
        for col in list(node.find_all(exp.Column)):
            if not col.table and col.name.lower() in out_map:
                col.replace(out_map[col.name.lower()].copy())
    return tree


def _sqls(nodes: Iterable[exp.Expression]) -> List[str]:
    return [n.sql(dialect=DIALECT) for n in nodes]


def clause_atoms(sql: str, schema: DbSchema) -> ClauseAtoms:
    """Canonical per-clause atoms.  Raises SqlParseError / SchemaResolutionError."""
    tree = _parse(sql)
    structure = _structure(tree)
    if structure != "plain":
        return ClauseAtoms(frozenset(), False, frozenset(), frozenset(),
                           frozenset(), frozenset(), frozenset(), (), None,
                           structure)
    try:
        tree = qualify(tree, schema=schema.qualify_mapping(), dialect=DIALECT)
    except OptimizeError as err:
        raise SchemaResolutionError(str(err)) from err
    tree = _expand_output_aliases(_dealias(tree))

    select = []
    for e in tree.expressions:
        select.append((e.this if isinstance(e, exp.Alias) else e).sql(dialect=DIALECT))
    joins = tree.args.get("joins") or []
    join_conds = []
    for j in joins:
        if j.args.get("on") is not None:
            join_conds += _sqls(_and_conjuncts(j.args["on"]))
    where = tree.args.get("where")
    group = tree.args.get("group")
    having = tree.args.get("having")
    order = tree.args.get("order")
    limit = tree.args.get("limit")
    return ClauseAtoms(
        select=frozenset(select),
        distinct=bool(tree.args.get("distinct")),
        from_tables=frozenset(t.name.lower() for t in tree.find_all(exp.Table)),
        join_conds=frozenset(join_conds),
        where=frozenset(_sqls(_and_conjuncts(where.this if where else None))),
        group_by=frozenset(_sqls(group.expressions)) if group else frozenset(),
        having=frozenset(_sqls(_and_conjuncts(having.this if having else None))),
        order_by=tuple(
            f"{e.this.sql(dialect=DIALECT)} {'DESC' if e.args.get('desc') else 'ASC'}"
            for e in order.expressions) if order else (),
        limit=limit.expression.sql(dialect=DIALECT) if limit else None,
        structure="plain")


def monotone_edit(prev_sql: Optional[str], next_sql: str,
                  schema: DbSchema) -> MonotoneVerdict:
    """Plan §3: monotone iff the next query only ADDS atoms, except that at
    most one clause category may be rewritten (its removals all fall in that
    single category).  Set operations, subqueries, and self-joins are outside
    the covered composition story entirely.
    """
    try:
        nxt = clause_atoms(next_sql, schema)
    except SqlParseError:
        return MonotoneVerdict(False, "parse_error")
    except SchemaResolutionError:
        return MonotoneVerdict(False, "schema_error")
    if nxt.structure != "plain":
        return MonotoneVerdict(False, f"structure_change:{nxt.structure}")
    if prev_sql is None:
        return MonotoneVerdict(True, None)
    try:
        prv = clause_atoms(prev_sql, schema)
    except (SqlParseError, SchemaResolutionError):
        return MonotoneVerdict(False, "prev_unparseable")
    if prv.structure != "plain":
        return MonotoneVerdict(False, f"structure_change:{prv.structure}")

    added: Dict[str, tuple] = {}
    removed: Dict[str, tuple] = {}
    for cat in ("select", "from_tables", "join_conds", "where",
                "group_by", "having"):
        p, n = getattr(prv, cat), getattr(nxt, cat)
        if n - p:
            added[cat] = tuple(sorted(n - p))
        if p - n:
            removed[cat] = tuple(sorted(p - n))
    if prv.distinct != nxt.distinct:
        key = "added" if nxt.distinct else "removed"
        d = added if key == "added" else removed
        d["select"] = tuple(sorted(d.get("select", ()) + ("DISTINCT",)))
    for cat in ("order_by", "limit"):
        p, n = getattr(prv, cat), getattr(nxt, cat)
        p_tuple = p if cat == "order_by" else ((p,) if p is not None else ())
        n_tuple = n if cat == "order_by" else ((n,) if n is not None else ())
        if p_tuple == n_tuple:
            continue
        if n_tuple:
            added[cat] = tuple(n_tuple)
        if p_tuple:
            removed[cat] = tuple(p_tuple)

    removed_cats = [c for c in CATEGORIES if c in removed]
    if len(removed_cats) <= 1:
        return MonotoneVerdict(True, None, added, removed)
    return MonotoneVerdict(False, "multi_clause_change:" + ",".join(removed_cats),
                           added, removed)


def monotone_sequence(sqls: List[str], schema: DbSchema) -> List[MonotoneVerdict]:
    verdicts = []
    prev = None
    for sql in sqls:
        verdicts.append(monotone_edit(prev, sql, schema))
        prev = sql
    return verdicts


# ---------------------------------------------------------------------------
# static guard (plan §11.1)
# ---------------------------------------------------------------------------

_NUMERIC_RE = re.compile(r"^-?\d+(\.\d+)?$")


def _literals_typed(tree: exp.Expression, schema: DbSchema) -> bool:
    """Lenient literal/column type agreement: flag only the two clear cases.

    Spider stores numbers loosely (ids as text and vice versa), so strictness
    here is measured, not assumed -- this check only fails a NUMBER column
    compared to a non-numeric string, or a TEXT column compared to a bare
    number.  Ambiguous column names are skipped.
    """
    comparisons = (exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE, exp.Like)
    for node in tree.find_all(*comparisons):
        col, lit = None, None
        for side in (node.this, node.expression):
            if isinstance(side, exp.Column):
                col = side
            elif isinstance(side, exp.Literal):
                lit = side
        if col is None or lit is None:
            continue
        ty = schema.column_type(col.table or None, col.name)
        if ty == "number" and lit.is_string and not _NUMERIC_RE.match(lit.name.strip()):
            return False
        if ty == "text" and not lit.is_string:
            return False
    for node in tree.find_all(exp.In):
        col = node.this if isinstance(node.this, exp.Column) else None
        if col is None:
            continue
        ty = schema.column_type(col.table or None, col.name)
        for lit in node.expressions:
            if not isinstance(lit, exp.Literal):
                continue
            if ty == "number" and lit.is_string and not _NUMERIC_RE.match(lit.name.strip()):
                return False
            if ty == "text" and not lit.is_string:
                return False
    return True


def static_check(sql: str, schema: DbSchema,
                 prev_sql: Optional[str] = None) -> Dict[str, Any]:
    """Plan §11.1: parses / schema-valid / typed literals / monotone edit."""
    out = {"parses": False, "schema_valid": False, "literals_typed": False,
           "monotone": False, "reason": None}
    try:
        tree = _parse(sql)
    except SqlParseError as err:
        out["reason"] = f"parse:{err}"
        return out
    out["parses"] = True
    try:
        qualified = qualify(tree.copy(), schema=schema.qualify_mapping(),
                            dialect=DIALECT)
    except OptimizeError as err:
        out["reason"] = f"schema:{err}"
        return out
    out["schema_valid"] = True
    out["literals_typed"] = _literals_typed(qualified, schema)
    if not out["literals_typed"]:
        out["reason"] = "literal_type"
        return out
    verdict = monotone_edit(prev_sql, sql, schema)
    out["monotone"] = verdict.monotone
    if not verdict.monotone:
        out["reason"] = f"monotone:{verdict.reason}"
    return out


# ---------------------------------------------------------------------------
# execution and denotational comparison (plan §11.2-11.3, §16 scoring)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExecResult:
    ok: bool
    rows: Optional[Tuple[tuple, ...]]
    n_columns: int
    error: Optional[str]
    truncated: bool = False


def execute_sql(db_path: Path, sql: str, *, timeout_s: float = 15.0,
                max_rows: int = 5000) -> ExecResult:
    """Read-only execution with an interrupt-based timeout.

    immutable=1 skips locking, which matters on network filesystems; writes
    fail under mode=ro, so a hallucinated INSERT cannot touch the data.
    """
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro&immutable=1", uri=True)
    except sqlite3.Error as err:
        return ExecResult(False, None, 0, f"connect:{err}")
    conn.text_factory = lambda b: b.decode("utf-8", "backslashreplace")
    timer = threading.Timer(timeout_s, conn.interrupt)
    timer.start()
    try:
        cur = conn.execute(sql)
        rows = cur.fetchmany(max_rows + 1)
        n_cols = len(cur.description) if cur.description else 0
        truncated = len(rows) > max_rows
        return ExecResult(True, tuple(tuple(r) for r in rows[:max_rows]),
                          n_cols, None, truncated)
    except sqlite3.Error as err:
        return ExecResult(False, None, 0, str(err))
    finally:
        timer.cancel()
        conn.close()


def _canon_value(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, bool):
        return float(int(v))
    if isinstance(v, (int, float)):
        return float(f"{float(v):.6g}")
    if isinstance(v, bytes):
        return v.decode("utf-8", "backslashreplace").strip()
    return str(v).strip()


def _canon_rows(rows: Iterable[tuple]) -> List[tuple]:
    return [tuple(_canon_value(v) for v in r) for r in rows]


def has_order_by(sql: str) -> bool:
    try:
        return _parse(sql).find(exp.Order) is not None
    except SqlParseError:
        return False


def results_equal(a: ExecResult, b: ExecResult, order_sensitive: bool, *,
                  permute_columns: bool = False, max_permute_cols: int = 6) -> bool:
    if not (a.ok and b.ok) or a.truncated or b.truncated:
        return False
    if a.n_columns != b.n_columns:
        return False
    ra, rb = _canon_rows(a.rows), _canon_rows(b.rows)
    if len(ra) != len(rb):
        return False

    def eq(x: List[tuple], y: List[tuple]) -> bool:
        return x == y if order_sensitive else Counter(x) == Counter(y)

    if eq(ra, rb):
        return True
    n = a.n_columns
    if permute_columns and 1 < n <= max_permute_cols:
        for perm in permutations(range(n)):
            if eq(ra, [tuple(r[i] for i in perm) for r in rb]):
                return True
    return False


def suite_paths(testsuite_root: Path, db_root: Path, db_id: str) -> List[Path]:
    """Distilled multi-instance suite when released, else the original sqlite.

    The EMNLP-2020 release covers the Spider DEV databases (plus the classical
    single-db tasks); train databases fall back to their single original
    instance, and the caller logs n_instances so the guard-strength difference
    stays visible (plan §17.6).
    """
    suite_dir = testsuite_root / db_id
    if suite_dir.is_dir():
        found = sorted(suite_dir.rglob("*.sqlite"))
        if found:
            return found
    return [db_root / db_id / f"{db_id}.sqlite"]


def suite_agree(paths: List[Path], sql_a: str, sql_b: str, *,
                timeout_s: float = 15.0) -> bool:
    """Plan §11.3 two-view guard: identical results on EVERY instance.

    Strict on purpose: no column permutation (both candidates came from the
    same model; column order is part of agreement), and order-sensitive if
    either candidate orders its output.
    """
    order = has_order_by(sql_a) or has_order_by(sql_b)
    for p in paths:
        ra = execute_sql(p, sql_a, timeout_s=timeout_s)
        rb = execute_sql(p, sql_b, timeout_s=timeout_s)
        if not results_equal(ra, rb, order):
            return False
    return True


def suite_correct(paths: List[Path], pred_sql: str, gold_sql: str, *,
                  timeout_s: float = 15.0) -> bool:
    """Test-suite execution accuracy: order matters iff GOLD orders, column
    permutation tolerated (official semantics)."""
    order = has_order_by(gold_sql)
    for p in paths:
        rp = execute_sql(p, pred_sql, timeout_s=timeout_s)
        rg = execute_sql(p, gold_sql, timeout_s=timeout_s)
        if not rg.ok:
            continue                      # gold broken on this instance: skip it
        if not results_equal(rp, rg, order, permute_columns=True):
            return False
    return True


# ---------------------------------------------------------------------------
# result-shape and existence checks (plan §11.2)
# ---------------------------------------------------------------------------

_COUNT_RE = re.compile(r"\bhow many\b|\bcount\b|\bnumber of\b|\btotal number\b")
_EXISTENCE_TOLERANT_RE = re.compile(r"\bif any\b|\bare there any\b|\bis there\b|\bwhether\b")


def result_shape_ok(question: str, res: ExecResult) -> Dict[str, Any]:
    """Column-shape sanity for the sink query against the question form."""
    if not res.ok:
        return {"ok": False, "reason": "exec_error"}
    q = question.lower()
    if _COUNT_RE.search(q):
        if res.n_columns != 1:
            return {"ok": False, "reason": "count_expects_one_column"}
        for row in res.rows or ():
            v = row[0]
            if v is not None and not isinstance(v, (int, float)):
                return {"ok": False, "reason": "count_expects_numeric"}
    elif res.n_columns < 1:
        return {"ok": False, "reason": "no_columns"}
    return {"ok": True, "reason": None}


def presupposes_existence(question: str) -> bool:
    return not _EXISTENCE_TOLERANT_RE.search(question.lower())


# ---------------------------------------------------------------------------
# guard levels (plan §11.4)
# ---------------------------------------------------------------------------

STATIC_KEYS = ("parses", "schema_valid", "literals_typed", "monotone")
EXEC_KEYS = ("exec_ok", "shape_ok", "nonempty_ok")


def guard_levels(step_checks: List[Dict[str, Any]]) -> Dict[str, bool]:
    """Assemble L1-L4 from per-turn check dicts.

    L1  every turn produced extractable SQL (no guard beyond that)
    L2  L1 + static checks at every turn
    L3  L2 + execution, shape, and existence checks at every turn
    L4  L3 + two-view denotational agreement at every turn
    """
    produced = all(c.get("produced") for c in step_checks)
    static = produced and all(
        all(c.get(k) for k in STATIC_KEYS) for c in step_checks)
    execution = static and all(
        all(c.get(k) for k in EXEC_KEYS) for c in step_checks)
    twoview = execution and all(c.get("twoview_agree") for c in step_checks)
    return {"L1": produced, "L2": static, "L3": execution, "L4": twoview}


# ---------------------------------------------------------------------------
# prompts (plan §9) and SQL extraction
# ---------------------------------------------------------------------------

def turn_prompt(schema_text: str, current_sql: Optional[str], question: str,
                template: str = "A") -> str:
    cur = current_sql if current_sql else "(none)"
    if template == "A":
        return ("[MODE: TURN]\n"
                f"Schema:\n{schema_text}\n"
                f"Current SQL: {cur}\n"
                f"Request: {question}\n"
                "Return only the new SQL query.")
    if template == "B":
        return ("[MODE: TURN]\n"
                f"Request: {question}\n"
                f"Current SQL: {cur}\n"
                f"Schema:\n{schema_text}\n"
                "Write the updated SQLite query that satisfies the request. "
                "Output SQL only.")
    raise ValueError(f"unknown template {template!r}")


def seq_prompt(schema_text: str, turns: List[str]) -> str:
    body = "\n".join(f"Turn {i}: {q}" for i, q in enumerate(turns, 1))
    return ("[MODE: SEQ]\n"
            f"Schema:\n{schema_text}\n"
            f"{body}\n"
            f"Return only the SQL for Turn {len(turns)}.")


def single_prompt(schema_text: str, question: str) -> str:
    return ("[MODE: SINGLE]\n"
            f"Schema:\n{schema_text}\n"
            f"Question: {question}\n"
            "Return only the SQL.")


_FENCE_RE = re.compile(r"```(?:sql)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_SELECT_RE = re.compile(r"\bselect\b", re.IGNORECASE)


def extract_sql(prediction: str) -> str:
    """First SELECT statement in the prediction, whitespace-collapsed.

    Tolerates code fences, 'SQL:' prefixes, and trailing prose after a
    semicolon or blank line.  Returns '' when no SELECT is found -- callers
    treat that as 'no SQL produced'.
    """
    text = prediction.strip()
    fence = _FENCE_RE.search(text)
    if fence:
        text = fence.group(1).strip()
    m = _SELECT_RE.search(text)
    if not m:
        return ""
    text = text[m.start():]
    for stop in (";", "\n\n"):
        cut = text.find(stop)
        if cut != -1:
            text = text[:cut]
    return " ".join(text.split())


# ---------------------------------------------------------------------------
# AtomicExample constructor
# ---------------------------------------------------------------------------

def sparc_ex(prompt_text: str, target: str, sid: str, db_id: str,
             k: int) -> AtomicExample:
    return AtomicExample(
        task="sparc", source_id=sid, source_group_id=db_id, split="x",
        messages=({"role": "user", "content": prompt_text},),
        target=target, evaluator={}, metadata={"k": k, "db_id": db_id})
