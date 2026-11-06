"""Guard against Python-style comments inside Cypher query strings.

Runnable with:  python tests/test_cypher_comments.py

Cypher comments are ``//``; a ``#`` inside a query makes Neo4j reject the WHOLE
statement (``Invalid input '#'``). This harness:

1. statically scans every ``.py`` file in the plugin for a ``#`` that survives
   outside quoted Cypher literals inside a Cypher-bearing string (the historical
   bug: ``#`` comment between ``SET r.concept_gen`` and ``SET r.weight``);
2. when a Neo4j instance is reachable, extracts the concept-relation
   ``RELATED_TO`` template from ``graphrag_handler.py`` and executes it against
   a throwaway tenant, asserting no ``SyntaxError``, that the edge exists, and
   that re-running it does not accumulate weight (idempotent ``coalesce``).

Import-safe: the Cat plugin loader imports every ``.py`` under the plugin, so
the entire body lives inside ``if __name__ == "__main__":``.
"""

if __name__ == "__main__":

    import argparse
    import ast
    import io
    import os
    import pathlib
    import re
    import sys
    import tokenize
    import uuid
    from typing import LiteralString, cast

    REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # A Cypher-bearing string has a clause keyword starting a line. This keeps
    # LLM prompt templates (Markdown ``##`` headings, prose) out of the scan.
    CLAUSE = re.compile(
        r"(?m)^\s*(?:OPTIONAL\s+MATCH|MATCH|MERGE|CREATE|RETURN|UNWIND|CALL|"
        r"FOREACH|DETACH\s+DELETE|DELETE|SET|REMOVE|WITH)\b"
    )

    # Single- or double-quoted Cypher literal ('' / "" escapes are intra-line).
    QUOTED = re.compile(r"'(?:[^']|'')*'|\"(?:[^\"]|\"\")*\"")

    def _strip_quoted(line):
        return QUOTED.sub("", line)

    def _iter_cypher_strings(path):
        source = path.read_text(encoding="utf-8")
        try:
            tokens = list(
                tokenize.generate_tokens(io.StringIO(source).readline)
            )
        except (tokenize.TokenError, IndentationError):
            return
        for tok in tokens:
            if tok.type != tokenize.STRING:
                continue
            if CLAUSE.search(tok.string):
                try:
                    text = ast.literal_eval(tok.string)
                except (ValueError, SyntaxError):
                    text = tok.string.strip("'\"")
                yield tok.start[0], text

    def static_audit():
        offenders = []
        for path in sorted(pathlib.Path(REPO_ROOT).rglob("*.py")):
            # Plugin sources only: hidden dirs (.git, a local .venv, tool
            # caches) hold third-party code that may even vanish mid-walk.
            rel_parts = path.relative_to(REPO_ROOT).parts
            if "__pycache__" in rel_parts or any(p.startswith(".") for p in rel_parts):
                continue
            for start_line, text in _iter_cypher_strings(path):
                for offset, line in enumerate(text.splitlines()):
                    if "#" in _strip_quoted(line):
                        offenders.append(
                            f"{path.relative_to(REPO_ROOT)}:"
                            f"{start_line + offset}: {line.strip()}"
                        )
        return offenders

    def _concept_relation_query():
        path = pathlib.Path(REPO_ROOT) / "graphrag_handler.py"
        for start_line, text in _iter_cypher_strings(path):
            if "RELATED_TO" in text and "SET r.weight" in text:
                return start_line, text
        raise SystemExit(
            "ERROR: could not find the concept-relation RELATED_TO template "
            "in graphrag_handler.py"
        )

    def _live_check(uri, user, password):
        from neo4j import GraphDatabase, Query
        from neo4j.exceptions import Neo4jError

        tenant = f"cypher-comment-test-{uuid.uuid4().hex}"
        rel_type = "TEST_RELATION"
        start_line, query = _concept_relation_query()
        params = dict(
            tenant_id=tenant,
            subject="Subject Probe",
            subject_id=f"{tenant}-s",
            s_type="CONCEPT",
            s_gen="testgen",
            s_metadata="{}",
            object="Object Probe",
            object_id=f"{tenant}-o",
            o_type="CONCEPT",
            t_gen="testgen",
            t_metadata="{}",
            rel_type=rel_type,
            rel_text="probe",
            r_gen="testgen",
            source_files=["probe.txt"],
        )
        read_weight = (
            "MATCH (s:Entity {tenant_id: $tenant_id})"
            "-[r:RELATED_TO {type: $rel_type}]->"
            "(t:Entity {tenant_id: $tenant_id}) "
            "RETURN r.weight AS weight, r.concept_gen AS gen"
        )
        driver = GraphDatabase.driver(uri, auth=(user, password))
        try:
            with driver.session() as session:
                try:
                    cypher = Query(cast(LiteralString, query))
                    session.run(cypher, params).consume()
                    cypher = Query(cast(LiteralString, query))
                    session.run(cypher, params).consume()
                    record = session.run(
                        Query(cast(LiteralString, read_weight)),
                        {"tenant_id": tenant, "rel_type": rel_type},
                    ).single()
                except Neo4jError as exc:
                    code = getattr(exc, "code", "") or ""
                    if "SyntaxError" in code or "Invalid input" in str(exc):
                        raise SystemExit(
                            f"FAIL live execution of template at "
                            f"graphrag_handler.py:{start_line}: {code} {exc}"
                        )
                    raise
                if record is None:
                    raise SystemExit(
                        "FAIL: concept-relation query ran but no RELATED_TO "
                        "edge was stored"
                    )
                weight = record["weight"]
                if weight != 1.0:
                    raise SystemExit(
                        "FAIL: weight is not idempotent (expected 1.0 after two "
                        f"runs, got {weight!r})"
                    )
                print(
                    "PASS concept-relation template executes; RELATED_TO edge "
                    f"stored with weight={weight}"
                )
        finally:
            with driver.session() as session:
                session.run(
                    "MATCH (e:Entity {tenant_id: $tenant_id}) DETACH DELETE e",
                    tenant_id=tenant,
                ).consume()
            driver.close()

    def main():
        parser = argparse.ArgumentParser()
        parser.add_argument("--uri", default=os.environ.get("NEO4J_URI", "bolt://localhost:7687"))
        parser.add_argument("--user", default=os.environ.get("NEO4J_USER", "neo4j"))
        parser.add_argument("--password", default=os.environ.get("NEO4J_PASSWORD"))
        args = parser.parse_args()

        offenders = static_audit()
        if offenders:
            print("FAIL: Python '#' comment inside a Cypher string:")
            for line in offenders:
                print(f"  {line}")
            return 1
        print("PASS static audit: no '#' inside any Cypher string")

        if not args.password:
            print(
                "SKIP live execution: no Neo4j password (set NEO4J_PASSWORD or "
                "--password). Static audit above is the fallback check."
            )
            return 0

        from neo4j.exceptions import ServiceUnavailable

        try:
            _live_check(args.uri, args.user, args.password)
        except (ServiceUnavailable, OSError) as exc:
            print(
                f"SKIP live execution: Neo4j not reachable at {args.uri} ({exc}). "
                "Static audit above is the fallback check."
            )
            return 0
        return 0

    sys.exit(main())
