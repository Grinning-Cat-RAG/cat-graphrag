#!/usr/bin/env python3
"""
Script per eliminare vettori nulli dal database Neo4j.

Usage:
    # Singolo database
    python cleanup_null_vectors.py --uri <neo4j_uri> --user <user> --password <password> --database <db>

    # Tutti i database (Enterprise)
    python cleanup_null_vectors.py --uri <neo4j_uri> --user <user> --password <password> --all-databases

    # Database di sistema (solo Enterprise)
    python cleanup_null_vectors.py --uri <neo4j_uri> --user <user> --password <password> --system

Example:
    python cleanup_null_vectors.py --uri bolt://localhost:7687 --user neo4j --password password --all-databases
"""

import argparse
import asyncio
import sys

from neo4j import AsyncGraphDatabase


async def list_databases(session) -> list[str]:
    """Lista tutti i database disponibili."""
    result = await session.run("SHOW DATABASES")
    records = await result.data()
    return [r["name"] for r in records if r["name"] not in ["system", "neo4j"] or r["name"] == "neo4j"]


async def cleanup_database(session, database: str, generation: str = "v1") -> dict:
    """Controlla e pulisce un singolo database.

    Only nodes carrying an EXPLICITLY EMPTY or broken vector are deleted
    (``embedding_{gen} = []`` / ``entity_embedding_{gen} = []``). Nodes whose
    versioned embedding property is simply MISSING (``IS NULL``) are legitimate
    chunks/entities stored without a vector (the two-phase ingestion) and are
    preserved. Legacy unversioned properties (``embedding`` / ``e.embedding``)
    are reported but never touched — the boot backfill migrates them.
    """
    doc_prop = f"embedding_{generation}"
    ent_prop = f"entity_embedding_{generation}"
    print(f"\n--- Database: {database} ---")

    # Controlla documenti con embedding esplicitamente vuoto
    doc_result = await session.run(
        "MATCH (d:Document) WHERE d.$prop = [] RETURN count(d) AS count",
        prop=doc_prop,
    )
    doc_count = await doc_result.single()
    doc_null_count = (doc_count["count"] if doc_count else 0) or 0

    # Controlla entità con embedding esplicitamente vuoto
    ent_result = await session.run(
        "MATCH (e:Entity) WHERE e.$prop = [] RETURN count(e) AS count",
        prop=ent_prop,
    )
    ent_count = await ent_result.single()
    ent_null_count = (ent_count["count"] if ent_count else 0) or 0

    # Segnala (senza cancellare) i nodi che hanno SOLO la property legacy
    legacy_doc = await session.run(
        "MATCH (d:Document) WHERE d.embedding IS NOT NULL AND d.$prop IS NULL RETURN count(d) AS count",
        prop=doc_prop,
    )
    legacy_doc_count = ((await legacy_doc.single()) or {}).get("count") or 0
    legacy_ent = await session.run(
        "MATCH (e:Entity) WHERE e.embedding IS NOT NULL AND e.$prop IS NULL RETURN count(e) AS count",
        prop=ent_prop,
    )
    legacy_ent_count = ((await legacy_ent.single()) or {}).get("count") or 0
    if legacy_doc_count or legacy_ent_count:
        print(
            f"  NOTE: {legacy_doc_count} doc / {legacy_ent_count} entity still carry "
            f"only the legacy unversioned embedding property (not touched — the "
            f"boot backfill migrates them to {doc_prop} / {ent_prop})."
        )

    total = doc_null_count + ent_null_count

    if total == 0:
        print(f"  Nessun vettore esplicitamente vuoto trovato ({doc_prop} / {ent_prop}).")
        return {"database": database, "doc_deleted": 0, "ent_deleted": 0, "total": 0}

    print(f"  Documenti con embedding vuoto: {doc_null_count}")
    print(f"  Entità con embedding vuoto: {ent_null_count}")
    print(f"  Totale da eliminare: {total}")

    # Elimina documenti
    if doc_null_count > 0:
        await session.run(
            "MATCH (d:Document) WHERE d.$prop = [] DETACH DELETE d",
            prop=doc_prop,
        )

    # Elimina entità
    if ent_null_count > 0:
        await session.run(
            "MATCH (e:Entity) WHERE e.$prop = [] DETACH DELETE e",
            prop=ent_prop,
        )

    # Verifica
    doc_check = await session.run(
        "MATCH (d:Document) WHERE d.$prop = [] RETURN count(d) AS count",
        prop=doc_prop,
    )
    doc_remaining = ((await doc_check.single()) or {}).get("count") or 0

    ent_check = await session.run(
        "MATCH (e:Entity) WHERE e.$prop = [] RETURN count(e) AS count",
        prop=ent_prop,
    )
    ent_remaining = ((await ent_check.single()) or {}).get("count") or 0

    remaining = doc_remaining + ent_remaining
    if remaining > 0:
        print(f"  ATTENZIONE: {remaining} nodi ancora con embedding vuoto!")
    else:
        print(f"  Pulito: {total} nodi eliminati")

    return {
        "database": database,
        "doc_deleted": doc_null_count,
        "ent_deleted": ent_null_count,
        "total": total
    }


async def cleanup_all_databases(uri: str, user: str, password: str, exclude_system: bool = True, generation: str = "v1") -> list[dict]:
    """Esplora e pulisce tutti i database."""
    driver = AsyncGraphDatabase.driver(uri, auth=(user, password))

    # Prima ottiene la lista dei database usando il sessione di sistema
    async with driver.session(database="system") as sys_session:
        result = await sys_session.run("SHOW DATABASES")
        records = await result.data()
        
        databases = []
        for r in records:
            db_name = r["name"]
            # Escludi database di sistema a meno che non sia esplicitamente richiesto
            if exclude_system and db_name in ("system", "neo4j"):
                continue
            if db_name == "neo4j":
                databases.append("neo4j")
            elif not exclude_system or db_name not in ("system",):
                databases.append(db_name)

    results = []

    for db in databases:
        try:
            async with driver.session(database=db) as session:
                res = await cleanup_database(session, db, generation=generation)
                results.append(res)
        except Exception as e:
            print(f"  Errore sul database {db}: {e}")
            results.append({"database": db, "error": str(e)})

    await driver.close()
    return results


async def cleanup_null_vectors(
    uri: str, 
    user: str, 
    password: str, 
    database: str | None = None,
    all_databases: bool = False,
    generation: str = "v1",
) -> None:
    driver = AsyncGraphDatabase.driver(uri, auth=(user, password))

    if all_databases:
        print("=== Controllo tutti i database ===")
        results = await cleanup_all_databases(uri, user, password, exclude_system=True, generation=generation)
        
        print("\n" + "=" * 50)
        print("=== RIEPILOGO ===")
        print("=" * 50)
        total = 0
        for r in results:
            if "error" in r:
                print(f"  {r['database']}: ERRORE - {r['error']}")
            else:
                print(f"  {r['database']}: {r['total']} nodi eliminati")
                total += r["total"]
        print(f"\nTotale nodi eliminati: {total}")
        
    else:
        db = database or "neo4j"
        async with driver.session(database=db) as session:
            print(f"=== Controllo database: {db} ===\n")
            res = await cleanup_database(session, db, generation=generation)
            
            print("\n" + "=" * 50)
            print(f"=== RIEPILOGO: {res['total']} nodi eliminati ===")

    await driver.close()


def main():
    parser = argparse.ArgumentParser(
        description="Elimina vettori nulli dal database Neo4j"
    )
    parser.add_argument(
        "--uri", 
        default="bolt://localhost:7687",
        help="URI di Neo4j (default: bolt://localhost:7687)"
    )
    parser.add_argument(
        "--user", 
        default="neo4j",
        help="Username Neo4j (default: neo4j)"
    )
    parser.add_argument(
        "--password", 
        required=True,
        help="Password Neo4j"
    )
    parser.add_argument(
        "--database", 
        default="neo4j",
        help="Nome del database (default: neo4j)"
    )
    parser.add_argument(
        "--all-databases",
        action="store_true",
        help="Esplora e pulisci tutti i database disponibili"
    )
    parser.add_argument(
        "--include-system",
        action="store_true",
        help="Includi il database di sistema (solo Enterprise)"
    )
    parser.add_argument(
        "--generation",
        default="v1",
        help="Generazione (era) dello schema versionato: property "
             "embedding_<gen> / entity_embedding_<gen> (default: v1)"
    )
    
    args = parser.parse_args()
    
    if args.all_databases and args.database != "neo4j":
        print("Errore: --all-databases e --database sono mutualmente esclusivi")
        sys.exit(1)

    try:
        asyncio.run(cleanup_null_vectors(
            args.uri, 
            args.user, 
            args.password, 
            database=args.database if not args.all_databases else None,
            all_databases=args.all_databases,
            generation=args.generation,
        ))
    except KeyboardInterrupt:
        print("\nOperazione annullata.")
        sys.exit(1)
    except Exception as e:
        print(f"Errore: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()