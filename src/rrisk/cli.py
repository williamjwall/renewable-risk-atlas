"""rrisk — renewable site risk.

Everyday use:
  rrisk app                 open the local web app (map, tables, SQL, pipeline)
  rrisk update              bring everything up to date (runs only what changed)
  rrisk status              what's built, what's stale, last errors

Also:
  rrisk run STEP [STEP...]  run specific steps (e.g. hazard_global score)
  rrisk site NAME           print a risk card
  rrisk query "SQL"         query the warehouse (view: site_risk)
  rrisk export              write a single shareable HTML file
  rrisk validate            back-test the hazard model and check known loss sites
  rrisk download [GROUP]    fetch sources: us | global | losses | maps | <name>
"""

from __future__ import annotations

import argparse
import sys

from . import config


def _status() -> None:
    from . import pipeline

    marks = {"ok": "✓ up to date", "stale": "↻ needs update", "missing": "· not built"}
    print(f"{'step':18} {'state':16} {'last run':20} note")
    for r in pipeline.status():
        note = r["error"] or ""
        print(f"{r['name']:18} {marks[r['state']]:16} {(r['last_run'] or '—'):20} {note}")


def _query(sql: str, csv: bool, max_rows: int) -> None:
    import duckdb
    import pandas as pd

    con = duckdb.connect(str(config.WAREHOUSE), read_only=True)
    df = con.sql(sql).df()
    if csv:
        df.to_csv(sys.stdout, index=False)
    else:
        with pd.option_context("display.width", 200, "display.max_columns", 30, "display.max_colwidth", 60):
            print(df.to_string(max_rows=max_rows))


def _site(q: str) -> None:
    import duckdb

    con = duckdb.connect(str(config.WAREHOUSE), read_only=True)
    df = con.execute(
        "select * from site_risk where site_id = ? or name ilike ? order by score desc nulls last limit 5",
        [q, f"%{q}%"],
    ).df()
    if df.empty:
        print("no matching site")
        return
    for _, r in df.iterrows():
        where = ", ".join(str(x) for x in (r["county"], r["state"], r["country_name"]) if isinstance(x, str))
        print(f"\n{r['name']} ({r['site_id']}) — {where}")
        print(f"  {r['techs']}  {r['total_mw']:.1f} MW  first year {r['first_year']}")
        basis = "US full model" if r["score_basis"] == "us_full" else "global perils"
        print(f"  score {r['score']} ({r['tier']}, {basis})   global-comparable score {r['score_global']}")
        print(f"    hazard index x{r['hazard_index']:.2f} vs an average {r['primary_tech']} site")
        print(f"  drivers: {r['drivers']}")
        rows = [("tc64", "hurricane-force winds"), ("tc34", "tropical-storm winds"),
                ("xwind", "damaging wind days"), ("xrain", "extreme rain days"), ("heat", "extreme heat days")]
        if r["country"] == "USA":
            rows = [("rhail2", 'hail>=2" near site (radar)'), ("rhail1", 'hail>=1" near site (radar)'),
                    ("wind65", "gust>=75 mph within 25 mi"), ("tor1", "EF1+ tornado within 25 mi"), *rows]
        for k, lab in rows:
            print(f"    {lab:30} {r[f'{k}_p']:6.1%} /yr   {r[f'{k}_rate']:.2f} per yr")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="rrisk", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True, metavar="command")

    a = sub.add_parser("app", help="open the local web app")
    a.add_argument("--port", type=int)
    a.add_argument("--no-browser", action="store_true")
    u = sub.add_parser("update", help="run every step whose inputs changed")
    u.add_argument("--force", action="store_true", help="re-run all automatic steps")
    sub.add_parser("status", help="show pipeline state")
    r = sub.add_parser("run", help="run named steps (see `rrisk status`)")
    r.add_argument("steps", nargs="+")
    s = sub.add_parser("site", help="print a risk card for a site (name or site_id)")
    s.add_argument("query")
    q = sub.add_parser("query", help="SQL against the warehouse")
    q.add_argument("sql")
    q.add_argument("--csv", action="store_true")
    q.add_argument("--max-rows", type=int, default=60)
    e = sub.add_parser("export", help="write a shareable single-file HTML")
    e.add_argument("--out")
    sub.add_parser("validate", help="back-test hazard; check known losses")
    d = sub.add_parser("download", help="fetch source data")
    d.add_argument("what", nargs="*", help="us | global | losses | maps | source name (default: all)")

    args = p.parse_args(argv)
    if args.cmd == "app":
        from .app.server import serve

        serve(port=args.port, open_browser=not args.no_browser)
    elif args.cmd == "update":
        from . import pipeline

        sys.exit(0 if pipeline.update(force=args.force) else 1)
    elif args.cmd == "status":
        _status()
    elif args.cmd == "run":
        from . import pipeline

        sys.exit(0 if pipeline.update(args.steps) else 1)
    elif args.cmd == "site":
        _site(args.query)
    elif args.cmd == "query":
        _query(args.sql, args.csv, args.max_rows)
    elif args.cmd == "export":
        from .app import export

        export.build(args.out)
    elif args.cmd == "validate":
        from . import validate

        validate.run()
    elif args.cmd == "download":
        from .sources import download

        download.run(args.what or None)


if __name__ == "__main__":
    main()
