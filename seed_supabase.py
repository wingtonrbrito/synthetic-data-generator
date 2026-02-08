#!/usr/bin/env python3
"""
Supabase Seeder
====================================
Uploads generated academic data to a Supabase instance.

Setup:
    1. Create a Supabase project at https://supabase.com/dashboard
    2. Run the schema SQL in the Supabase SQL Editor:
       schema/supabase_schema.sql
    3. Get your project URL and service-role key from Project Settings -> API
    4. Run this script:

Usage:
    python seed_supabase.py \\
        --url https://your-project.supabase.co \\
        --key your-anon-or-service-key

    # Or use environment variables:
    export SUPABASE_URL=https://your-project.supabase.co
    export SUPABASE_KEY=your-service-role-key
    python seed_supabase.py

    # With custom data directory:
    python seed_supabase.py --data-dir ./my_data

Author: Wington Brito
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from supabase import create_client, Client


BATCH_SIZE = 100  # Supabase REST API limit per request


def load_json(path: Path) -> list[dict]:
    """Load JSON file, return list of records."""
    with open(path) as f:
        data = json.load(f)
    return data if isinstance(data, list) else [data]


def upsert_batch(client: Client, table: str, records: list[dict], batch_size: int = BATCH_SIZE):
    """Insert records in batches, using upsert to handle re-runs."""
    total = len(records)
    inserted = 0

    for i in range(0, total, batch_size):
        batch = records[i : i + batch_size]
        try:
            client.table(table).upsert(batch).execute()
            inserted += len(batch)
            print(f"  {table}: {inserted}/{total} records", end="\r")
        except Exception as e:
            print(f"\n  ERROR on {table} batch {i}-{i + len(batch)}: {e}")
            # Try inserting one by one to find the problematic record
            for j, record in enumerate(batch):
                try:
                    client.table(table).upsert([record]).execute()
                    inserted += 1
                except Exception as e2:
                    print(f"  SKIP {table}[{i + j}]: {e2}")

    print(f"  {table}: {inserted}/{total} records ✓")
    return inserted


def seed_database(client: Client, data_dir: Path) -> dict[str, int]:
    """Seed all tables from generated JSON data."""
    results = {}

    # Order matters — foreign keys
    tables = [
        ("courses", "courses.json", transform_courses),
        ("students", "students.json", transform_students),
        ("enrollments", "enrollments.json", transform_enrollments),
    ]

    for table_name, filename, transform_fn in tables:
        path = data_dir / filename
        if not path.exists():
            print(f"  SKIP {table_name}: {filename} not found")
            continue

        records = load_json(path)
        if transform_fn:
            records = transform_fn(records)

        print(f"\n  Seeding {table_name} ({len(records)} records)...")
        results[table_name] = upsert_batch(client, table_name, records)

    return results


def transform_courses(records: list[dict]) -> list[dict]:
    """Transform generated course data to match Supabase schema."""
    rows = []
    for r in records:
        rows.append({
            "id": r["id"],
            "code": r["code"],
            "name": r["title"],
            "department": r["department"],
            "credits": r["credits"],
            "level": r.get("level", 100),
            "max_enrollment": r["capacity"],
            "prerequisites": r.get("prerequisites", []),
            "description": r.get("description", ""),
            "is_active": True,
        })
    return rows


def transform_students(records: list[dict]) -> list[dict]:
    """Transform generated student data to match Supabase schema."""
    rows = []
    for r in records:
        rows.append({
            "id": r["id"],
            "student_id": r["student_id"],
            "first_name": r["first_name"],
            "last_name": r["last_name"],
            "email": r["email"],
            "date_of_birth": r["date_of_birth"],
            "gpa": r["gpa"],
            "credits_completed": r["credits_earned"],
            "enrollment_status": r["enrollment_status"],
            "metadata": json.dumps({
                "major": r["major"],
                "classification": r["classification"],
                "financial_need": r["financial_need"],
                "phone": r.get("phone", ""),
            }),
        })
    return rows


def transform_enrollments(records: list[dict]) -> list[dict]:
    """Transform generated enrollment data to match Supabase schema."""
    rows = []
    for r in records:
        rows.append({
            "id": r["id"],
            "student_id": r["student_id"],
            "course_id": r["course_id"],
            "semester": r["semester_name"],
            "year": r["semester_year"],
            "grade": r.get("grade"),
            "status": (
                "dropped" if r.get("dropped")
                else ("completed" if r.get("grade") else "enrolled")
            ),
        })
    return rows


def clean_tables(client: Client):
    """Truncate tables in reverse FK order for idempotent re-seeding."""
    print("\n  Cleaning existing data...")
    for table in ["enrollments", "students", "courses"]:
        try:
            client.table(table).delete().neq("id", "00000000-0000-0000-0000-000000000000").execute()
            print(f"    {table}: cleared")
        except Exception as e:
            print(f"    {table}: ERROR — {e}")


def verify_data(client: Client):
    """Quick verification of seeded data."""
    print("\n  Verification:")
    for table in ["courses", "students", "enrollments"]:
        try:
            result = client.table(table).select("id", count="exact").limit(1).execute()
            count = result.count if result.count is not None else "?"
            print(f"    {table}: {count} rows")
        except Exception as e:
            print(f"    {table}: ERROR — {e}")


def main():
    parser = argparse.ArgumentParser(description="Seed Supabase with academic data")
    parser.add_argument(
        "--url",
        default=os.environ.get("SUPABASE_URL", ""),
        help="Supabase project URL (or set SUPABASE_URL env var)",
    )
    parser.add_argument(
        "--key",
        default=os.environ.get("SUPABASE_KEY", os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")),
        help="Supabase API key — service_role recommended (or set SUPABASE_KEY env var)",
    )
    parser.add_argument(
        "--data-dir",
        default="./output",
        help="Directory with generated JSON data",
    )
    parser.add_argument("--verify-only", action="store_true", help="Only verify existing data")
    parser.add_argument("--clean", action="store_true", help="Truncate tables before seeding (idempotent re-runs)")
    parser.add_argument("--dry-run", action="store_true", help="Preview record counts without connecting to Supabase")
    args = parser.parse_args()

    # Dry-run: validate data and report counts without connecting
    if args.dry_run:
        data_dir = Path(args.data_dir)
        required = ["courses.json", "students.json", "enrollments.json"]
        print("=" * 60)
        print("  Dry Run — Preview")
        print("=" * 60)
        for filename in required:
            path = data_dir / filename
            if path.exists():
                records = load_json(path)
                print(f"  {filename}: {len(records):,} records")
            else:
                print(f"  {filename}: NOT FOUND")
        print("\n  No data was written.")
        sys.exit(0)

    if not args.url or not args.key:
        print("ERROR: Supabase URL and key are required.")
        print()
        print("  Option 1 — CLI args:")
        print("    python seed_supabase.py \\")
        print("        --url https://your-project.supabase.co \\")
        print("        --key your-service-role-key")
        print()
        print("  Option 2 — env vars:")
        print("    export SUPABASE_URL=https://your-project.supabase.co")
        print("    export SUPABASE_KEY=your-service-role-key")
        print("    python seed_supabase.py")
        print()
        print("  Find these at: Supabase Dashboard -> Project Settings -> API")
        sys.exit(1)

    print("=" * 60)
    print("  Supabase Seeder")
    print("=" * 60)
    print(f"  URL: {args.url}")
    print(f"  Key: {args.key[:12]}...{args.key[-4:]}")
    print(f"  Data: {args.data_dir}")

    # Validate data files exist before connecting (fail fast)
    if not args.verify_only:
        data_dir = Path(args.data_dir)
        if not data_dir.exists():
            print(f"\nERROR: Data directory not found: {data_dir}")
            print("Run the generator first:")
            print("  python generate_academic_data.py --output all")
            sys.exit(1)

        required = ["courses.json", "students.json", "enrollments.json"]
        missing = [f for f in required if not (data_dir / f).exists()]
        if missing:
            print(f"\nERROR: Missing data files in {data_dir}:")
            for f in missing:
                print(f"  - {f}")
            print("\nRun the generator first:")
            print("  python generate_academic_data.py --output all")
            sys.exit(1)

    client = create_client(args.url, args.key)

    if args.verify_only:
        verify_data(client)
    else:
        if args.clean:
            clean_tables(client)

        results = seed_database(client, data_dir)
        verify_data(client)

        print("\n" + "=" * 60)
        total = sum(results.values())
        print(f"  Done! {total:,} records seeded across {len(results)} tables.")
        print("=" * 60)


if __name__ == "__main__":
    main()
