# Synthetic Academic Data Generator

A self-contained Python tool that generates realistic higher education data for university registrar systems. Produces multi-semester enrollment simulations with correlated student attributes, prerequisite-aware registration, anomaly detection, and publication-quality visualizations.


## Architecture

```
synthetic-data-generator/
├── generate_academic_data.py   # Core generator engine
├── visualize_academic_data.py  # 6 matplotlib charts
├── seed_supabase.py            # Supabase uploader
├── schema/
│   └── supabase_schema.sql     # Full Supabase schema with RLS
├── tests/
│   └── test_generate_academic_data.py  # Full test suite
├── sample_output/charts/       # Pre-generated chart PNGs
└── requirements.txt
```

## Data Model

- **12 departments** with typed config (`DepartmentConfig(TypedDict)`) -- each has growth rate + demand weight
- **56 courses** with prerequisite chains (e.g., CS 350 requires CS 202 + MATH 310)
- **Staggered cohort admits** -- 40% start semester 0, rest distributed across later semesters, producing organic enrollment growth (+112% over 8 semesters)
- **Correlated student attributes** -- GPA correlates with enrollment status, grades, financial need

## 7 Registration Anomaly Flags

| Flag | Trigger |
|------|---------|
| `OVER_ENROLLED` | Section exceeds capacity |
| `PREREQUISITE_MISSING` | Student lacks required courses |
| `TIME_CONFLICT` | Day-pattern + time-range overlap detection |
| `CREDIT_OVERLOAD` | Exceeds 15 credits (9 for probation students) |
| `HOLD_FINANCIAL` | Probabilistic for high-need students (15%) |
| `HOLD_ACADEMIC` | Probabilistic for probation students (30%) |
| `MEAN_GPA_DROP` | Semester GPA drops >0.5 below 2-semester rolling average |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Generate data (all formats: JSON, CSV, SQL)
python generate_academic_data.py \
    --students 500 --semesters 8 --seed 42 --output all --verbose

# Generate charts from the output
python visualize_academic_data.py --data-dir ./output

# Run tests
pytest tests/ -v
```

### CLI Options

```
--seed INT          Random seed for reproducibility (default: 42)
--students INT      Number of students (default: 200)
--semesters INT     Number of semesters (default: 6)
--start-year INT    Starting academic year (default: 2023)
--output FORMAT     report | json | csv | supabase | all (default: report)
--output-dir PATH   Output directory (default: ./output)
--verbose, -v       Enable debug logging
```

## Sample Output (seed=42, 500 students, 8 semesters)

```
Students:    500     |  Courses:     56
Sections:    963     |  Enrollments: 11,795
Dropped:     983     |  Flagged:     10,291
Over-Cap:    27      |  Critical:    6
GPA Warnings: 177    |  Growth:     +112%

Flag Distribution:
  prereq_missing:  7,728
  credit_overload: 5,437
  time_conflict:   3,419
  hold_academic:     279
  hold_financial:    245
  mean_gpa_drop:     177
  over_enrolled:     146
```

### Charts

| Dashboard | Enrollment Growth |
|-----------|-------------------|
| ![Dashboard](sample_output/charts/dashboard.png) | ![Growth](sample_output/charts/enrollment_growth.png) |

| Registration Flags | Over-Enrollment Alerts |
|--------------------|----------------------|
| ![Flags](sample_output/charts/registration_flags.png) | ![Over](sample_output/charts/over_enrollment.png) |

| GPA Trend | Department Breakdown |
|-----------|---------------------|
| ![GPA](sample_output/charts/gpa_trend.png) | ![Dept](sample_output/charts/department_breakdown.png) |

## Engineering Patterns

- `(str, Enum)` for JSON-serializable enums
- `TypedDict` for typed configuration dictionaries
- `Random()` instance encapsulation (not global `random.seed()`)
- `@staticmethod` factory methods (`classify_level`, `classify_standing`)
- `@property` computed values on dataclasses
- Pre-built `dict` indexes for O(n) analytics (not O(n^2) list comprehension filtering)
- Input validation guards in `__init__`
- `logging` module with `--verbose` CLI flag
- Deterministic output via seed (same seed = identical results)

## Testing

Pytest suite covering:

- Pure functions (`_days_overlap`, `_times_overlap`, `detect_gpa_decline`, `generate_correlated_grade`)
- Enrollment flag detection (prerequisite, over-enrollment, time conflict, credit overload)
- Student classification (`classify_level`, `classify_standing`)
- Dataclass computed properties and serialization (`to_supabase_row` idempotency)
- Input validation guards
- End-to-end generation (determinism, data growth, export formats)

```bash
pytest tests/ -v
```

## Supabase Integration

```bash
# 1. Create a Supabase project at https://supabase.com/dashboard
# 2. Run the schema in SQL Editor:
#    schema/supabase_schema.sql
# 3. Seed the database:
python seed_supabase.py \
    --url https://YOUR-PROJECT.supabase.co \
    --key YOUR-SERVICE-ROLE-KEY
```

## Author

**Wington Brito** -- wingtonbrito@gmail.com
