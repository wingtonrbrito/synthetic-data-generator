#!/usr/bin/env python3
"""
Synthetic Academic Data Generator
=====================================================
Generates realistic higher education data for demonstration and testing:

- Course catalog with prerequisite chains
- Semester schedules with section assignments (time-conflict aware)
- Student rosters with correlated academic attributes
- Multi-semester enrollment simulation with staggered cohort admits
- Registration anomaly detection (7 flag types)
- GPA trend tracking with early-warning decline detection
- Enrollment growth analysis with registrar over-capacity alerts
- Supabase-ready output (JSON, CSV, SQL INSERT statements)

Usage:
    python generate_academic_data.py
    python generate_academic_data.py --students 300 --semesters 8
    python generate_academic_data.py --output json --output-dir ./data
    python generate_academic_data.py --output supabase
    python generate_academic_data.py --seed 42

Author: Wington Brito
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import statistics
import time
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from enum import Enum
from pathlib import Path
from random import Random
from typing import Any, TypedDict

__all__ = [
    "AcademicDataGenerator",
    "Course",
    "Section",
    "Semester",
    "Student",
    "Enrollment",
    "GPATrend",
    "OverEnrollmentAlert",
    "SemesterSnapshot",
    "Classification",
    "StudentStatus",
    "RegistrationFlag",
    "FinancialNeed",
    "AlertSeverity",
    "generate_correlated_grade",
    "detect_enrollment_flags",
    "detect_gpa_decline",
    "has_time_conflict",
]

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# ENUMS (str, Enum — JSON-serializable without .value)
# ──────────────────────────────────────────────────────────────────────────────

class Classification(str, Enum):
    FRESHMAN = "freshman"
    SOPHOMORE = "sophomore"
    JUNIOR = "junior"
    SENIOR = "senior"


class StudentStatus(str, Enum):
    ACTIVE = "active"
    PROBATION = "probation"
    SUSPENDED = "suspended"
    GRADUATED = "graduated"
    WITHDRAWN = "withdrawn"


class RegistrationFlag(str, Enum):
    OVER_ENROLLED = "over_enrolled"
    PREREQUISITE_MISSING = "prereq_missing"
    TIME_CONFLICT = "time_conflict"
    CREDIT_OVERLOAD = "credit_overload"
    HOLD_FINANCIAL = "hold_financial"
    HOLD_ACADEMIC = "hold_academic"
    MEAN_GPA_DROP = "mean_gpa_drop"


class FinancialNeed(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"


class AlertSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ──────────────────────────────────────────────────────────────────────────────
# TYPED CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

class DepartmentConfig(TypedDict):
    prefix: str
    growth: float   # enrollment growth rate per semester
    demand: float   # base demand weight for student major selection


DEPARTMENTS: dict[str, DepartmentConfig] = {
    "Computer Science":         {"prefix": "CS",   "growth": 0.08, "demand": 1.15},
    "Data Science":             {"prefix": "DS",   "growth": 0.12, "demand": 1.25},
    "Business Administration":  {"prefix": "BUS",  "growth": 0.06, "demand": 1.05},
    "Biology":                  {"prefix": "BIO",  "growth": 0.05, "demand": 0.95},
    "Mathematics":              {"prefix": "MATH", "growth": 0.03, "demand": 0.85},
    "Chemistry":                {"prefix": "CHEM", "growth": 0.02, "demand": 0.75},
    "Engineering":              {"prefix": "ENGR", "growth": 0.07, "demand": 1.00},
    "Psychology":               {"prefix": "PSY",  "growth": 0.04, "demand": 0.90},
    "English":                  {"prefix": "ENG",  "growth": -0.01, "demand": 0.65},
    "Economics":                {"prefix": "ECON", "growth": 0.04, "demand": 0.80},
    "Nursing":                  {"prefix": "NURS", "growth": 0.06, "demand": 1.10},
    "Education":                {"prefix": "EDUC", "growth": 0.01, "demand": 0.70},
}

COURSE_CATALOG: dict[str, list[tuple[str, str, int, list[str]]]] = {
    "Computer Science": [
        ("CS 101", "Intro to Computer Science", 3, []),
        ("CS 201", "Data Structures", 3, ["CS 101"]),
        ("CS 202", "Algorithms", 3, ["CS 201"]),
        ("CS 301", "Database Systems", 3, ["CS 201"]),
        ("CS 310", "Software Engineering", 3, ["CS 202"]),
        ("CS 320", "Computer Networks", 3, ["CS 201"]),
        ("CS 350", "Machine Learning", 3, ["CS 202", "MATH 310"]),
        ("CS 360", "AI Fundamentals", 3, ["CS 202"]),
        ("CS 401", "Senior Capstone", 3, ["CS 310"]),
    ],
    "Data Science": [
        ("DS 101", "Intro to Data Science", 3, []),
        ("DS 201", "Statistical Learning", 3, ["DS 101", "MATH 201"]),
        ("DS 301", "Big Data Analytics", 3, ["DS 201"]),
        ("DS 310", "Data Visualization", 3, ["DS 201"]),
        ("DS 350", "Natural Language Processing", 3, ["DS 201", "CS 202"]),
        ("DS 401", "Deep Learning", 3, ["DS 301", "CS 350"]),
    ],
    "Business Administration": [
        ("BUS 101", "Intro to Business", 3, []),
        ("BUS 201", "Financial Accounting", 3, ["BUS 101"]),
        ("BUS 202", "Managerial Accounting", 3, ["BUS 201"]),
        ("BUS 301", "Marketing", 3, ["BUS 101"]),
        ("BUS 310", "Business Analytics", 3, ["BUS 201", "MATH 101"]),
        ("BUS 401", "Strategic Management", 3, ["BUS 301"]),
    ],
    "Biology": [
        ("BIO 101", "General Biology I", 4, []),
        ("BIO 102", "General Biology II", 4, ["BIO 101"]),
        ("BIO 201", "Genetics", 3, ["BIO 102"]),
        ("BIO 301", "Molecular Biology", 3, ["BIO 201", "CHEM 201"]),
        ("BIO 310", "Ecology", 3, ["BIO 102"]),
        ("BIO 401", "Senior Research", 3, ["BIO 301"]),
    ],
    "Mathematics": [
        ("MATH 101", "College Algebra", 3, []),
        ("MATH 102", "Precalculus", 3, ["MATH 101"]),
        ("MATH 201", "Calculus I", 4, ["MATH 102"]),
        ("MATH 202", "Calculus II", 4, ["MATH 201"]),
        ("MATH 301", "Linear Algebra", 3, ["MATH 202"]),
        ("MATH 310", "Statistics", 3, ["MATH 201"]),
    ],
    "Chemistry": [
        ("CHEM 101", "General Chemistry I", 4, []),
        ("CHEM 102", "General Chemistry II", 4, ["CHEM 101"]),
        ("CHEM 201", "Organic Chemistry", 4, ["CHEM 102"]),
    ],
    "Engineering": [
        ("ENGR 101", "Intro to Engineering", 3, []),
        ("ENGR 201", "Statics", 3, ["ENGR 101", "MATH 201"]),
        ("ENGR 301", "Thermodynamics", 3, ["ENGR 201"]),
        ("ENGR 310", "Circuit Analysis", 3, ["ENGR 201"]),
    ],
    "Psychology": [
        ("PSY 101", "Intro to Psychology", 3, []),
        ("PSY 201", "Developmental Psychology", 3, ["PSY 101"]),
        ("PSY 301", "Abnormal Psychology", 3, ["PSY 201"]),
        ("PSY 310", "Research Methods", 3, ["PSY 201", "MATH 310"]),
    ],
    "English": [
        ("ENG 101", "English Composition", 3, []),
        ("ENG 201", "Technical Writing", 3, ["ENG 101"]),
        ("ENG 301", "American Literature", 3, ["ENG 101"]),
    ],
    "Economics": [
        ("ECON 101", "Microeconomics", 3, []),
        ("ECON 102", "Macroeconomics", 3, ["ECON 101"]),
        ("ECON 301", "Econometrics", 3, ["ECON 102", "MATH 310"]),
    ],
    "Nursing": [
        ("NURS 101", "Fundamentals of Nursing", 4, []),
        ("NURS 201", "Health Assessment", 3, ["NURS 101"]),
        ("NURS 301", "Pharmacology", 3, ["NURS 201", "BIO 201"]),
    ],
    "Education": [
        ("EDUC 101", "Foundations of Education", 3, []),
        ("EDUC 201", "Educational Psychology", 3, ["EDUC 101", "PSY 101"]),
        ("EDUC 301", "Curriculum Design", 3, ["EDUC 201"]),
    ],
}

TIME_SLOTS = [
    ("08:00", "09:15"), ("09:30", "10:45"), ("11:00", "12:15"),
    ("13:00", "14:15"), ("14:30", "15:45"), ("16:00", "17:15"),
    ("18:00", "19:15"), ("19:30", "20:45"),
]

DAYS_PATTERNS = ["MWF", "TR", "MW", "WF", "MTWRF"]

BUILDINGS = [
    "Science Hall", "Liberal Arts", "Engineering Bldg",
    "Business School", "Main Hall", "Health Sciences",
]

INSTRUCTORS = [
    "Dr. Thompson", "Dr. Ramirez", "Prof. Chen", "Dr. Williams",
    "Prof. Nakamura", "Dr. Okafor", "Prof. Santos", "Dr. Mueller",
    "Prof. Eriksson", "Dr. Patel", "Prof. Kim", "Dr. Ali",
    "Dr. Reyes", "Prof. O'Brien", "Dr. Yamamoto", "Prof. Johansson",
]

FIRST_NAMES = [
    "Emma", "Liam", "Olivia", "Noah", "Ava", "James", "Sophia", "Lucas",
    "Isabella", "Mason", "Mia", "Ethan", "Amelia", "Aiden", "Harper",
    "Carlos", "Maria", "Wei", "Aisha", "Raj", "Fatima", "Yuki", "Ahmed",
    "Priya", "Diego", "Mei", "Omar", "Sana", "Jamal", "Leila",
    "Andres", "Keiko", "Hassan", "Zara", "Jin", "Valentina", "Kwame",
    "Ingrid", "Mateo", "Noor", "Pavel", "Aaliyah", "Dmitri", "Esmeralda",
    "Hiroshi", "Camila", "Tariq", "Bianca", "Chen", "Naomi",
]

LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
    "Davis", "Rodriguez", "Martinez", "Chen", "Wang", "Kim", "Patel",
    "Singh", "Nguyen", "Ali", "Lopez", "Lee", "Gonzalez", "Wilson",
    "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin", "White",
    "Thompson", "Robinson", "Yamamoto", "Okafor", "Johansson", "Mueller",
    "Santos", "Petrov", "Nakamura", "Okonkwo", "Eriksson", "Fernandez",
    "O'Brien", "Reyes", "Park", "Tanaka", "Kowalski", "Ivanova",
]

GRADE_POINTS: dict[str, float] = {
    "A+": 4.0, "A": 4.0, "A-": 3.7, "B+": 3.3, "B": 3.0, "B-": 2.7,
    "C+": 2.3, "C": 2.0, "C-": 1.7, "D+": 1.3, "D": 1.0, "F": 0.0,
}

MAX_CREDITS = 18                # Absolute enrollment cap per semester
CREDIT_OVERLOAD_THRESHOLD = 15  # Flag threshold for normal students
PROBATION_CREDIT_CAP = 9        # Flag threshold for probation students
DROP_RATE = 0.08


# ──────────────────────────────────────────────────────────────────────────────
# DATA MODELS (Supabase-aligned dataclasses with serialization)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Course:
    """Course catalog entry."""
    id: str
    code: str
    title: str
    department: str
    credits: int
    capacity: int
    level: int = 100
    prerequisites: list[str] = field(default_factory=list)
    description: str = ""

    @property
    def has_prerequisites(self) -> bool:
        return bool(self.prerequisites)

    def to_supabase_row(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "code": self.code,
            "name": self.title,
            "department": self.department,
            "credits": self.credits,
            "level": self.level,
            "max_enrollment": self.capacity,
            "prerequisites": json.dumps(self.prerequisites),
            "description": self.description,
            "is_active": True,
        }


@dataclass
class Section:
    """A scheduled section of a course for a specific semester."""
    id: str
    course_id: str
    course_code: str
    semester_id: str
    instructor: str
    days: str
    start_time: str
    end_time: str
    room: str
    capacity: int
    enrolled_count: int = 0

    @property
    def is_over_capacity(self) -> bool:
        return self.enrolled_count > self.capacity

    @property
    def overflow_count(self) -> int:
        return max(0, self.enrolled_count - self.capacity)

    @property
    def fill_rate(self) -> float:
        return self.enrolled_count / self.capacity if self.capacity > 0 else 0.0



@dataclass
class Semester:
    """Academic semester with registration window."""
    id: str
    name: str
    term: str  # spring, fall
    year: int
    start_date: str
    end_date: str
    registration_open: str
    registration_close: str


@dataclass
class Student:
    """Student record with correlated academic attributes."""
    id: str
    student_id: str  # stable ID like S23-A4F2B
    first_name: str
    last_name: str
    email: str
    date_of_birth: str
    classification: Classification
    major: str
    gpa: float
    credits_earned: int
    enrollment_status: StudentStatus
    financial_need: FinancialNeed
    admit_semester_idx: int = 0  # which semester index this student was admitted
    phone: str = ""

    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}"

    @property
    def is_active(self) -> bool:
        return self.enrollment_status in (StudentStatus.ACTIVE, StudentStatus.PROBATION)

    @staticmethod
    def classify_level(credits: int) -> Classification:
        """Classify academic level from credit hours."""
        if credits < 30:
            return Classification.FRESHMAN
        elif credits < 60:
            return Classification.SOPHOMORE
        elif credits < 90:
            return Classification.JUNIOR
        return Classification.SENIOR

    @staticmethod
    def classify_standing(gpa: float) -> StudentStatus:
        """Classify enrollment status from GPA."""
        if gpa < 1.5:
            return StudentStatus.SUSPENDED
        elif gpa < 2.0:
            return StudentStatus.PROBATION
        return StudentStatus.ACTIVE

    def to_supabase_row(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "student_id": self.student_id,
            "first_name": self.first_name,
            "last_name": self.last_name,
            "email": self.email,
            "date_of_birth": self.date_of_birth,
            "gpa": self.gpa,
            "credits_completed": self.credits_earned,
            "enrollment_status": self.enrollment_status,
            "metadata": json.dumps({
                "major": self.major,
                "classification": self.classification,
                "financial_need": self.financial_need,
            }),
        }


@dataclass
class Enrollment:
    """Student-section enrollment with anomaly flags."""
    id: str
    student_id: str
    section_id: str
    course_id: str
    semester_id: str
    semester_name: str
    semester_year: int
    course_code: str
    grade: str | None = None
    flags: list[str] = field(default_factory=list)
    registered_at: str = ""
    dropped: bool = False

    @property
    def is_flagged(self) -> bool:
        return bool(self.flags)

    def to_supabase_row(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "student_id": self.student_id,
            "course_id": self.course_id,
            "semester": self.semester_name,
            "year": self.semester_year,
            "grade": self.grade,
            "status": "dropped" if self.dropped else ("completed" if self.grade else "enrolled"),
        }


@dataclass
class GPATrend:
    """Per-student per-semester GPA tracking for early warning."""
    student_id: str
    student_name: str
    semester_id: str
    semester_name: str
    semester_gpa: float
    cumulative_gpa: float
    credits_attempted: int
    credits_earned: int
    flags: list[str] = field(default_factory=list)


@dataclass
class OverEnrollmentAlert:
    """Registrar alert when a section exceeds capacity."""
    course_code: str
    course_name: str
    semester_name: str
    section_id: str
    instructor: str
    max_capacity: int
    actual_enrollment: int
    overflow_count: int
    overflow_pct: float
    severity: AlertSeverity = AlertSeverity.LOW

    def __post_init__(self):
        if self.overflow_pct > 0.25:
            self.severity = AlertSeverity.CRITICAL
        elif self.overflow_pct > 0.15:
            self.severity = AlertSeverity.HIGH
        elif self.overflow_pct > 0.05:
            self.severity = AlertSeverity.MEDIUM
        else:
            self.severity = AlertSeverity.LOW


@dataclass
class SemesterSnapshot:
    """Aggregate enrollment statistics for one semester."""
    semester_name: str
    term: str
    year: int
    total_enrollments: int = 0
    unique_students: int = 0
    unique_sections: int = 0
    total_drops: int = 0
    total_flagged: int = 0
    avg_class_size: float = 0.0
    over_enrolled_sections: int = 0
    total_overflow_seats: int = 0
    mean_credits_per_student: float = 0.0
    mean_gpa: float = 0.0


# ──────────────────────────────────────────────────────────────────────────────
# BUSINESS RULES (pure functions — testable and composable)
# ──────────────────────────────────────────────────────────────────────────────

def generate_correlated_grade(rng: Random, student_gpa: float) -> str:
    """
    Grade distribution correlated with student GPA + noise.
    Note: mutates rng state for sampling.
    """
    if student_gpa >= 3.5:
        return rng.choices(
            ["A+", "A", "A-", "B+", "B", "B-"],
            weights=[5, 30, 25, 20, 15, 5], k=1,
        )[0]
    elif student_gpa >= 2.5:
        return rng.choices(
            ["A", "A-", "B+", "B", "B-", "C+", "C"],
            weights=[8, 12, 15, 20, 15, 15, 15], k=1,
        )[0]
    elif student_gpa >= 1.5:
        return rng.choices(
            ["B-", "C+", "C", "C-", "D+", "D", "F"],
            weights=[10, 12, 20, 18, 15, 15, 10], k=1,
        )[0]
    else:
        return rng.choices(
            ["C-", "D+", "D", "F"],
            weights=[10, 20, 30, 40], k=1,
        )[0]


def has_time_conflict(
    section: Section,
    enrolled_sections: list[Section],
) -> bool:
    """Check if a section's time slot overlaps with already-enrolled sections."""
    for existing in enrolled_sections:
        # Same days pattern and overlapping time
        if _days_overlap(section.days, existing.days):
            if _times_overlap(
                section.start_time, section.end_time,
                existing.start_time, existing.end_time,
            ):
                return True
    return False


def _days_overlap(days_a: str, days_b: str) -> bool:
    """Check if two day patterns share any day."""
    set_a = set(days_a)
    set_b = set(days_b)
    return bool(set_a & set_b)


def _times_overlap(
    start_a: str, end_a: str,
    start_b: str, end_b: str,
) -> bool:
    """Check if two time ranges overlap."""
    return start_a < end_b and start_b < end_a


def detect_enrollment_flags(
    student: Student,
    course: Course,
    section: Section,
    completed_courses: set[str],
    total_credits: int,
    enrolled_sections: list[Section],
    rng: Random,
) -> list[str]:
    """
    Detect registration anomalies for a single enrollment.
    Note: uses rng for probabilistic hold flags.
    """
    flags: list[str] = []

    # Prerequisite check
    missing = [p for p in course.prerequisites if p not in completed_courses]
    if missing:
        flags.append(RegistrationFlag.PREREQUISITE_MISSING)

    # Over-enrollment (check post-increment count)
    if section.enrolled_count > section.capacity:
        flags.append(RegistrationFlag.OVER_ENROLLED)

    # Time conflict
    if has_time_conflict(section, enrolled_sections):
        flags.append(RegistrationFlag.TIME_CONFLICT)

    # Credit overload
    cap = PROBATION_CREDIT_CAP if student.enrollment_status == StudentStatus.PROBATION else CREDIT_OVERLOAD_THRESHOLD
    if total_credits > cap:
        flags.append(RegistrationFlag.CREDIT_OVERLOAD)

    # Academic hold (probation students, 30% chance)
    if student.enrollment_status == StudentStatus.PROBATION and rng.random() < 0.30:
        flags.append(RegistrationFlag.HOLD_ACADEMIC)

    # Financial hold (high-need students, 15% chance)
    if student.financial_need == FinancialNeed.HIGH and rng.random() < 0.15:
        flags.append(RegistrationFlag.HOLD_FINANCIAL)

    return flags


def detect_gpa_decline(
    current_gpa: float,
    gpa_history: list[float],
    threshold: float = 0.5,
) -> bool:
    """Flag if semester GPA drops > threshold below recent 2-semester average."""
    if len(gpa_history) < 2:
        return False
    recent_avg = sum(gpa_history[-2:]) / 2
    return current_gpa < recent_avg - threshold


# ──────────────────────────────────────────────────────────────────────────────
# GENERATOR ENGINE
# ──────────────────────────────────────────────────────────────────────────────

class AcademicDataGenerator:
    """
    Multi-semester enrollment simulator with:
    - Staggered cohort admits (enrollment grows organically each semester)
    - Correlated student attributes (GPA <-> status <-> grades)
    - Department-specific enrollment growth trends
    - 7 types of registration anomaly flags (all implemented)
    - GPA trend tracking with decline early-warning
    - Semester-over-semester trend analysis
    - Supabase-aligned output
    """

    def __init__(
        self,
        num_students: int = 200,
        num_semesters: int = 6,
        start_year: int = 2023,
        seed: int = 42,
    ):
        if num_students < 1:
            raise ValueError(f"num_students must be >= 1, got {num_students}")
        if num_semesters < 1:
            raise ValueError(f"num_semesters must be >= 1, got {num_semesters}")
        if start_year < 2000 or start_year > 2100:
            raise ValueError(f"start_year must be 2000-2100, got {start_year}")

        self.rng = Random(seed)
        self.num_students = num_students
        self.num_semesters = num_semesters
        self.start_year = start_year
        self.seed = seed

        # Generated data
        self.courses: list[Course] = []
        self.semesters: list[Semester] = []
        self.students: list[Student] = []
        self.sections: list[Section] = []
        self.enrollments: list[Enrollment] = []
        self.gpa_trends: list[GPATrend] = []
        self.alerts: list[OverEnrollmentAlert] = []
        self.snapshots: list[SemesterSnapshot] = []

        # Internal tracking
        self._course_lookup: dict[str, Course] = {}
        self._completed: dict[str, set[str]] = defaultdict(set)
        self._gpa_history: dict[str, list[float]] = defaultdict(list)
        self._generation_time_ms: float = 0

        # Pre-built indexes (populated during generate)
        self._enrollments_by_semester: dict[str, list[Enrollment]] = defaultdict(list)
        self._sections_by_semester: dict[str, list[Section]] = defaultdict(list)
        self._trends_by_semester: dict[str, list[GPATrend]] = defaultdict(list)

    # ── Public API ────────────────────────────────────────────────────────

    def generate(self) -> dict[str, Any]:
        """Run the full generation pipeline. Returns summary dict.

        Safe to call multiple times — internal state is reset on each run.
        """
        # Reset state for idempotent re-runs
        self.sections = []
        self.enrollments = []
        self.gpa_trends = []
        self.alerts = []
        self.snapshots = []
        self._enrollments_by_semester = defaultdict(list)
        self._sections_by_semester = defaultdict(list)
        self._trends_by_semester = defaultdict(list)
        self._completed = defaultdict(set)
        self._gpa_history = defaultdict(list)

        start = time.perf_counter()

        self.courses = self._build_courses()
        self._course_lookup = {c.code: c for c in self.courses}
        self.semesters = self._build_semesters()
        self.students = self._build_students()

        logger.info(
            "Built base data: %d courses, %d semesters, %d students",
            len(self.courses), len(self.semesters), len(self.students),
        )

        for sem_idx, semester in enumerate(self.semesters):
            sem_sections = self._create_sections(semester, sem_idx)
            self.sections.extend(sem_sections)
            self._sections_by_semester[semester.id] = sem_sections

            sem_enrollments, sem_trends = self._simulate_semester(
                semester, sem_sections, sem_idx,
            )
            self.enrollments.extend(sem_enrollments)
            self._enrollments_by_semester[semester.id] = sem_enrollments
            self.gpa_trends.extend(sem_trends)
            self._trends_by_semester[semester.id] = sem_trends

            logger.info(
                "%s: %d enrollments, %d flags",
                semester.name, len(sem_enrollments),
                sum(len(e.flags) for e in sem_enrollments),
            )

        self._compute_snapshots()
        self._detect_over_enrollment()

        self._generation_time_ms = (time.perf_counter() - start) * 1000
        return self.summary()

    def summary(self) -> dict[str, Any]:
        """Full generation summary with trend analysis."""
        trends = self._compute_trends()
        return {
            "seed": self.seed,
            "generation_time_ms": round(self._generation_time_ms, 1),
            "total_students": len(self.students),
            "total_courses": len(self.courses),
            "total_sections": len(self.sections),
            "total_enrollments": len(self.enrollments),
            "semesters_generated": len(self.semesters),
            "total_drops": sum(1 for e in self.enrollments if e.dropped),
            "total_flagged_enrollments": sum(1 for e in self.enrollments if e.is_flagged),
            "over_enrollment_alerts": len(self.alerts),
            "critical_alerts": sum(1 for a in self.alerts if a.severity == AlertSeverity.CRITICAL),
            "gpa_decline_alerts": sum(
                1 for t in self.gpa_trends
                if RegistrationFlag.MEAN_GPA_DROP in t.flags
            ),
            "snapshots": [asdict(s) for s in self.snapshots],
            "trends": trends,
        }

    # ── Builders ──────────────────────────────────────────────────────────

    def _build_courses(self) -> list[Course]:
        """Build course catalog from templates."""
        courses = []
        for dept, templates in COURSE_CATALOG.items():
            for code, title, credits, prereqs in templates:
                digits = "".join(c for c in code.split()[-1] if c.isdigit())
                level = (int(digits) // 100) * 100 if digits else 100
                courses.append(Course(
                    id=str(uuid.uuid4()),
                    code=code,
                    title=title,
                    department=dept,
                    credits=credits,
                    capacity=self.rng.choice([25, 30, 35, 40, 50, 60]),
                    level=level,
                    prerequisites=prereqs,
                    description=f"{title} — {dept}, {credits} credit hours.",
                ))
        return courses

    def _build_semesters(self) -> list[Semester]:
        """Build chronological semester list with registration windows."""
        semesters = []
        year = self.start_year
        terms = ["spring", "fall"]

        for i in range(self.num_semesters):
            term = terms[i % 2]
            if term == "spring":
                start = date(year, 1, 13)
                end = date(year, 5, 10)
                reg_open = date(year - 1, 11, 1)
                reg_close = date(year, 1, 10)
            else:
                start = date(year, 8, 25)
                end = date(year, 12, 15)
                reg_open = date(year, 4, 1)
                reg_close = date(year, 8, 22)

            semesters.append(Semester(
                id=str(uuid.uuid4()),
                name=f"{term.title()} {year}",
                term=term,
                year=year,
                start_date=start.isoformat(),
                end_date=end.isoformat(),
                registration_open=reg_open.isoformat(),
                registration_close=reg_close.isoformat(),
            ))
            if term == "fall":
                year += 1

        return semesters

    def _build_students(self) -> list[Student]:
        """
        Generate students with correlated attributes and staggered admits.
        ~40% start in semester 0, remainder distributed across later semesters.
        """
        students = []
        dept_names = list(DEPARTMENTS.keys())
        dept_weights = [DEPARTMENTS[d]["demand"] for d in dept_names]

        for i in range(self.num_students):
            major = self.rng.choices(dept_names, weights=dept_weights, k=1)[0]

            # Stagger admits: 40% in sem 0, rest spread across semesters
            if self.rng.random() < 0.40:
                admit_idx = 0
            else:
                admit_idx = self.rng.randint(0, self.num_semesters - 1)

            gpa = round(max(0.0, min(4.0, self.rng.gauss(2.95, 0.65))), 2)
            classification = Student.classify_level(self.rng.randint(0, 120))

            # Status correlated with GPA
            base_status = Student.classify_standing(gpa)
            if base_status == StudentStatus.SUSPENDED:
                status = self.rng.choices(
                    [StudentStatus.SUSPENDED, StudentStatus.PROBATION, StudentStatus.WITHDRAWN],
                    weights=[40, 40, 20],
                )[0]
            elif base_status == StudentStatus.PROBATION:
                status = self.rng.choices(
                    [StudentStatus.PROBATION, StudentStatus.ACTIVE, StudentStatus.WITHDRAWN],
                    weights=[60, 30, 10],
                )[0]
            else:
                status = self.rng.choices(
                    [StudentStatus.ACTIVE, StudentStatus.PROBATION],
                    weights=[92, 8],
                )[0]

            financial = self.rng.choices(
                list(FinancialNeed),
                weights=[15, 35, 30, 20],
            )[0]

            credit_ranges = {
                Classification.FRESHMAN: (0, 29),
                Classification.SOPHOMORE: (30, 59),
                Classification.JUNIOR: (60, 89),
                Classification.SENIOR: (90, 130),
            }
            cr_min, cr_max = credit_ranges[classification]

            age = self.rng.randint(18, 25)
            dob = date.today() - timedelta(days=age * 365 + self.rng.randint(0, 364))

            first = self.rng.choice(FIRST_NAMES)
            last = self.rng.choice(LAST_NAMES)

            # Stable student_id generated once
            sid = f"S{dob.year % 100:02d}-{uuid.uuid4().hex[:5].upper()}"

            students.append(Student(
                id=str(uuid.uuid4()),
                student_id=sid,
                first_name=first,
                last_name=last,
                email=f"{first.lower()}.{last.lower()}{self.rng.randint(1, 99)}@university.edu",
                date_of_birth=dob.isoformat(),
                classification=classification,
                major=major,
                gpa=gpa,
                credits_earned=self.rng.randint(cr_min, cr_max),
                enrollment_status=status,
                financial_need=financial,
                admit_semester_idx=admit_idx,
                phone=f"({self.rng.randint(200, 999)}) {self.rng.randint(200, 999)}-{self.rng.randint(1000, 9999)}",
            ))

        return students

    # ── Section Creation ──────────────────────────────────────────────────

    def _create_sections(self, semester: Semester, sem_idx: int) -> list[Section]:
        """Create course sections with growth-aware section counts."""
        sections = []

        for course in self.courses:
            dept = course.department
            growth = DEPARTMENTS[dept]["growth"]

            # More sections for higher-demand departments in later semesters
            demand_factor = 1.0 + growth * sem_idx
            base_sections = self.rng.choices([1, 2, 3], weights=[40, 40, 20])[0]
            num_sections = min(4, max(1, round(base_sections * demand_factor)))

            for _ in range(num_sections):
                slot = self.rng.choice(TIME_SLOTS)
                sections.append(Section(
                    id=str(uuid.uuid4()),
                    course_id=course.id,
                    course_code=course.code,
                    semester_id=semester.id,
                    instructor=self.rng.choice(INSTRUCTORS),
                    days=self.rng.choice(DAYS_PATTERNS),
                    start_time=slot[0],
                    end_time=slot[1],
                    room=f"{self.rng.choice(BUILDINGS)} {self.rng.randint(100, 400)}",
                    capacity=course.capacity,
                ))

        return sections

    # ── Enrollment Simulation ─────────────────────────────────────────────

    def _simulate_semester(
        self,
        semester: Semester,
        sections: list[Section],
        sem_idx: int,
    ) -> tuple[list[Enrollment], list[GPATrend]]:
        """Simulate one semester of enrollment with flag detection."""
        section_by_course: dict[str, list[Section]] = defaultdict(list)
        for sec in sections:
            section_by_course[sec.course_code].append(sec)

        enrollments: list[Enrollment] = []
        trends: list[GPATrend] = []

        # Only students admitted by this semester
        eligible = [
            s for s in self.students
            if s.admit_semester_idx <= sem_idx
            and s.enrollment_status not in (
                StudentStatus.SUSPENDED, StudentStatus.GRADUATED, StudentStatus.WITHDRAWN,
            )
        ]

        for student in eligible:
            # Course load: probation students take fewer
            if student.enrollment_status == StudentStatus.PROBATION:
                target = self.rng.randint(2, 3)
            else:
                target = self.rng.randint(3, 6)

            codes = list(section_by_course.keys())
            self.rng.shuffle(codes)

            picked: list[tuple[Section, Course]] = []
            picked_sections: list[Section] = []
            total_credits = 0

            for code in codes:
                if len(picked) >= target:
                    break
                course = self._course_lookup.get(code)
                if not course:
                    continue
                if total_credits + course.credits > MAX_CREDITS:
                    continue
                sec = self.rng.choice(section_by_course[code])
                picked.append((sec, course))
                picked_sections.append(sec)
                total_credits += course.credits

            term_enrollments: list[Enrollment] = []

            for sec, course in picked:
                sec.enrolled_count += 1

                flags = detect_enrollment_flags(
                    student=student,
                    course=course,
                    section=sec,
                    completed_courses=self._completed.get(student.id, set()),
                    total_credits=total_credits,
                    enrolled_sections=[
                        s for s in picked_sections if s.id != sec.id
                    ],
                    rng=self.rng,
                )

                grade = generate_correlated_grade(self.rng, student.gpa)
                dropped = self.rng.random() < DROP_RATE

                reg_date = (
                    datetime.fromisoformat(semester.registration_open)
                    + timedelta(days=self.rng.randint(0, 30))
                )

                enrollment = Enrollment(
                    id=str(uuid.uuid4()),
                    student_id=student.id,
                    section_id=sec.id,
                    course_id=sec.course_id,
                    semester_id=semester.id,
                    semester_name=semester.name,
                    semester_year=semester.year,
                    course_code=sec.course_code,
                    grade=None if dropped else grade,
                    flags=flags,
                    registered_at=reg_date.isoformat(),
                    dropped=dropped,
                )
                enrollments.append(enrollment)
                term_enrollments.append(enrollment)

                # Track completed courses
                if not dropped and grade not in ("F", None):
                    self._completed[student.id].add(sec.course_code)

            # GPA tracking
            graded = [
                GRADE_POINTS.get(e.grade, 0.0)
                for e in term_enrollments
                if e.grade and not e.dropped
            ]
            if not graded:
                continue

            sem_gpa = round(sum(graded) / len(graded), 2)

            # GPA decline detection
            gpa_flags: list[str] = []
            if detect_gpa_decline(sem_gpa, self._gpa_history[student.id]):
                gpa_flags.append(RegistrationFlag.MEAN_GPA_DROP)

            self._gpa_history[student.id].append(sem_gpa)
            all_gpas = self._gpa_history[student.id]
            cumulative = round(sum(all_gpas) / len(all_gpas), 2)

            trends.append(GPATrend(
                student_id=student.id,
                student_name=student.full_name,
                semester_id=semester.id,
                semester_name=semester.name,
                semester_gpa=sem_gpa,
                cumulative_gpa=cumulative,
                credits_attempted=total_credits,
                credits_earned=total_credits if sem_gpa >= 1.0 else 0,
                flags=gpa_flags,
            ))

        return enrollments, trends

    # ── Analytics (uses pre-built indexes) ────────────────────────────────

    def _compute_snapshots(self):
        """Compute aggregate statistics per semester using indexed data."""
        for semester in self.semesters:
            sem_enrollments = self._enrollments_by_semester[semester.id]
            sem_sections = self._sections_by_semester[semester.id]
            sem_trends = self._trends_by_semester[semester.id]

            if not sem_enrollments:
                continue

            unique_students = len({e.student_id for e in sem_enrollments})

            # Class sizes from section counts
            active_sections = [s for s in sem_sections if s.enrolled_count > 0]
            avg_size = (
                statistics.mean(s.enrolled_count for s in active_sections)
                if active_sections else 0
            )

            # Credits per student
            student_credits: dict[str, int] = defaultdict(int)
            for e in sem_enrollments:
                if not e.dropped:
                    c = self._course_lookup.get(e.course_code)
                    if c:
                        student_credits[e.student_id] += c.credits
            mean_credits = statistics.mean(student_credits.values()) if student_credits else 0

            mean_gpa = (
                statistics.mean(t.semester_gpa for t in sem_trends)
                if sem_trends else 0
            )

            over_enrolled = sum(1 for s in sem_sections if s.is_over_capacity)
            overflow_seats = sum(s.overflow_count for s in sem_sections)

            self.snapshots.append(SemesterSnapshot(
                semester_name=semester.name,
                term=semester.term,
                year=semester.year,
                total_enrollments=len(sem_enrollments),
                unique_students=unique_students,
                unique_sections=len(sem_sections),
                total_drops=sum(1 for e in sem_enrollments if e.dropped),
                total_flagged=sum(1 for e in sem_enrollments if e.is_flagged),
                avg_class_size=round(avg_size, 1),
                over_enrolled_sections=over_enrolled,
                total_overflow_seats=overflow_seats,
                mean_credits_per_student=round(mean_credits, 1),
                mean_gpa=round(mean_gpa, 2),
            ))

    def _detect_over_enrollment(self):
        """Build registrar alerts for sections over capacity."""
        semester_lookup = {s.id: s for s in self.semesters}
        for section in self.sections:
            if not section.is_over_capacity:
                continue
            course = self._course_lookup.get(section.course_code)
            semester = semester_lookup.get(section.semester_id)
            if not course or not semester:
                continue

            self.alerts.append(OverEnrollmentAlert(
                course_code=section.course_code,
                course_name=course.title,
                semester_name=semester.name,
                section_id=section.id,
                instructor=section.instructor,
                max_capacity=section.capacity,
                actual_enrollment=section.enrolled_count,
                overflow_count=section.overflow_count,
                overflow_pct=round(section.overflow_count / section.capacity, 3),
            ))

    def _compute_trends(self) -> dict[str, Any]:
        """Semester-over-semester enrollment growth analysis."""
        if len(self.snapshots) < 2:
            return {"message": "Need 2+ semesters for trend analysis"}

        totals = [s.total_enrollments for s in self.snapshots]
        students = [s.unique_students for s in self.snapshots]

        enrollment_changes = [
            round((totals[i] - totals[i - 1]) / max(totals[i - 1], 1) * 100, 1)
            for i in range(1, len(totals))
        ]

        student_changes = [
            round((students[i] - students[i - 1]) / max(students[i - 1], 1) * 100, 1)
            for i in range(1, len(students))
        ]

        return {
            "mean_enrollment_increase_pct": (
                round(statistics.mean(enrollment_changes), 2)
                if enrollment_changes else 0
            ),
            "median_enrollment_increase_pct": (
                round(statistics.median(enrollment_changes), 2)
                if enrollment_changes else 0
            ),
            "mean_student_growth_pct": (
                round(statistics.mean(student_changes), 2)
                if student_changes else 0
            ),
            "overall_growth_pct": (
                round((totals[-1] - totals[0]) / max(totals[0], 1) * 100, 1)
            ),
            "gpa_trend": [
                {"semester": s.semester_name, "mean_gpa": s.mean_gpa}
                for s in self.snapshots
            ],
            "enrollment_changes": [
                {
                    "from": self.snapshots[i].semester_name,
                    "to": self.snapshots[i + 1].semester_name,
                    "enrollment_change_pct": enrollment_changes[i],
                    "student_change_pct": student_changes[i],
                }
                for i in range(len(enrollment_changes))
            ],
        }

    # ── Export ────────────────────────────────────────────────────────────

    def to_json(self, output_dir: str) -> dict[str, str]:
        """Export as JSON files (Supabase-importable)."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        files = {}

        datasets: dict[str, list] = {
            "courses": self.courses,
            "semesters": self.semesters,
            "students": self.students,
            "sections": self.sections,
            "enrollments": self.enrollments,
            "gpa_trends": self.gpa_trends,
            "over_enrollment_alerts": self.alerts,
            "semester_snapshots": self.snapshots,
        }

        for name, records in datasets.items():
            path = out / f"{name}.json"
            path.write_text(json.dumps(
                [asdict(r) for r in records], indent=2, default=str,
            ))
            files[name] = str(path)

        path = out / "summary.json"
        path.write_text(json.dumps(self.summary(), indent=2, default=str))
        files["summary"] = str(path)

        return files

    def to_csv(self, output_dir: str) -> dict[str, str]:
        """Export as CSV files for analysis."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        files = {}

        datasets: dict[str, list] = {
            "courses": self.courses,
            "semesters": self.semesters,
            "students": self.students,
            "sections": self.sections,
            "enrollments": self.enrollments,
            "gpa_trends": self.gpa_trends,
        }

        for name, records in datasets.items():
            if not records:
                continue
            path = out / f"{name}.csv"
            rows = [asdict(r) for r in records]
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                for row in rows:
                    flat = {
                        k: json.dumps(v) if isinstance(v, list) else v
                        for k, v in row.items()
                    }
                    writer.writerow(flat)
            files[name] = str(path)

        if self.alerts:
            path = out / "over_enrollment_report.csv"
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "course_code", "course_name", "semester", "instructor",
                    "max_cap", "actual", "overflow", "overflow_pct", "severity",
                ])
                for a in sorted(self.alerts, key=lambda x: (-x.overflow_pct, x.semester_name)):
                    writer.writerow([
                        a.course_code, a.course_name, a.semester_name, a.instructor,
                        a.max_capacity, a.actual_enrollment, a.overflow_count,
                        f"{a.overflow_pct:.1%}", a.severity,
                    ])
            files["over_enrollment_report"] = str(path)

        if self.snapshots:
            path = out / "semester_trends.csv"
            rows = [asdict(s) for s in self.snapshots]
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            files["semester_trends"] = str(path)

        return files

    def to_supabase_sql(self, output_dir: str) -> str:
        """Generate Supabase INSERT statements."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / "seed_data.sql"

        lines = [
            "-- Supabase Seed Data",
            f"-- Generated: {datetime.now().isoformat()}",
            f"-- Seed: {self.seed}",
            f"-- Students: {len(self.students)} | Courses: {len(self.courses)} | "
            f"Enrollments: {len(self.enrollments)}",
            "",
        ]

        table_data: dict[str, list[dict]] = {
            "courses": [c.to_supabase_row() for c in self.courses],
            "students": [s.to_supabase_row() for s in self.students],
            "enrollments": [e.to_supabase_row() for e in self.enrollments],
        }

        for table_name, rows in table_data.items():
            if not rows:
                continue
            cols = list(rows[0].keys())
            lines.append(f"-- {table_name} ({len(rows)} records)")
            for row in rows:
                vals = []
                for c in cols:
                    v = row[c]
                    if v is None:
                        vals.append("NULL")
                    elif isinstance(v, bool):
                        vals.append("true" if v else "false")
                    elif isinstance(v, (int, float)):
                        vals.append(str(v))
                    elif isinstance(v, list):
                        vals.append(f"'{json.dumps(v)}'::jsonb")
                    else:
                        escaped = str(v).replace("'", "''")
                        vals.append(f"'{escaped}'")
                lines.append(
                    f"INSERT INTO {table_name} ({', '.join(cols)}) "
                    f"VALUES ({', '.join(vals)});"
                )
            lines.append("")

        path.write_text("\n".join(lines))
        return str(path)


# ──────────────────────────────────────────────────────────────────────────────
# REPORT PRINTER
# ──────────────────────────────────────────────────────────────────────────────

def print_report(gen: AcademicDataGenerator):
    """Print comprehensive registrar report to stdout."""
    summary = gen.summary()
    trends = summary.get("trends", {})

    print(f"\n{'=' * 72}")
    print("  UNIVERSITY REGISTRAR — ENROLLMENT REPORT")
    print(f"{'=' * 72}")
    print(f"  Seed: {gen.seed} | Generated in {summary['generation_time_ms']:.0f}ms")

    print(f"\n  Students:    {summary['total_students']:,}")
    print(f"  Courses:     {summary['total_courses']:,}")
    print(f"  Sections:    {summary['total_sections']:,}")
    print(f"  Enrollments: {summary['total_enrollments']:,}")
    print(f"  Dropped:     {summary['total_drops']:,}")
    print(f"  Flagged:     {summary['total_flagged_enrollments']:,}")

    # Semester-by-semester table
    print(f"\n{'─' * 72}")
    print("  SEMESTER ENROLLMENT")
    print(f"  {'Semester':<16} {'Enroll':>8} {'Students':>10} "
          f"{'Drops':>7} {'Flagged':>9} {'Avg Size':>10} {'GPA':>6} {'Over-Cap':>10}")
    print(f"  {'─' * 76}")

    for snap in gen.snapshots:
        flag = " !!" if snap.over_enrolled_sections > 0 else ""
        print(
            f"  {snap.semester_name:<16} {snap.total_enrollments:>8,} "
            f"{snap.unique_students:>10,} {snap.total_drops:>7,} "
            f"{snap.total_flagged:>9,} {snap.avg_class_size:>10.1f} "
            f"{snap.mean_gpa:>6.2f} {snap.over_enrolled_sections:>10}{flag}"
        )

    # Trend analysis
    if isinstance(trends, dict) and "mean_enrollment_increase_pct" in trends:
        print(f"\n{'─' * 72}")
        print("  TREND ANALYSIS")
        print(f"  {'─' * 60}")
        print(f"  Mean semester increase:    {trends['mean_enrollment_increase_pct']:+.1f}%")
        print(f"  Median semester increase:  {trends['median_enrollment_increase_pct']:+.1f}%")
        print(f"  Overall growth:            {trends['overall_growth_pct']:+.1f}%")
        print(f"  Student body growth:       {trends['mean_student_growth_pct']:+.1f}%")

        if trends.get("enrollment_changes"):
            print(f"\n  {'Period':<32} {'Enrollment':>12} {'Students':>12}")
            print(f"  {'─' * 56}")
            for ch in trends["enrollment_changes"]:
                print(
                    f"  {ch['from']} -> {ch['to']:<10} "
                    f"{ch['enrollment_change_pct']:>+11.1f}% "
                    f"{ch['student_change_pct']:>+11.1f}%"
                )

    # Registration flags
    all_flags: list[str] = []
    for e in gen.enrollments:
        all_flags.extend(e.flags)
    for t in gen.gpa_trends:
        all_flags.extend(t.flags)

    if all_flags:
        print(f"\n{'─' * 72}")
        print(f"  REGISTRATION FLAGS ({len(all_flags):,} total)")
        print(f"  {'─' * 60}")
        flag_counts: dict[str, int] = defaultdict(int)
        for f in all_flags:
            flag_counts[f] += 1
        for fl, count in sorted(flag_counts.items(), key=lambda x: -x[1]):
            print(f"  {fl:25s}: {count:,}")

    # Over-enrollment alerts
    if gen.alerts:
        print(f"\n{'─' * 72}")
        print(f"  OVER-ENROLLMENT ALERTS ({len(gen.alerts)} sections)")
        print(f"  {'─' * 60}")

        by_severity: dict[str, list[OverEnrollmentAlert]] = defaultdict(list)
        for a in gen.alerts:
            by_severity[a.severity].append(a)

        for sev, limit in [
            (AlertSeverity.CRITICAL, 5),
            (AlertSeverity.HIGH, 5),
            (AlertSeverity.MEDIUM, 3),
        ]:
            alerts = by_severity.get(sev, [])
            if alerts:
                print(f"\n  [{sev.value.upper()}] ({len(alerts)}):")
                for a in sorted(alerts, key=lambda x: -x.overflow_pct)[:limit]:
                    print(
                        f"    {a.course_code:<10} {a.course_name:<28} "
                        f"{a.semester_name:<14} "
                        f"{a.actual_enrollment}/{a.max_capacity} "
                        f"(+{a.overflow_count}, {a.overflow_pct:.0%} over)"
                    )

    # GPA decline alerts
    decline_trends = [t for t in gen.gpa_trends if RegistrationFlag.MEAN_GPA_DROP in t.flags]
    if decline_trends:
        print(f"\n{'─' * 72}")
        print(f"  GPA DECLINE EARLY WARNINGS ({len(decline_trends)} students)")
        print(f"  {'─' * 60}")
        # Index for O(n) lookup instead of O(n^2)
        trends_by_student: dict[str, list[GPATrend]] = defaultdict(list)
        for t in gen.gpa_trends:
            trends_by_student[t.student_id].append(t)

        for t in decline_trends[:10]:
            hist = trends_by_student[t.student_id]
            trend_str = " -> ".join(f"{tr.semester_gpa:.2f}" for tr in hist)
            print(
                f"  {t.student_name:<25} | {t.semester_name}: "
                f"sem={t.semester_gpa:.2f} cum={t.cumulative_gpa:.2f} | {trend_str}"
            )

    print(f"\n{'=' * 72}\n")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Synthetic Academic Data Generator",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--students", type=int, default=200, help="Number of students")
    parser.add_argument("--semesters", type=int, default=6, help="Number of semesters")
    parser.add_argument("--start-year", type=int, default=2023, help="Start year")
    parser.add_argument(
        "--output", choices=["report", "json", "csv", "supabase", "all"],
        default="report", help="Output format",
    )
    parser.add_argument("--output-dir", default="./output", help="Output directory")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    gen = AcademicDataGenerator(
        num_students=args.students,
        num_semesters=args.semesters,
        start_year=args.start_year,
        seed=args.seed,
    )
    gen.generate()

    print_report(gen)

    if args.output in ("json", "all"):
        files = gen.to_json(args.output_dir)
        print(f"  JSON -> {args.output_dir}/ ({len(files)} files)")

    if args.output in ("csv", "all"):
        files = gen.to_csv(args.output_dir)
        print(f"  CSV  -> {args.output_dir}/ ({len(files)} files)")

    if args.output in ("supabase", "all"):
        path = gen.to_supabase_sql(args.output_dir)
        print(f"  SQL  -> {path}")


if __name__ == "__main__":
    main()
