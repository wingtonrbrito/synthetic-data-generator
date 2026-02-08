"""Tests for the synthetic academic data generator.

Covers pure functions, dataclass invariants, input validation,
and end-to-end generation correctness.
"""

from __future__ import annotations

import json
import pytest
from random import Random

from generate_academic_data import (
    AcademicDataGenerator,
    AlertSeverity,
    Classification,
    Course,
    Enrollment,
    FinancialNeed,
    GPATrend,
    OverEnrollmentAlert,
    RegistrationFlag,
    Section,
    Student,
    StudentStatus,
    _days_overlap,
    _times_overlap,
    detect_enrollment_flags,
    detect_gpa_decline,
    generate_correlated_grade,
    has_time_conflict,
    GRADE_POINTS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng() -> Random:
    return Random(42)


@pytest.fixture
def sample_course() -> Course:
    return Course(
        id="c-1",
        code="CS 201",
        title="Data Structures",
        department="Computer Science",
        credits=3,
        capacity=30,
        prerequisites=["CS 101"],
    )


@pytest.fixture
def sample_section() -> Section:
    return Section(
        id="sec-1",
        course_id="c-1",
        course_code="CS 201",
        semester_id="sem-1",
        instructor="Dr. Thompson",
        days="MWF",
        start_time="09:30",
        end_time="10:45",
        room="Science Hall 200",
        capacity=30,
    )


@pytest.fixture
def sample_student() -> Student:
    return Student(
        id="s-1",
        student_id="S01-ABC12",
        first_name="Alice",
        last_name="Smith",
        email="alice.smith@university.edu",
        date_of_birth="2002-05-15",
        classification=Classification.SOPHOMORE,
        major="Computer Science",
        gpa=3.2,
        credits_earned=45,
        enrollment_status=StudentStatus.ACTIVE,
        financial_need=FinancialNeed.MEDIUM,
    )


# ---------------------------------------------------------------------------
# Pure function: _days_overlap
# ---------------------------------------------------------------------------

class TestDaysOverlap:
    def test_same_pattern(self):
        assert _days_overlap("MWF", "MWF") is True

    def test_partial_overlap(self):
        assert _days_overlap("MWF", "MW") is True

    def test_no_overlap(self):
        assert _days_overlap("MWF", "TR") is False

    def test_single_day_overlap(self):
        assert _days_overlap("M", "MWF") is True

    def test_all_days(self):
        assert _days_overlap("MTWRF", "TR") is True


# ---------------------------------------------------------------------------
# Pure function: _times_overlap
# ---------------------------------------------------------------------------

class TestTimesOverlap:
    def test_exact_overlap(self):
        assert _times_overlap("09:00", "10:00", "09:00", "10:00") is True

    def test_partial_overlap(self):
        assert _times_overlap("09:00", "10:00", "09:30", "10:30") is True

    def test_no_overlap(self):
        assert _times_overlap("09:00", "10:00", "10:00", "11:00") is False

    def test_no_overlap_separate(self):
        assert _times_overlap("08:00", "09:00", "11:00", "12:00") is False

    def test_contained(self):
        assert _times_overlap("09:00", "12:00", "10:00", "11:00") is True

    def test_adjacent_no_overlap(self):
        assert _times_overlap("09:00", "10:15", "10:15", "11:30") is False


# ---------------------------------------------------------------------------
# Pure function: has_time_conflict
# ---------------------------------------------------------------------------

class TestHasTimeConflict:
    def test_conflict_same_day_and_time(self):
        new_sec = Section(
            id="s-new", course_id="c-1", course_code="CS 101",
            semester_id="sem-1", instructor="Dr. X", days="MWF",
            start_time="09:30", end_time="10:45", room="R1", capacity=30,
        )
        existing = Section(
            id="s-old", course_id="c-2", course_code="CS 202",
            semester_id="sem-1", instructor="Dr. Y", days="MW",
            start_time="09:00", end_time="10:00", room="R2", capacity=30,
        )
        assert has_time_conflict(new_sec, [existing]) is True

    def test_no_conflict_different_days(self):
        new_sec = Section(
            id="s-new", course_id="c-1", course_code="CS 101",
            semester_id="sem-1", instructor="Dr. X", days="MWF",
            start_time="09:00", end_time="10:00", room="R1", capacity=30,
        )
        existing = Section(
            id="s-old", course_id="c-2", course_code="CS 202",
            semester_id="sem-1", instructor="Dr. Y", days="TR",
            start_time="09:00", end_time="10:00", room="R2", capacity=30,
        )
        assert has_time_conflict(new_sec, [existing]) is False

    def test_no_conflict_different_times(self):
        new_sec = Section(
            id="s-new", course_id="c-1", course_code="CS 101",
            semester_id="sem-1", instructor="Dr. X", days="MWF",
            start_time="08:00", end_time="09:00", room="R1", capacity=30,
        )
        existing = Section(
            id="s-old", course_id="c-2", course_code="CS 202",
            semester_id="sem-1", instructor="Dr. Y", days="MWF",
            start_time="11:00", end_time="12:00", room="R2", capacity=30,
        )
        assert has_time_conflict(new_sec, [existing]) is False

    def test_empty_enrolled_sections(self):
        sec = Section(
            id="s-1", course_id="c-1", course_code="CS 101",
            semester_id="sem-1", instructor="Dr. X", days="MWF",
            start_time="09:00", end_time="10:00", room="R1", capacity=30,
        )
        assert has_time_conflict(sec, []) is False


# ---------------------------------------------------------------------------
# Pure function: detect_gpa_decline
# ---------------------------------------------------------------------------

class TestDetectGpaDecline:
    def test_decline_detected(self):
        assert detect_gpa_decline(2.0, [3.5, 3.4]) is True

    def test_no_decline(self):
        assert detect_gpa_decline(3.3, [3.5, 3.4]) is False

    def test_insufficient_history(self):
        assert detect_gpa_decline(1.0, [3.5]) is False

    def test_empty_history(self):
        assert detect_gpa_decline(2.0, []) is False

    def test_exact_threshold(self):
        # Recent avg = 3.0; threshold = 0.5 -> decline if current < 2.5
        assert detect_gpa_decline(2.5, [3.0, 3.0]) is False
        assert detect_gpa_decline(2.49, [3.0, 3.0]) is True

    def test_custom_threshold(self):
        # Recent avg = 3.0; threshold 0.3 -> decline if current < 2.7
        assert detect_gpa_decline(2.6, [3.0, 3.0], threshold=0.3) is True
        assert detect_gpa_decline(2.8, [3.0, 3.0], threshold=0.3) is False


# ---------------------------------------------------------------------------
# Pure function: generate_correlated_grade
# ---------------------------------------------------------------------------

class TestGenerateCorrelatedGrade:
    def test_returns_valid_grade(self, rng):
        for gpa in [0.5, 1.5, 2.5, 3.5, 4.0]:
            grade = generate_correlated_grade(rng, gpa)
            assert grade in GRADE_POINTS

    def test_high_gpa_never_gets_F(self, rng):
        """High GPA students shouldn't get F — sampling pool doesn't include it."""
        grades = [generate_correlated_grade(rng, 3.8) for _ in range(200)]
        assert "F" not in grades

    def test_low_gpa_never_gets_A_plus(self, rng):
        """Low GPA students shouldn't get A+ — sampling pool doesn't include it."""
        grades = [generate_correlated_grade(rng, 0.8) for _ in range(200)]
        assert "A+" not in grades

    def test_deterministic_with_seed(self):
        """Same seed produces same grade sequence."""
        rng1 = Random(99)
        rng2 = Random(99)
        grades1 = [generate_correlated_grade(rng1, 3.0) for _ in range(10)]
        grades2 = [generate_correlated_grade(rng2, 3.0) for _ in range(10)]
        assert grades1 == grades2


# ---------------------------------------------------------------------------
# detect_enrollment_flags
# ---------------------------------------------------------------------------

class TestDetectEnrollmentFlags:
    def test_prereq_missing_flag(self, sample_student, sample_course, sample_section, rng):
        flags = detect_enrollment_flags(
            student=sample_student,
            course=sample_course,
            section=sample_section,
            completed_courses=set(),  # CS 101 not completed
            total_credits=6,
            enrolled_sections=[],
            rng=rng,
        )
        assert RegistrationFlag.PREREQUISITE_MISSING in flags

    def test_prereq_satisfied_no_flag(self, sample_student, sample_course, sample_section, rng):
        flags = detect_enrollment_flags(
            student=sample_student,
            course=sample_course,
            section=sample_section,
            completed_courses={"CS 101"},
            total_credits=6,
            enrolled_sections=[],
            rng=rng,
        )
        assert RegistrationFlag.PREREQUISITE_MISSING not in flags

    def test_over_enrolled_flag(self, sample_student, sample_course, sample_section, rng):
        sample_section.enrolled_count = 31  # Over capacity of 30
        flags = detect_enrollment_flags(
            student=sample_student,
            course=sample_course,
            section=sample_section,
            completed_courses={"CS 101"},
            total_credits=6,
            enrolled_sections=[],
            rng=rng,
        )
        assert RegistrationFlag.OVER_ENROLLED in flags

    def test_credit_overload_flag(self, sample_student, sample_course, sample_section, rng):
        flags = detect_enrollment_flags(
            student=sample_student,
            course=sample_course,
            section=sample_section,
            completed_courses={"CS 101"},
            total_credits=16,  # Over CREDIT_OVERLOAD_THRESHOLD of 15
            enrolled_sections=[],
            rng=rng,
        )
        assert RegistrationFlag.CREDIT_OVERLOAD in flags

    def test_probation_lower_credit_cap(self, sample_course, sample_section, rng):
        prob_student = Student(
            id="s-2", student_id="S01-XYZ99",
            first_name="Bob", last_name="Jones",
            email="bob@university.edu",
            date_of_birth="2001-01-01",
            classification=Classification.JUNIOR,
            major="CS", gpa=1.8,
            credits_earned=62,
            enrollment_status=StudentStatus.PROBATION,
            financial_need=FinancialNeed.LOW,
        )
        flags = detect_enrollment_flags(
            student=prob_student,
            course=sample_course,
            section=sample_section,
            completed_courses={"CS 101"},
            total_credits=10,  # Over PROBATION_CREDIT_CAP of 9
            enrolled_sections=[],
            rng=rng,
        )
        assert RegistrationFlag.CREDIT_OVERLOAD in flags

    def test_time_conflict_flag(self, sample_student, sample_course, sample_section, rng):
        conflicting = Section(
            id="sec-conflict", course_id="c-2", course_code="CS 202",
            semester_id="sem-1", instructor="Dr. Y", days="MW",
            start_time="09:00", end_time="10:30", room="R2", capacity=30,
        )
        flags = detect_enrollment_flags(
            student=sample_student,
            course=sample_course,
            section=sample_section,
            completed_courses={"CS 101"},
            total_credits=6,
            enrolled_sections=[conflicting],
            rng=rng,
        )
        assert RegistrationFlag.TIME_CONFLICT in flags


# ---------------------------------------------------------------------------
# Student classmethods
# ---------------------------------------------------------------------------

class TestStudentClassify:
    @pytest.mark.parametrize("credits,expected", [
        (0, Classification.FRESHMAN),
        (29, Classification.FRESHMAN),
        (30, Classification.SOPHOMORE),
        (59, Classification.SOPHOMORE),
        (60, Classification.JUNIOR),
        (89, Classification.JUNIOR),
        (90, Classification.SENIOR),
        (130, Classification.SENIOR),
    ])
    def test_classify_level(self, credits, expected):
        assert Student.classify_level(credits) == expected

    @pytest.mark.parametrize("gpa,expected", [
        (0.0, StudentStatus.SUSPENDED),
        (1.4, StudentStatus.SUSPENDED),
        (1.5, StudentStatus.PROBATION),
        (1.9, StudentStatus.PROBATION),
        (2.0, StudentStatus.ACTIVE),
        (4.0, StudentStatus.ACTIVE),
    ])
    def test_classify_standing(self, gpa, expected):
        assert Student.classify_standing(gpa) == expected


# ---------------------------------------------------------------------------
# Dataclass computed properties
# ---------------------------------------------------------------------------

class TestCourseProperties:
    def test_level_field(self):
        c = Course(id="1", code="CS 301", title="X", department="CS",
                   credits=3, capacity=30, level=300)
        assert c.level == 300

    def test_has_prerequisites(self, sample_course):
        assert sample_course.has_prerequisites is True

    def test_no_prerequisites(self):
        c = Course(id="1", code="CS 101", title="Intro", department="CS",
                   credits=3, capacity=30, prerequisites=[])
        assert c.has_prerequisites is False


class TestSectionProperties:
    def test_over_capacity(self, sample_section):
        sample_section.enrolled_count = 31
        assert sample_section.is_over_capacity is True
        assert sample_section.overflow_count == 1

    def test_under_capacity(self, sample_section):
        sample_section.enrolled_count = 25
        assert sample_section.is_over_capacity is False
        assert sample_section.overflow_count == 0

    def test_fill_rate(self, sample_section):
        sample_section.enrolled_count = 15
        assert sample_section.fill_rate == 0.5


class TestStudentProperties:
    def test_full_name(self, sample_student):
        assert sample_student.full_name == "Alice Smith"

    def test_is_active(self, sample_student):
        assert sample_student.is_active is True

    def test_probation_is_active(self, sample_student):
        sample_student.enrollment_status = StudentStatus.PROBATION
        assert sample_student.is_active is True

    def test_suspended_not_active(self, sample_student):
        sample_student.enrollment_status = StudentStatus.SUSPENDED
        assert sample_student.is_active is False


class TestStudentIdStability:
    def test_to_supabase_row_idempotent(self, sample_student):
        """student_id should not change between calls."""
        row1 = sample_student.to_supabase_row()
        row2 = sample_student.to_supabase_row()
        assert row1["student_id"] == row2["student_id"]
        assert row1["id"] == row2["id"]


# ---------------------------------------------------------------------------
# OverEnrollmentAlert severity
# ---------------------------------------------------------------------------

class TestAlertSeverity:
    @pytest.mark.parametrize("pct,expected", [
        (0.30, AlertSeverity.CRITICAL),
        (0.26, AlertSeverity.CRITICAL),
        (0.20, AlertSeverity.HIGH),
        (0.16, AlertSeverity.HIGH),
        (0.10, AlertSeverity.MEDIUM),
        (0.06, AlertSeverity.MEDIUM),
        (0.04, AlertSeverity.LOW),
        (0.00, AlertSeverity.LOW),
    ])
    def test_severity_grading(self, pct, expected):
        alert = OverEnrollmentAlert(
            course_code="CS 101",
            course_name="Intro",
            semester_name="Fall 2025",
            section_id="s-1",
            instructor="Dr. X",
            max_capacity=30,
            actual_enrollment=int(30 * (1 + pct)),
            overflow_count=int(30 * pct),
            overflow_pct=pct,
        )
        assert alert.severity == expected


# ---------------------------------------------------------------------------
# Enrollment serialization
# ---------------------------------------------------------------------------

class TestEnrollmentSerialization:
    def test_to_supabase_row_has_semester_fields(self):
        e = Enrollment(
            id="e-1", student_id="s-1", section_id="sec-1",
            course_id="c-1", semester_id="sem-1", semester_name="Fall",
            semester_year=2025, course_code="CS 201", grade="B+",
        )
        row = e.to_supabase_row()
        assert row["semester"] == "Fall"
        assert row["year"] == 2025
        assert row["status"] == "completed"
        assert row["course_id"] == "c-1"

    def test_dropped_status(self):
        e = Enrollment(
            id="e-1", student_id="s-1", section_id="sec-1",
            course_id="c-1", semester_id="sem-1", semester_name="Spring",
            semester_year=2024, course_code="CS 101",
            grade=None, dropped=True,
        )
        row = e.to_supabase_row()
        assert row["status"] == "dropped"

    def test_enrolled_status(self):
        e = Enrollment(
            id="e-1", student_id="s-1", section_id="sec-1",
            course_id="c-1", semester_id="sem-1", semester_name="Spring",
            semester_year=2024, course_code="CS 101",
            grade=None, dropped=False,
        )
        row = e.to_supabase_row()
        assert row["status"] == "enrolled"


# ---------------------------------------------------------------------------
# Input Validation
# ---------------------------------------------------------------------------

class TestInputValidation:
    def test_zero_students_raises(self):
        with pytest.raises(ValueError, match="num_students"):
            AcademicDataGenerator(num_students=0)

    def test_negative_students_raises(self):
        with pytest.raises(ValueError, match="num_students"):
            AcademicDataGenerator(num_students=-5)

    def test_zero_semesters_raises(self):
        with pytest.raises(ValueError, match="num_semesters"):
            AcademicDataGenerator(num_semesters=0)

    def test_start_year_too_low(self):
        with pytest.raises(ValueError, match="start_year"):
            AcademicDataGenerator(start_year=1999)

    def test_start_year_too_high(self):
        with pytest.raises(ValueError, match="start_year"):
            AcademicDataGenerator(start_year=2101)

    def test_valid_params_no_raise(self):
        gen = AcademicDataGenerator(num_students=10, num_semesters=2, start_year=2024)
        assert gen.num_students == 10


# ---------------------------------------------------------------------------
# End-to-end generation
# ---------------------------------------------------------------------------

class TestEndToEnd:
    @pytest.fixture
    def gen(self) -> AcademicDataGenerator:
        g = AcademicDataGenerator(num_students=50, num_semesters=4, seed=99)
        g.generate()
        return g

    def test_generation_produces_data(self, gen):
        assert len(gen.courses) > 0
        assert len(gen.students) == 50
        assert len(gen.semesters) == 4
        assert len(gen.sections) > 0
        assert len(gen.enrollments) > 0
        assert len(gen.gpa_trends) > 0
        assert len(gen.snapshots) == 4

    def test_enrollment_growth(self, gen):
        """Later semesters should have more enrollments (staggered admits)."""
        enrollment_counts = [s.total_enrollments for s in gen.snapshots]
        # Last semester should be larger than first (staggered admits)
        assert enrollment_counts[-1] > enrollment_counts[0]

    def test_deterministic_output(self):
        """Same seed produces identical results."""
        gen1 = AcademicDataGenerator(num_students=20, num_semesters=3, seed=77)
        gen2 = AcademicDataGenerator(num_students=20, num_semesters=3, seed=77)
        s1 = gen1.generate()
        s2 = gen2.generate()
        assert s1["total_enrollments"] == s2["total_enrollments"]
        assert s1["total_drops"] == s2["total_drops"]

    def test_summary_has_required_keys(self, gen):
        s = gen.summary()
        required = [
            "seed", "total_students", "total_courses", "total_sections",
            "total_enrollments", "semesters_generated", "total_drops",
            "total_flagged_enrollments", "over_enrollment_alerts",
            "gpa_decline_alerts", "snapshots", "trends",
        ]
        for key in required:
            assert key in s, f"Missing key: {key}"

    def test_indexes_populated(self, gen):
        """Pre-built indexes should have entries for all semesters."""
        for sem in gen.semesters:
            assert sem.id in gen._enrollments_by_semester
            assert sem.id in gen._sections_by_semester

    def test_flags_include_prereq_missing(self, gen):
        """With no prior history, early semesters should have prereq flags."""
        all_flags = []
        for e in gen.enrollments:
            all_flags.extend(e.flags)
        assert RegistrationFlag.PREREQUISITE_MISSING in all_flags

    def test_time_conflict_flags_present(self, gen):
        """Time conflicts should be detected when students take multiple courses."""
        all_flags = []
        for e in gen.enrollments:
            all_flags.extend(e.flags)
        assert RegistrationFlag.TIME_CONFLICT in all_flags

    def test_json_export(self, gen, tmp_path):
        files = gen.to_json(str(tmp_path))
        assert "courses" in files
        assert "students" in files
        assert "enrollments" in files
        # Verify JSON is valid
        courses = json.loads((tmp_path / "courses.json").read_text())
        assert len(courses) == len(gen.courses)

    def test_csv_export(self, gen, tmp_path):
        files = gen.to_csv(str(tmp_path))
        assert "courses" in files
        assert "students" in files
        assert (tmp_path / "courses.csv").exists()

    def test_supabase_sql_export(self, gen, tmp_path):
        path = gen.to_supabase_sql(str(tmp_path))
        content = (tmp_path / "seed_data.sql").read_text()
        assert "INSERT INTO courses" in content
        assert "INSERT INTO students" in content
        assert "INSERT INTO enrollments" in content
