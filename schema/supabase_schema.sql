-- =============================================================================
-- Supabase Schema for Academic Data Generator
-- =============================================================================
-- Relational schema for university registrar data.
-- Designed for Supabase (PostgreSQL 15+).
-- Tables: courses, students, enrollments + audit log.
-- =============================================================================

-- ---------------------------------------------------------------------------
-- SECTION 1 — Extensions
-- ---------------------------------------------------------------------------

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";      -- uuid_generate_v4()

-- ---------------------------------------------------------------------------
-- SECTION 2 — Enum Types
-- ---------------------------------------------------------------------------

CREATE TYPE enrollment_status AS ENUM (
    'active', 'probation', 'graduated', 'suspended', 'withdrawn'
);

CREATE TYPE enrollment_grade_status AS ENUM (
    'enrolled', 'completed', 'dropped'
);

CREATE TYPE audit_action AS ENUM ('INSERT', 'UPDATE', 'DELETE');

-- ---------------------------------------------------------------------------
-- SECTION 3 — Core Tables
-- ---------------------------------------------------------------------------

-- Courses
CREATE TABLE courses (
    id                uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
    code              text UNIQUE NOT NULL,          -- e.g. CS101
    name              text NOT NULL,
    department        text NOT NULL,
    credits           int NOT NULL DEFAULT 3,
    description       text,
    level             int NOT NULL DEFAULT 100,      -- 100, 200, 300, 400
    prerequisites     jsonb DEFAULT '[]',            -- array of course codes
    max_enrollment    int NOT NULL DEFAULT 35,
    is_active         boolean NOT NULL DEFAULT true,
    created_at        timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX idx_courses_department ON courses (department);

-- Students
CREATE TABLE students (
    id                uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
    student_id        text UNIQUE NOT NULL,          -- e.g. S24-12345
    first_name        text NOT NULL,
    last_name         text NOT NULL,
    email             text UNIQUE NOT NULL,
    date_of_birth     date,
    gpa               numeric(3,2) DEFAULT 0.00,
    credits_completed int NOT NULL DEFAULT 0,
    enrollment_status enrollment_status NOT NULL DEFAULT 'active',
    metadata          jsonb DEFAULT '{}',
    created_at        timestamptz NOT NULL DEFAULT now(),
    updated_at        timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX idx_students_status ON students (enrollment_status);
CREATE INDEX idx_students_gpa ON students (gpa);

-- Enrollments (student ↔ course junction)
CREATE TABLE enrollments (
    id          uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
    student_id  uuid NOT NULL REFERENCES students(id) ON DELETE CASCADE,
    course_id   uuid NOT NULL REFERENCES courses(id) ON DELETE CASCADE,
    semester    text NOT NULL,                      -- e.g. "Fall 2025"
    year        int NOT NULL,
    grade       text,                               -- A+, A, A-, B+, …, F
    status      enrollment_grade_status NOT NULL DEFAULT 'enrolled',
    created_at  timestamptz NOT NULL DEFAULT now(),
    UNIQUE (student_id, course_id, semester, year)
);

CREATE INDEX idx_enrollments_student ON enrollments (student_id);
CREATE INDEX idx_enrollments_course ON enrollments (course_id);

-- ---------------------------------------------------------------------------
-- SECTION 4 — Row Level Security (RLS)
-- ---------------------------------------------------------------------------

ALTER TABLE students ENABLE ROW LEVEL SECURITY;
ALTER TABLE enrollments ENABLE ROW LEVEL SECURITY;

-- Students can view their own record
CREATE POLICY students_self_access ON students
    FOR SELECT
    USING (auth.uid() = id);

-- Students can view their own enrollments
CREATE POLICY enrollments_student_access ON enrollments
    FOR SELECT
    USING (student_id = auth.uid());

-- Admin role has full access
CREATE POLICY admin_full_students ON students
    FOR ALL USING (auth.jwt()->>'role' = 'admin');

CREATE POLICY admin_full_enrollments ON enrollments
    FOR ALL USING (auth.jwt()->>'role' = 'admin');

-- ---------------------------------------------------------------------------
-- SECTION 5 — Audit Log
-- ---------------------------------------------------------------------------

CREATE TABLE audit_log (
    id              uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
    table_name      text NOT NULL,
    record_id       uuid NOT NULL,
    action          audit_action NOT NULL,
    old_data        jsonb,
    new_data        jsonb,
    performed_by    uuid,
    performed_at    timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX idx_audit_table ON audit_log (table_name);
CREATE INDEX idx_audit_record ON audit_log (record_id);
CREATE INDEX idx_audit_time ON audit_log (performed_at);

-- Generic audit trigger function
CREATE OR REPLACE FUNCTION audit_trigger_func()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        INSERT INTO audit_log (table_name, record_id, action, new_data, performed_by)
        VALUES (TG_TABLE_NAME, NEW.id, 'INSERT', to_jsonb(NEW), auth.uid());
        RETURN NEW;
    ELSIF TG_OP = 'UPDATE' THEN
        INSERT INTO audit_log (table_name, record_id, action, old_data, new_data, performed_by)
        VALUES (TG_TABLE_NAME, NEW.id, 'UPDATE', to_jsonb(OLD), to_jsonb(NEW), auth.uid());
        RETURN NEW;
    ELSIF TG_OP = 'DELETE' THEN
        INSERT INTO audit_log (table_name, record_id, action, old_data, performed_by)
        VALUES (TG_TABLE_NAME, OLD.id, 'DELETE', to_jsonb(OLD), auth.uid());
        RETURN OLD;
    END IF;
    RETURN NULL;
END;
$$;

CREATE TRIGGER audit_students
    AFTER INSERT OR UPDATE OR DELETE ON students
    FOR EACH ROW EXECUTE FUNCTION audit_trigger_func();

CREATE TRIGGER audit_courses
    AFTER INSERT OR UPDATE OR DELETE ON courses
    FOR EACH ROW EXECUTE FUNCTION audit_trigger_func();

CREATE TRIGGER audit_enrollments
    AFTER INSERT OR UPDATE OR DELETE ON enrollments
    FOR EACH ROW EXECUTE FUNCTION audit_trigger_func();

-- ---------------------------------------------------------------------------
-- SECTION 6 — Updated-at Trigger
-- ---------------------------------------------------------------------------

CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$;

CREATE TRIGGER set_updated_at_students
    BEFORE UPDATE ON students
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();
