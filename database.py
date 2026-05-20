import json
import sqlite3
from pathlib import Path

import pandas as pd

DB_PATH = Path("postediting_data.db")


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def create_tables():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT UNIQUE NOT NULL,
            student_name TEXT,
            group_name TEXT,
            semester TEXT
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id TEXT UNIQUE NOT NULL,
            task_title TEXT,
            source_language TEXT,
            target_language TEXT,
            domain TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS segments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            segment_id TEXT UNIQUE NOT NULL,
            student_id TEXT NOT NULL,
            task_id TEXT NOT NULL,
            source_text TEXT NOT NULL,
            mt_output TEXT NOT NULL,
            post_edited_text TEXT NOT NULL,
            editing_time_seconds REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (student_id) REFERENCES students(student_id),
            FOREIGN KEY (task_id) REFERENCES tasks(task_id)
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS annotations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            annotation_id TEXT UNIQUE NOT NULL,
            segment_id TEXT NOT NULL,
            category TEXT,
            subcategory TEXT,
            severity TEXT,
            teacher_comment TEXT,
            suggested_revision TEXT,
            annotator_id TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (segment_id) REFERENCES segments(segment_id)
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS ai_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            feedback_id TEXT UNIQUE NOT NULL,
            segment_id TEXT NOT NULL,
            ai_overall_comment TEXT,
            ai_possible_errors TEXT,
            ai_rubric_scores TEXT,
            ai_raw_json TEXT,
            ai_risk_level TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (segment_id) REFERENCES segments(segment_id)
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS teacher_reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            review_id TEXT UNIQUE NOT NULL,
            feedback_id TEXT NOT NULL,
            segment_id TEXT NOT NULL,
            review_status TEXT,
            teacher_final_feedback TEXT,
            teacher_notes TEXT,
            reviewer_id TEXT,
            usefulness_rating INTEGER,
            teacher_review_time_seconds REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (feedback_id) REFERENCES ai_feedback(feedback_id),
            FOREIGN KEY (segment_id) REFERENCES segments(segment_id)
        )
        """
    )

    conn.commit()
    conn.close()


def add_column_if_not_exists(table_name, column_name, column_type):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = [row[1] for row in cursor.fetchall()]

    if column_name not in columns:
        cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")

    conn.commit()
    conn.close()


def add_research_columns():
    add_column_if_not_exists("teacher_reviews", "usefulness_rating", "INTEGER")
    add_column_if_not_exists("teacher_reviews", "teacher_review_time_seconds", "REAL")


def add_student(student_id, student_name=None, group_name=None, semester=None):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT OR IGNORE INTO students (student_id, student_name, group_name, semester)
        VALUES (?, ?, ?, ?)
        """,
        (student_id, student_name, group_name, semester),
    )
    conn.commit()
    conn.close()


def add_task(task_id, task_title, source_language, target_language, domain):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT OR IGNORE INTO tasks
        (task_id, task_title, source_language, target_language, domain)
        VALUES (?, ?, ?, ?, ?)
        """,
        (task_id, task_title, source_language, target_language, domain),
    )
    conn.commit()
    conn.close()


def add_segment(
    segment_id,
    student_id,
    task_id,
    source_text,
    mt_output,
    post_edited_text,
    editing_time_seconds,
):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT OR REPLACE INTO segments
        (segment_id, student_id, task_id, source_text, mt_output, post_edited_text, editing_time_seconds)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            segment_id,
            student_id,
            task_id,
            source_text,
            mt_output,
            post_edited_text,
            editing_time_seconds,
        ),
    )
    conn.commit()
    conn.close()


def add_annotation(
    annotation_id,
    segment_id,
    category,
    subcategory,
    severity,
    teacher_comment,
    suggested_revision,
    annotator_id,
):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT OR REPLACE INTO annotations
        (annotation_id, segment_id, category, subcategory, severity, teacher_comment, suggested_revision, annotator_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            annotation_id,
            segment_id,
            category,
            subcategory,
            severity,
            teacher_comment,
            suggested_revision,
            annotator_id,
        ),
    )
    conn.commit()
    conn.close()


def save_ai_feedback(
    feedback_id,
    segment_id,
    ai_overall_comment,
    ai_possible_errors,
    ai_rubric_scores,
    ai_raw_json,
    ai_risk_level,
):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT OR REPLACE INTO ai_feedback
        (feedback_id, segment_id, ai_overall_comment, ai_possible_errors, ai_rubric_scores, ai_raw_json, ai_risk_level)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            feedback_id,
            segment_id,
            ai_overall_comment,
            json.dumps(ai_possible_errors, ensure_ascii=False),
            json.dumps(ai_rubric_scores, ensure_ascii=False),
            json.dumps(ai_raw_json, ensure_ascii=False),
            ai_risk_level,
        ),
    )
    conn.commit()
    conn.close()


def save_teacher_review(
    review_id,
    feedback_id,
    segment_id,
    review_status,
    teacher_final_feedback,
    teacher_notes,
    reviewer_id,
    usefulness_rating=None,
    teacher_review_time_seconds=None,
):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT OR REPLACE INTO teacher_reviews
        (review_id, feedback_id, segment_id, review_status, teacher_final_feedback,
         teacher_notes, reviewer_id, usefulness_rating, teacher_review_time_seconds)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            review_id,
            feedback_id,
            segment_id,
            review_status,
            teacher_final_feedback,
            teacher_notes,
            reviewer_id,
            usefulness_rating,
            teacher_review_time_seconds,
        ),
    )
    conn.commit()
    conn.close()


def load_table(table_name):
    allowed = {"students", "tasks", "segments", "annotations", "ai_feedback", "teacher_reviews"}
    if table_name not in allowed:
        raise ValueError(f"Unsupported table name: {table_name}")
    conn = get_connection()
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    conn.close()
    return df


def load_segments_with_annotations():
    conn = get_connection()
    query = """
        SELECT
            segments.segment_id,
            segments.student_id,
            segments.task_id,
            segments.source_text,
            segments.mt_output,
            segments.post_edited_text,
            segments.editing_time_seconds,
            annotations.category,
            annotations.subcategory,
            annotations.severity,
            annotations.teacher_comment,
            annotations.suggested_revision,
            annotations.annotator_id
        FROM segments
        LEFT JOIN annotations ON segments.segment_id = annotations.segment_id
        ORDER BY segments.segment_id
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def get_segment_ids():
    conn = get_connection()
    df = pd.read_sql_query("SELECT segment_id FROM segments ORDER BY segment_id", conn)
    conn.close()
    return df["segment_id"].tolist()


def get_segment_by_id(segment_id):
    conn = get_connection()
    df = pd.read_sql_query(
        "SELECT * FROM segments WHERE segment_id = ?",
        conn,
        params=(segment_id,),
    )
    conn.close()
    if df.empty:
        return None
    return df.iloc[0].to_dict()


def load_ai_feedback():
    conn = get_connection()
    query = """
        SELECT
            ai_feedback.feedback_id,
            ai_feedback.segment_id,
            segments.source_text,
            segments.mt_output,
            segments.post_edited_text,
            segments.editing_time_seconds,
            ai_feedback.ai_overall_comment,
            ai_feedback.ai_possible_errors,
            ai_feedback.ai_rubric_scores,
            ai_feedback.ai_raw_json,
            ai_feedback.ai_risk_level,
            ai_feedback.created_at
        FROM ai_feedback
        LEFT JOIN segments ON ai_feedback.segment_id = segments.segment_id
        ORDER BY ai_feedback.created_at DESC
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def load_teacher_reviews():
    conn = get_connection()
    query = """
        SELECT
            teacher_reviews.review_id,
            teacher_reviews.feedback_id,
            teacher_reviews.segment_id,
            teacher_reviews.review_status,
            teacher_reviews.teacher_final_feedback,
            teacher_reviews.teacher_notes,
            teacher_reviews.reviewer_id,
            teacher_reviews.usefulness_rating,
            teacher_reviews.teacher_review_time_seconds,
            teacher_reviews.created_at,
            ai_feedback.ai_overall_comment,
            ai_feedback.ai_risk_level
        FROM teacher_reviews
        LEFT JOIN ai_feedback ON teacher_reviews.feedback_id = ai_feedback.feedback_id
        ORDER BY teacher_reviews.created_at DESC
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def load_evaluation_dataset():
    conn = get_connection()
    query = """
        SELECT
            segments.segment_id,
            segments.student_id,
            segments.task_id,
            segments.source_text,
            segments.mt_output,
            segments.post_edited_text,
            segments.editing_time_seconds,
            annotations.category AS teacher_category,
            annotations.subcategory AS teacher_subcategory,
            annotations.severity AS teacher_severity,
            annotations.teacher_comment,
            annotations.suggested_revision,
            ai_feedback.feedback_id,
            ai_feedback.ai_overall_comment,
            ai_feedback.ai_possible_errors,
            ai_feedback.ai_rubric_scores,
            ai_feedback.ai_raw_json,
            ai_feedback.ai_risk_level,
            teacher_reviews.review_id,
            teacher_reviews.review_status,
            teacher_reviews.teacher_final_feedback,
            teacher_reviews.teacher_notes,
            teacher_reviews.reviewer_id,
            teacher_reviews.usefulness_rating,
            teacher_reviews.teacher_review_time_seconds
        FROM segments
        LEFT JOIN annotations ON segments.segment_id = annotations.segment_id
        LEFT JOIN ai_feedback ON segments.segment_id = ai_feedback.segment_id
        LEFT JOIN teacher_reviews ON ai_feedback.feedback_id = teacher_reviews.feedback_id
        ORDER BY segments.segment_id
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df
