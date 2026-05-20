import csv
from pathlib import Path

from database import add_annotation, add_segment, add_student, add_task, create_tables

DEMO_PATH = Path("sample_data/demo_segments.csv")


def load_demo_data():
    create_tables()
    add_task("T001", "University Announcement", "Arabic", "English", "Institutional")
    add_task("T002", "Registration Notice", "Arabic", "English", "Institutional")

    with DEMO_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            add_student(row["student_id"], row["student_id"], "Demo Group", "Demo Semester")
            add_segment(
                segment_id=row["segment_id"],
                student_id=row["student_id"],
                task_id=row["task_id"],
                source_text=row["source_text"],
                mt_output=row["mt_output"],
                post_edited_text=row["post_edited_text"],
                editing_time_seconds=float(row["editing_time_seconds"]),
            )
            add_annotation(
                annotation_id=f"A_{row['segment_id']}",
                segment_id=row["segment_id"],
                category=row["teacher_category"],
                subcategory=row["teacher_subcategory"],
                severity=row["teacher_severity"],
                teacher_comment="Demo teacher annotation.",
                suggested_revision=row["post_edited_text"],
                annotator_id="Teacher_1",
            )


if __name__ == "__main__":
    load_demo_data()
    print("Demo data loaded.")
