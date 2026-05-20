import streamlit as st
from database import add_annotation, create_tables, get_segment_by_id, get_segment_ids
from modules.taxonomy import ERROR_TAXONOMY, SEVERITY_LEVELS

create_tables()

st.set_page_config(page_title="Teacher Annotation", page_icon="📝", layout="wide")
st.title("Teacher Annotation")
st.write("Annotate translation errors using a structured taxonomy.")

segment_ids = get_segment_ids()

if len(segment_ids) == 0:
    st.warning("No segments found. Please add a student submission first.")
else:
    selected_segment_id = st.selectbox("Select Segment ID", segment_ids)
    segment = get_segment_by_id(selected_segment_id)

    if segment:
        st.subheader("Segment Details")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Source Text**")
            st.write(segment["source_text"])
        with col2:
            st.markdown("**Machine Translation**")
            st.write(segment["mt_output"])
        with col3:
            st.markdown("**Post-Edited Text**")
            st.write(segment["post_edited_text"])

        st.divider()
        st.subheader("Annotation Form")

        annotation_id = st.text_input("Annotation ID", value=f"A_{selected_segment_id}")
        category = st.selectbox("Error Category", list(ERROR_TAXONOMY.keys()))
        subcategory = st.selectbox("Subcategory", ERROR_TAXONOMY[category])
        severity = st.selectbox("Severity", SEVERITY_LEVELS)
        teacher_comment = st.text_area("Teacher Comment", height=120)
        suggested_revision = st.text_area("Suggested Revision", height=120)
        annotator_id = st.text_input("Annotator ID", value="Teacher_1")

        if st.button("Save Annotation"):
            add_annotation(
                annotation_id=annotation_id,
                segment_id=selected_segment_id,
                category=category,
                subcategory=subcategory,
                severity=severity,
                teacher_comment=teacher_comment,
                suggested_revision=suggested_revision,
                annotator_id=annotator_id,
            )
            st.success("Annotation saved successfully.")
