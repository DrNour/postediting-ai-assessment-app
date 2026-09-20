"""Automatic right-to-left presentation for Arabic content in Streamlit."""

from __future__ import annotations

import re


_ARABIC_CHARACTER = re.compile(
    "["
    "\u0600-\u06ff"
    "\u0750-\u077f"
    "\u08a0-\u08ff"
    "\ufb50-\ufdff"
    "\ufe70-\ufeff"
    "]"
)
_LATIN_CHARACTER = re.compile(r"[A-Za-z]")


def is_arabic_dominant(value: object) -> bool:
    """Return True when visible letter content is predominantly Arabic."""

    text = "" if value is None else str(value)
    arabic_count = len(_ARABIC_CHARACTER.findall(text))
    latin_count = len(_LATIN_CHARACTER.findall(text))
    return arabic_count >= 2 and arabic_count >= latin_count


def install_arabic_text_layout() -> None:
    """Install global Arabic-aware layout without changing English UI direction."""

    import streamlit as st
    import streamlit.components.v1 as components

    st.markdown(
        """
        <style>
        .eduapp-arabic-block {
            direction: rtl !important;
            text-align: justify !important;
            text-align-last: right !important;
            unicode-bidi: plaintext !important;
            line-height: 1.9 !important;
        }

        textarea.eduapp-arabic-input {
            direction: rtl !important;
            text-align: justify !important;
            text-align-last: right !important;
            unicode-bidi: plaintext !important;
            line-height: 1.9 !important;
        }

        input.eduapp-arabic-input,
        [role="option"].eduapp-arabic-input,
        [role="gridcell"].eduapp-arabic-input,
        td.eduapp-arabic-input,
        th.eduapp-arabic-input {
            direction: rtl !important;
            text-align: right !important;
            unicode-bidi: plaintext !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    components.html(
        r"""
        <script>
        (() => {
          let rootDoc;
          let rootWin;
          try {
            rootDoc = window.parent.document;
            rootWin = window.parent;
          } catch (error) {
            rootDoc = document;
            rootWin = window;
          }

          const arabicPattern = /[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]/g;
          const latinPattern = /[A-Za-z]/g;

          const isArabicDominant = (value) => {
            const text = String(value || "");
            const arabicCount = (text.match(arabicPattern) || []).length;
            const latinCount = (text.match(latinPattern) || []).length;
            return arabicCount >= 2 && arabicCount >= latinCount;
          };

          const setAutomaticDirection = (element, className, isArabic) => {
            if (isArabic) {
              element.classList.add(className);
              element.setAttribute("dir", "rtl");
              element.dataset.eduappAutoDirection = "rtl";
            } else if (element.dataset.eduappAutoDirection === "rtl") {
              element.classList.remove(className);
              element.removeAttribute("dir");
              delete element.dataset.eduappAutoDirection;
            }
          };

          const prepareEditable = (element) => {
            const refresh = () => setAutomaticDirection(
              element,
              "eduapp-arabic-input",
              isArabicDominant(element.value)
            );
            refresh();
            if (element.dataset.eduappArabicListener !== "1") {
              element.addEventListener("input", refresh, true);
              element.addEventListener("change", refresh, true);
              element.dataset.eduappArabicListener = "1";
            }
          };

          const prepareStatic = (element) => setAutomaticDirection(
            element,
            element.matches('[role="gridcell"], td, th, [role="option"]')
              ? "eduapp-arabic-input"
              : "eduapp-arabic-block",
            isArabicDominant(element.textContent)
          );

          const scan = () => {
            rootDoc.querySelectorAll("textarea, input").forEach(prepareEditable);
            rootDoc.querySelectorAll(
              '[data-testid="stMarkdownContainer"] p, '
              + '[data-testid="stMarkdownContainer"] li, '
              + '[data-testid="stAlert"] p, '
              + '.track-box, blockquote, [role="gridcell"], td, th, [role="option"]'
            ).forEach(prepareStatic);
          };

          let scanQueued = false;
          const queueScan = () => {
            if (scanQueued) return;
            scanQueued = true;
            rootWin.requestAnimationFrame(() => {
              scanQueued = false;
              scan();
            });
          };

          scan();
          if (!rootWin.__eduappArabicLayoutObserver) {
            const observer = new rootWin.MutationObserver(queueScan);
            observer.observe(rootDoc.documentElement, {
              childList: true,
              subtree: true,
              characterData: true,
            });
            rootWin.__eduappArabicLayoutObserver = observer;
          }
        })();
        </script>
        """,
        height=0,
        width=0,
    )

