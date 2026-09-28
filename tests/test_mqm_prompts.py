import unittest

from modules.mqm_prompts import MQM_PROMPT_OPTIONS, build_mqm_instruction, get_mqm_prompt_spec, prompt_event


class MqmPromptTests(unittest.TestCase):
    def test_every_ui_option_has_a_prompt_specification(self):
        for option in MQM_PROMPT_OPTIONS:
            spec = get_mqm_prompt_spec(option)
            self.assertTrue(spec["instruction"])
            self.assertTrue(spec["mqm_category"])

    def test_unknown_option_falls_back_to_accuracy(self):
        self.assertEqual(get_mqm_prompt_spec("Unknown")["help_type"], "Accuracy / meaning")

    def test_register_instruction_preserves_learner_decision(self):
        instruction = build_mqm_instruction("Register / audience").lower()
        self.assertIn("diagnostic question", instruction)

    def test_prompt_event_captures_category_and_minimised_counts(self):
        event = prompt_event(
            help_type="Terminology",
            student_question="Which term fits this context?",
            draft_before_prompt="A short learner draft",
            response_text="Brief diagnostic guidance",
            status="responded",
        )
        self.assertEqual(event["mqm_category"], "terminology")
        self.assertEqual(event["draft_word_count_before_prompt"], 4)
        self.assertEqual(event["response_word_count"], 3)
        self.assertEqual(event["status"], "responded")


if __name__ == "__main__":
    unittest.main()
