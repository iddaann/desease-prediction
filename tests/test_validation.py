"""Small regression tests for API input validation and symptom normalization."""
import unittest
from pydantic import ValidationError

from api.api_schemas import SymptomRequest
from src.symptom_normalizer import normalize_symptoms


class SymptomRequestTests(unittest.TestCase):
    def test_request_strips_whitespace_and_deduplicates(self):
        request = SymptomRequest(symptoms=[" cough ", "cough", "nausea"])
        self.assertEqual(request.symptoms, ["cough", "nausea"])

    def test_empty_symptom_list_is_rejected(self):
        with self.assertRaises(ValidationError):
            SymptomRequest(symptoms=[])

    def test_blank_values_are_removed(self):
        request = SymptomRequest(symptoms=["   ", "cough"])
        self.assertEqual(request.symptoms, ["cough"])


class SymptomNormalizerTests(unittest.TestCase):
    def test_known_and_unknown_symptoms_are_separated(self):
        known, unknown = normalize_symptoms(
            ["headache", "not_a_real_symptom"], ["headache", "nausea"]
        )
        self.assertEqual(known, ["headache"])
        self.assertEqual(unknown, ["not_a_real_symptom"])

    def test_duplicate_symptoms_are_normalized_once(self):
        known, unknown = normalize_symptoms(
            ["headache", "headache"], ["headache", "nausea"]
        )
        self.assertEqual(known, ["headache"])
        self.assertEqual(unknown, [])


if __name__ == "__main__":
    unittest.main()
