import unittest

from src.utils.normalize import normalize_text


class TestNormalizeText(unittest.TestCase):
    def test_non_string_input_returns_as_is(self):
        self.assertEqual(normalize_text(123), 123)
        self.assertEqual(normalize_text(None), None)
        self.assertEqual(normalize_text(["text"]), ["text"])

    def test_basic_string_unchanged(self):
        self.assertEqual(normalize_text("Hello world"), "Hello world")

    def test_unicode_nfkc_normalization(self):
        # ﬁ → fi
        self.assertEqual(normalize_text("ﬁle"), "file")

    def test_control_characters_replaced_with_space(self):
        text = "Hello\x00World\x1FTest\x7F"
        self.assertEqual(normalize_text(text), "Hello World Test")

    def test_non_breaking_space_replaced(self):
        text = "Hello\u00A0World"
        self.assertEqual(normalize_text(text), "Hello World")

    def test_multiple_whitespaces_collapsed(self):
        text = "Hello   \n\t   World"
        self.assertEqual(normalize_text(text), "Hello World")

    def test_leading_and_trailing_spaces_trimmed(self):
        text = "   Hello World   "
        self.assertEqual(normalize_text(text), "Hello World")

    def test_combined_normalization_pipeline(self):
        text = "\u00A0Hello\x00   ﬁle \n World\u00A0"
        self.assertEqual(normalize_text(text), "Hello file World")

    def test_empty_string(self):
        self.assertEqual(normalize_text(""), "")

    def test_string_with_only_whitespace(self):
        self.assertEqual(normalize_text(" \n\t "), "")


if __name__ == "__main__":
    unittest.main()
