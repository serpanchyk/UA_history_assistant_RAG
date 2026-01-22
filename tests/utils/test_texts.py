import unittest

from src.utils.texts import normalize_text, block_to_text, list_to_interval, get_textbook_source

class TestNormalizeText(unittest.TestCase):
    """Tests that normalize_text() consistently cleans and normalizes input text."""

    def test_non_string_input_returns_as_is(self):
        """Calls normalize_text() with non-string input and expects the value to be returned unchanged."""
        self.assertEqual(normalize_text(123), 123)
        self.assertEqual(normalize_text(None), None)
        self.assertEqual(normalize_text(["text"]), ["text"])

    def test_basic_string_unchanged(self):
        """Calls normalize_text() with a clean string and expects no modification."""
        self.assertEqual(normalize_text("Hello world"), "Hello world")

    def test_unicode_nfkc_normalization(self):
        """Calls normalize_text() to verify Unicode NFKC normalization is applied."""
        self.assertEqual(normalize_text("ﬁle"), "file")

    def test_control_characters_replaced_with_space(self):
        """Calls normalize_text() to ensure control characters are replaced with spaces."""
        text = "Hello\x00World\x1FTest\x7F"
        self.assertEqual(normalize_text(text), "Hello World Test")

    def test_remove_annoying_symbols(self):
        """Calls normalize_text() to verify removal of invalid or invisible symbols."""
        text = " Hello ­ World"
        self.assertEqual(normalize_text(text), "Hello World")

    def test_non_breaking_space_replaced(self):
        """Calls normalize_text() to ensure non-breaking spaces are converted to regular spaces."""
        text = "Hello\u00A0World"
        self.assertEqual(normalize_text(text), "Hello World")

    def test_multiple_whitespaces_collapsed(self):
        """Calls normalize_text() to verify collapsing of multiple whitespace characters."""
        text = "Hello   \n\t   World"
        self.assertEqual(normalize_text(text), "Hello World")

    def test_leading_and_trailing_spaces_trimmed(self):
        """Calls normalize_text() to ensure leading and trailing spaces are trimmed."""
        text = "   Hello World   "
        self.assertEqual(normalize_text(text), "Hello World")

    def test_combined_normalization_pipeline(self):
        """Calls normalize_text() to validate the full normalization pipeline on mixed input."""
        text = "\u00A0Hello\x00   ﬁle \n World\u00A0"
        self.assertEqual(normalize_text(text), "Hello file World")

    def test_empty_string(self):
        """Calls normalize_text() with an empty string and expects an empty result."""
        self.assertEqual(normalize_text(""), "")

    def test_string_with_only_whitespace(self):
        """Calls normalize_text() with whitespace-only input and expects an empty string."""
        self.assertEqual(normalize_text(" \n\t "), "")

    def test_remove_hyphenation(self):
        """Calls normalize_text() to ensure line-break hyphenation is removed correctly."""
        text = "He-\nllo World"
        self.assertEqual(normalize_text(text), "Hello World")


class BlockToTextTest(unittest.TestCase):
    """Tests that block_to_text() converts OCR blocks into valid normalized text."""

    def test_happy_path(self):
        """Calls block_to_text() with a valid OCR block and expects correctly concatenated text."""
        block = {'lines': [{'spans': [{'text': 'Hello'}, {'text': 'World!'}]},
                           {'spans': [{'text': 'I'}, {'text': 'like'}, {'text': 'ML!'}]},
                           {'spans': [{'text': 'Unchara-'}]},
                           {'spans': [{'text': 'cteristically'}]}]
                 }

        text = block_to_text(block)
        self.assertEqual(text, 'Hello World! I like ML! Uncharacteristically')

    def test_small_text(self):
        """Calls block_to_text() with insufficient content and expects None."""
        block = {'lines': [{'spans': [{'text': 'Hello'}]}]}
        self.assertIsNone(block_to_text(block))

    def test_empty_block(self):
        """Calls block_to_text() with empty spans and expects None."""
        block = {'lines': [{'spans': [{'text': ''}, {'text': ''}, {'text': ''}]},
                           {'spans': [{'text': ''}, {'text': ''}, {'text': ''}]},
                           {'spans': [{'text': ''}, {'text': ''}, {'text': ''}]},
                           {'spans': [{'text': ''}, {'text': ''}, {'text': ''}]}]
                 }
        self.assertIsNone(block_to_text(block))


class ListToIntervalTest(unittest.TestCase):
    """Tests that list_to_interval() converts lists to texts with commas."""

    def test_list_to_interval_with_empty_list(self):
         """Calls list_to_interval() with an empty list and expects an empty string."""

         self.assertEqual(list_to_interval([]), "")

    def test_list_to_interval_with_strings(self):
        """Calls list_to_interval() with a string and expects None."""

        self.assertEqual(list_to_interval(["Hello", "World"]), None)

    def test_list_to_interval_with_different_integers(self):
        """Calls list_to_interval() with different integers and expects an interval."""
        self.assertEqual(list_to_interval([123, 1, 2, 3,]), '1-123')

    def test_list_to_interval_with_the_same_integer(self):
        """Calls list_to_interval() with same integers one number."""
        self.assertEqual(list_to_interval([100, 100, 100]), '100')

    def test_list_to_interval_with_the_one_number(self):
        """Calls list_to_interval() with same integers one number."""
        self.assertEqual(list_to_interval(10.0), '10')
        self.assertEqual(list_to_interval(11), '11')
        self.assertEqual(list_to_interval('12'), '12')


class GetSourceTest(unittest.TestCase):
    """Tests that get_source() returns the correct source."""

    def test_get_source_returns_correct_source(self):
        """Calls get_source() and expects correct source."""
        self.assertEqual(get_textbook_source(6), "Історія України (поглиблений рівень) : підруч. для 9 класу закл. загал. серед. освіти / [В. С. Власов]. — Київ : Літера ЛТД, 2022. — 336 с. : іл.")

    def test_get_source_returns_none(self):
        """Calls get_source() with non existing doc id."""
        self.assertEqual(get_textbook_source(100), None)

if __name__ == "__main__":
    unittest.main()
