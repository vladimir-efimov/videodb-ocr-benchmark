import unittest
from utils import calculate_num_occurrences  # замени на имя твоего файла


class TestCalculateNumOccurrences(unittest.TestCase):

    def test_basic_match(self):
        # Простые слова, есть совпадения
        key_words = "привет мир"
        ocr = "привет, мир! привет ещё раз."
        occ, total = calculate_num_occurrences(key_words, ocr)
        self.assertEqual(total, 2)
        self.assertEqual(occ, 2)  # оба слова найдены хотя бы раз

    def test_case_insensitive(self):
        # OCR в другом регистре
        key_words = "Привет Мир"
        ocr = "ПРИВЕТ, мир! ещё раз."
        occ, total = calculate_num_occurrences(key_words, ocr)
        self.assertEqual(total, 2)
        self.assertEqual(occ, 2)

    def test_punctuation_and_boundaries(self):
        # Пунктуация вокруг слов, проверка, что считаем целые слова
        key_words = "дом сад"
        ocr = "В доме и саду. Дом, сад! Домик."
        occ, total = calculate_num_occurrences(key_words, ocr)
        # "дом" и "сад" есть как отдельные слова; "домик" не считается за "дом"
        self.assertEqual(total, 2)
        self.assertEqual(occ, 2)

    def test_partial_word_not_counted(self):
        # Частичное совпадение не должно считаться
        key_words = "кот"
        ocr = "котик, котёнок, котофей."
        occ, total = calculate_num_occurrences(key_words, ocr)
        self.assertEqual(total, 1)
        self.assertEqual(occ, 0)  # "кот" как целое слово отсутствует

    def test_multiple_occurrences_count_once(self):
        # Слово встречается много раз, но считаем факт наличия (1)
        key_words = "окно"
        ocr = "Окно, окно! Окно… ещё окно."
        occ, total = calculate_num_occurrences(key_words, ocr)
        self.assertEqual(total, 1)
        self.assertEqual(occ, 1)

    def test_no_matches(self):
        key_words = "яблоко груша"
        ocr = "апельсин банан киви"
        occ, total = calculate_num_occurrences(key_words, ocr)
        self.assertEqual(total, 2)
        self.assertEqual(occ, 0)

    def test_empty_ground_truth(self):
        key_words = ""
        ocr = "любой текст"
        occ, total = calculate_num_occurrences(key_words, ocr)
        self.assertEqual(total, 0)
        self.assertEqual(occ, 0)

    def test_empty_ocr_text(self):
        key_words = "слово"
        ocr = ""
        occ, total = calculate_num_occurrences(key_words, ocr)
        self.assertEqual(total, 1)
        self.assertEqual(occ, 0)

    def test_duplicate_words_in_ground_truth(self):
        # Дубли в ground_truth: считаем каждое слово отдельно,
        # но факт вхождения всё равно 1 на уникальное слово
        key_words = "мир мир"
        ocr = "Мир, мир! Ещё мир."
        occ, total = calculate_num_occurrences(key_words, ocr)
        self.assertEqual(total, 2)  # два слова в ground_truth
        self.assertEqual(occ, 2)    # оба вхождения «мир» считаются как найденные

    def test_numbers_and_mixed_tokens(self):
        # Проверка, что числа и смешанные токены обрабатываются корректно
        key_words = "123 abc"
        ocr = "abc, 123! abc123 не считается."
        occ, total = calculate_num_occurrences(key_words, ocr)
        self.assertEqual(total, 2)
        self.assertEqual(occ, 2)  # "123" и "abc" найдены как отдельные токены

    def test_unicode_and_cyrillic(self):
        # Кириллица и смешанный текст
        key_words = "Москва Питер"
        ocr = "в Москве, в Питере. Москва-сити не считается как Москва."
        occ, total = calculate_num_occurrences(key_words, ocr)
        self.assertEqual(total, 2)
        self.assertEqual(occ, 1)

    def test_function_call_recognition(self):
        # Проверка для фрагмениа кода
        key_words = "response requests post"
        ocr = "response = requests.post(\"{}\")"
        occ, total = calculate_num_occurrences(key_words, ocr)
        self.assertEqual(total, 3)
        self.assertEqual(occ, 3)

if __name__ == "__main__":
    unittest.main()