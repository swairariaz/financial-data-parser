import unittest
from format_parser import FormatParser

class TestFormatParser(unittest.TestCase):

    def test_parse_amount(self):
        samples = {
            "$1,234.56": 1234.56,
            "(2,500.00)": -2500.00,
            "€1.234,56": 1234.56,
            "1.5M": 1500000,
            "₹1,23,456": 123456,
            "1234.56-": -1234.56
        }
        for inp, expected in samples.items():
            self.assertAlmostEqual(FormatParser.parse_amount(inp), expected, places=2)

    def test_parse_date(self):
        samples = {
            "12/31/2023": "2023-12-31",
            "2023-12-31": "2023-12-31",
            "Q4 2023": "2023-10-01",
            "Dec-23": "2023-12-01",
            "44927": "2023-01-01"
        }
        for inp, expected in samples.items():
            dt = FormatParser.parse_date(inp)
            self.assertEqual(str(dt.date()), expected)

if __name__ == '__main__':
    unittest.main()
