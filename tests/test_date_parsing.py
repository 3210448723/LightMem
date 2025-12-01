import unittest
from datetime import datetime
from dateutil import parser

class TestDateutilParsing(unittest.TestCase):
    
    def parse_timestamp(self, raw_ts: str) -> datetime:
        """
        模拟生产环境中的解析逻辑：使用 fuzzy=True 来忽略干扰词（如 on, (Sat) 等）
        """
        try:
            # fuzzy=True 是关键，它允许解析器跳过无法识别的字符（比如 'on', '(Sat)'）
            return parser.parse(raw_ts, fuzzy=True)
        except Exception as e:
            raise ValueError(f"Parsing failed for: {raw_ts}") from e

    def test_target_format_english(self):
        """测试你特别要求的格式: 1:56 pm on 8 May, 2023"""
        raw = "1:56 pm on 8 May, 2023"
        expected = datetime(2023, 5, 8, 13, 56) # 注意 1:56 pm 转换成了 13:56
        
        result = self.parse_timestamp(raw)
        self.assertEqual(result, expected, "未能正确解析 '1:56 pm on ...' 格式")
        print(f"✅ pass: {raw} -> {result}")

    def test_original_format_with_weekday(self):
        """测试原有格式（包含括号内的星期）"""
        # dateutil 的 fuzzy=True 会自动忽略括号内的 (Sat)
        raw = "2023/05/20 (Sat) 00:44"
        expected = datetime(2023, 5, 20, 0, 44)
        
        result = self.parse_timestamp(raw)
        self.assertEqual(result, expected, "未能正确忽略干扰词 (Sat)")
        print(f"✅ pass: {raw} -> {result}")

    def test_ordinal_suffix(self):
        """测试带有 th, st, nd 后缀的日期（标准库无法处理，但 dateutil 可以）"""
        raw = "May 8th, 2023 10:00 AM"
        expected = datetime(2023, 5, 8, 10, 0)
        
        result = self.parse_timestamp(raw)
        self.assertEqual(result, expected)
        print(f"✅ pass: {raw} -> {result}")

    def test_complex_natural_language(self):
        """测试更复杂的自然语言混合"""
        # 即使文字顺序很乱，只要逻辑清晰，dateutil 通常也能猜对
        raw = "Total time: 2023-11-05 at 4:30pm" 
        expected = datetime(2023, 11, 5, 16, 30)
        
        result = self.parse_timestamp(raw)
        self.assertEqual(result, expected)
        print(f"✅ pass: {raw} -> {result}")

    def test_iso_format(self):
        """测试标准 ISO 格式"""
        raw = "2023-05-08T13:56:00"
        expected = datetime(2023, 5, 8, 13, 56)
        
        result = self.parse_timestamp(raw)
        self.assertEqual(result, expected)
        print(f"✅ pass: {raw} -> {result}")

if __name__ == '__main__':
    print("Starting Dateutil Tests...\n")
    unittest.main(argv=['first-arg-is-ignored'], exit=False)