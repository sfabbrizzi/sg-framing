from src.utils import clean_text


def test_clean_text():
    assert clean_text("Hello, World!") == "hello world"
    assert clean_text("Python is great!") == "python is great"
    assert clean_text("   Leading and trailing spaces   ") == "leading and trailing spaces"
    assert clean_text("_!!!$Special characters$!!!_") == "special characters"
    assert clean_text("Numbers 12345") == "numbers 12345"
    assert clean_text("hyph-ens") == "hyphens"
    assert clean_text("lower_case") == "lowercase"
