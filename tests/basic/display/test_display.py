import pathlib
import sys

parent_dir = pathlib.Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(parent_dir))


from tablemage._src.display.print_utils import (
    len_ignore_format,
    color_text,
    bold_text,
)


def test_len_ignore_format():
    words = [
        "hello",
        "lorum ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor "
        "incididunt ut labore et dolore magna aliqua ut enim ad minim veniam quis "
        "nostrud",
        "random words",
        "93m random words",
    ]

    for word in words:
        true_len = len(word)
        assert len_ignore_format(word) == true_len
        assert len_ignore_format(bold_text(word)) == true_len
        assert len_ignore_format(color_text(word, "red")) == true_len
        assert len_ignore_format(color_text(word, "yellow")) == true_len
        assert (
            len_ignore_format(color_text(color_text(word, "blue"), "green")) == true_len
        )
        assert len_ignore_format(color_text(bold_text(word), "purple")) == true_len
        assert len_ignore_format(bold_text(color_text(word, "purple"))) == true_len
