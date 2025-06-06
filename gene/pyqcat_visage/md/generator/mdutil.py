import base64
import os
from collections.abc import Iterable
from typing import List, Union

from pyqcat_visage.md.errors import IMGTypeErr

__all__ = [
    "StyleTable", "inset_img_base64", "inset_img_link", "internal_jump",
    "inset_table", "title", "strong_str", "strong_line", "cite_line",
    "code_block", "code_string", "list_block", "url_link", "divider"
]

SUPPORT_IMG_TYPE = ['png', 'jpg', 'jpeg']


class StyleTable:
    CENTER = ":---:"
    LEFT = ":---"
    RIGHT = "---:"


def check_str_ending(origin_str: str):
    if origin_str.endswith("\\"):
        origin_str += "\\"
        return origin_str


def image_bytes_to_base64string(image_bytes: bytes,
                                img_type: str = "png",
                                encode: str = 'utf-8') -> str:
    """
    converter image bytes to base64 string, use to add with markdown and html.
    Args:
        image_bytes: (bytes) the image bytes .
        img_type: (str) image type, usually is png and jpg.

    Returns:
        base64string
    """

    img_type = img_type.lower()
    if img_type not in SUPPORT_IMG_TYPE:
        raise IMGTypeErr(f"the msg type:{img_type}, is not support.")
    return base64.b64encode(image_bytes).decode(encoding=encode)


def img_read_from_disk(file_path: str):
    with open(file_path, "rb") as f:
        img_bytes = f.read()
        img_name = os.path.split(file_path)[1]
        return img_bytes, img_name


def inset_img_base64(img_link_path: str,
                     img_bytes: bytes,
                     img_type: str = "png") -> str:
    """

    :param png_link_path:
    :param png_bytes:
    :param png_type:
    :return:
    """

    img_base64 = image_bytes_to_base64string(img_bytes, img_type)
    return f"[{img_link_path}]:data:image/{img_type};base64," + img_base64 + "\n"


def inset_img_link(png_name: str,
                   link_path: str,
                   is_base64_local: bool = True):
    if is_base64_local:
        return f"![{png_name}][{link_path}]" + "\n"
    else:
        return f"![{png_name}]({link_path})" + "\n"


def url_link(url_link: str, display_name: str = None, title: str = None):
    url_link = check_str_ending(url_link)
    if display_name is None:
        display_name = url_link
    if title is not None:
        title = check_str_ending(title)
        return f"[{display_name}]({url_link} \"{title}\")" + "\n"
    else:
        return f"[{display_name}]({url_link})" + "\n"


def internal_jump(click_text: str,
                  link_to_id: str = None,
                  link_id: str = None, end_newline: bool = True) -> str:
    if link_to_id is not None:
        msg_doc = f"[{click_text}](#{link_to_id})"
    else:
        msg_doc = click_text
    link_str = f"<a id='{link_id}'></a>" if link_id is not None else ""
    if end_newline:
        return msg_doc + link_str + "\n"
    else:
        return msg_doc + link_str


def title(title: str, level: int):
    if level < 1:
        raise ValueError("level must in [1,6]")
    elif level < 7:
        title_doc = "#" * level
    else:
        title_doc = "#" * 6

    return f"{title_doc} {title}\n"


def inset_table(table_data: List[List],
                metre: List[str] = None,
                style: StyleTable = StyleTable.CENTER,
                default_metre: bool = False):
    def _pre_deal_table_data(table_data) -> List[List[str]]:
        if not table_data or table_data is None:
            raise ValueError("table data must list or dict.")

        if not isinstance(table_data, (list, dict)):
            raise TypeError("markdown table data must list or dict.")
        if isinstance(table_data, dict):
            return [[str(x), str(table_data[x])] for x in table_data.keys()]
        else:
            for i in range(len(table_data)):
                if isinstance(table_data[i], Iterable):
                    table_data[i] = [str(x) for x in table_data[i]]
                else:
                    table_data[i] = [str(table_data[i])]
            return table_data

    def _generator_raw(data_list: List[str]) -> str:
        raw_line = "|"
        for x in data_list:
            raw_line += " {data} |".format(data=x)
        return raw_line + "\n"

    def _tabulate(table_data, metre, style) -> str:
        table_str = ""

        table_str += _generator_raw(metre)
        table_str += _generator_raw([style for _ in range(len(metre))])
        for pre_line in table_data:
            table_str += _generator_raw(pre_line)
        return table_str

    if table_data is None:
        return ""
    try:
        table_data = _pre_deal_table_data(table_data)
    except ValueError or TypeError:
        return "\n\n"

    if metre is not None:
        if len(metre) != len(table_data[0]):
            raise ValueError(
                "table meter len must same with table_data pre len")
    else:
        # if default_metre:
        # metre = ["column-" + str(x) for x in range(len(table_data[0]))]
        metre = ["----" for x in range(len(table_data[0]))]
    # if not isinstance(style, StyleTable):
    #     style = StyleTable.CENTER

    return _tabulate(table_data=table_data, metre=metre, style=style) + "\n\n"


def strong_str(msg: Union[str, float, int],
               bold: bool = True,
               ltalic: bool = False) -> str:
    mark = ""
    if bold and ltalic:
        mark = "***"

    if bold and not ltalic:
        mark = "**"

    if ltalic and not bold:
        mark = "*"
    return mark + str(msg) + mark


def strong_line(strong_msg: str, text: str) -> str:
    return strong_str(strong_msg + ":") + " " + text + "\n"


def code_block(code_block: str, code_type: str = "python") -> str:
    mark = "```"
    code_block.replace("```", "\`\`\`")
    return mark + code_type + "\n" + code_block + "\n" + mark


def code_string(code_str: str, code_type: str = "") -> str:
    mark = "``"
    code_str.replace("``", "\`\`")
    return mark + code_type + "\n" + code_str + "\n" + mark


def cite_line(cite_str: str, level: int = 1):
    mark = ""
    if level > 1:
        mark = ">>"
    else:
        mark = ">"

    return mark + cite_str + "\n"


def list_block(block_msg: Union[str, List[Union[List, str]]],
               ordered_list: bool = True,
               level: int = 0,
               index: int = 1):
    retraction_block = "    "
    unordered_mark = "-"
    order_mark = f"{index}."

    list_res_str = ""
    if isinstance(block_msg, list):
        count = 1
        for x in block_msg:
            list_res_str += list_block(x,
                                       ordered_list=ordered_list,
                                       level=level + 1,
                                       index=count)
            count += 1
    else:
        if level <= 1:
            retraction = ""
        elif level < 5:
            retraction = retraction_block * level
        else:
            retraction = retraction_block * 5

        mark = order_mark if ordered_list else unordered_mark
        list_res_str += retraction + mark + " " + str(block_msg) + "\n"

    return list_res_str


def divider() -> str:
    """
    divider
    :return:
    """
    return "\n-----\n"
