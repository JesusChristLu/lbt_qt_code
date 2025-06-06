from pathlib import Path

from pyqcat_visage.md.converter import converter_md_to_html, html_to_pdf
from pyqcat_visage.md.converter.pdf import html_to_pdf2
from pyqcat_visage.md.generator import ExperimentGenerator
from pyqcat_visage.md.converter.extensions import BootStrapExtension
import os

from pyqcat_visage.md.converter.markdown import HtmlStyle


def converter_md_to_local(md_doc: str, doc_name: str = "demo", file_path: str = None):
    # with open("demo.md", "w", encoding='utf-8') as f:
    #     f.write(md_doc)
    #     print("md save.")

    # extensions = ['toc', 'tables', BootStrapExtension()]
    extensions = ['markdown.extensions.toc', 'markdown.extensions.tables', 'markdown.extensions.fenced_code',
                  BootStrapExtension()]
    md_html = converter_md_to_html(md_doc, extensions)
    if file_path is not None:
        file_doc = os.path.join(file_path, doc_name)
    else:
        file_doc = doc_name
    # css_list = ["notes-dark", "themeable-dark", "themeable-light", "torillic"]
    # css_list = ["test-dark"]
    css_list = ["white", "dark"]
    os.makedirs("html", exist_ok=True)
    os.makedirs("pdf", exist_ok=True)
    html = HtmlStyle()
    for css in css_list:
        html_data = html.html(md_html, css)
        file_suffix = css.split(".")[0]
        with open(f"./html/{file_doc}_{file_suffix}.html", "w", encoding='utf-8') as f:
            f.write(html_data)
            print(f"html save.{file_doc}.html 11")

        file_path = Path(f"./html/{file_doc}_{file_suffix}.html").absolute()
        # html_to_pdf2(file_path)
        res = html_to_pdf(html_data)
        pdf_path = str(file_path).replace(".html", ".pdf")
        with open(pdf_path, "wb")as fp:
            fp.write(res)


def read_local_md(doc: str = "demo.md", file_path: str = None):
    file_doc = doc
    if file_path is not None:
        file_doc = os.path.join(file_path, doc)
    with open(file_doc, encoding="utf-8") as f:
        md_doc = f.read()
        converter_md_to_local(md_doc, doc.split(".")[0], file_path)


def test_generator():
    pass


if __name__ == '__main__':
    read_local_md()
    # html_to_md()
    # floder_path = r"F:/md"
    # floder_files = [x for x in os.walk(floder_path)][0][2]
    # for x in floder_files:
    #     print(x)
    #     if x.split(".")[-1] != "md":
    #         floder_files.remove(x)
    # print(floder_files)
    # for x in floder_files:
    #     read_local_md(x, floder_path)

    # with open(os.path.join(floder_path, "demo.html"), encoding="utf-8") as f:
    #     doc = f.read()
    #     demo_pdf = html_to_pdf(doc)
    #     with open(os.path.join(floder_path, "demo2.pdf"),"wb") as f2:
    #         f2.write(demo_pdf)