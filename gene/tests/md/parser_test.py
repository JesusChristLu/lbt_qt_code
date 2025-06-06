import os

from pyqcat_visage.md.parser.parser_dag import DagParser
from pyQCat.invoker import Invoker, DataCenter
print(Invoker.get_env())
print(Invoker.verify_account("test_02", "123456"))


def test_dag():
    dag_parser = DagParser(id="637d8077ad82c5ba7c53be33")
    # dag_parser = DagParser(id="6369bf38ec9b18b10de02587")
    dag_parser.generator_options.language = "cn"
    dag_parser.generator_options.detail = "detail"
    dag_parser.converter_options.hold_html = True
    dag_parser.converter_options.hold_pdf = True
    dag_parser.parser()
    print("dag parser end")

    with open(r"F:\md\dag_report.md", "w", encoding="utf-8") as f:
        f.write(dag_parser.generator.markdown)

    with open(r"F:\md\dag_report.html", "w", encoding="utf-8") as f:
        f.write(dag_parser.converter_obj.doc_html)

    with open(r"F:\md\dag_report.pdf", "wb") as f:
        f.write(dag_parser.converter_obj.doc_pdf)


def test_invoker():
    db = DataCenter()
    print("dag details:\n", db.query_dag_record(dag_id="6360d36178c96f5d55065e22"))
    # print("exp details:\n", db.query_exp_record(experiment_id="1663291551"))
    # res=  db.query_qcomponent(name="q3")


def test_execute():
    from pyqcat_visage.md import execute
    save_type = "pdf"
    res = execute(id="638561c9175e478dbd22921f", save_type=save_type)
    file_path = r"C:\Users\BY210156\Desktop"
    file_name = f"test.{save_type}"
    if save_type in ["md", "html"]:
        with open(os.path.join(file_path, file_name), "w", encoding="utf-8") as f:
            f.write(res)
    elif save_type == "pdf":
        with open(os.path.join(file_path, file_name), "wb") as f:
            f.write(res)


if __name__ == '__main__':
    # test_invoker()
    test_execute()
