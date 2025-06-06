from pyqcat_visage.md.parser.plot_dag import Dag
import json


def test_dag(dag_json: dict):
    dag_test = Dag(**dag_json)
    dag_test.parser()
    if dag_test.img:
        for img in dag_test.img:
            with open(f"dag{img}.png", "wb") as f:
                f.write(dag_test.img[img])

    print("dag details: \n", print(dag_test.g_backtrack))


def get_dag_json() -> dict:
    with open("dag_plot.json", "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == '__main__':
    test_dag(get_dag_json())
