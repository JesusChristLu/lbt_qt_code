# -*- coding: utf-8 -*-

import json

from pyqcat_visage.execute.dag.flash_dag import Dag

if __name__ == '__main__':
    dag_json_file = r'./dag_history_0922.json'
    with open(dag_json_file, mode='r', encoding='utf-8') as fp:
        dag_data = json.load(fp)

    dag_obj = Dag.from_dict(dag_data)

    dag_obj.add_node("fg")
    dag_obj.add_edge("ff", "fg")
    dag_obj.validate()

    # # has circle
    # dag_obj.add_edge("fg", "df")
    # dag_obj.validate()

    # dag_obj.remove_node("ef")
    # dag_obj.remove_edge("cd", "ef")

    bfs_list = dag_obj.bfs()
    dfs_list = dag_obj.dfs()
    ws_list = dag_obj.weight_search()

    print(f"bfs: {bfs_list}")
    print(f"dfs: {dfs_list}")
    print(f"ws: {ws_list}")

    dag_obj.run()

    # clear dag
    dag_obj.clear()

    print(dag_obj)
