# -*- coding: utf-8 -*-

import json

from pyqcat_visage.execute.run import run

if __name__ == '__main__':
    json_file = "run_dag.json"
    with open(json_file, mode='r', encoding='utf-8') as fp:
        data = json.load(fp)

    run(data)
