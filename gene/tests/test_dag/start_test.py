import json

from pyqcat_visage.execute.start import run_sub_process, execute
from pyQCat.invoker import Invoker, DataCenter
from pyqcat_visage.execute.dag.flash_dag import Dag


def test_by_local_file():
    json_file = "new_run_exp.json"
    # json_file = "run_dag.json"
    data = None
    with open(json_file, mode='r', encoding='utf-8') as fp:
        data = json.load(fp)

    currency_params = data.get("currency_params")
    base_params = data.get("base_params")

    run_type = currency_params.get("run_type")
    save_context = currency_params.get("save_context")

    context_data = {
        "currency_params": currency_params,
        "base_params": base_params
    }

    if run_type == "experiment":
        run_data = data.get("experiment")
    elif run_type == "dag":
        run_data = data.get("dag")

    run_sub_process(context=context_data, run_type=run_type, run_data=run_data, update_context=save_context,
                    simulator=False, simulator_base_path=r"F:\Simulator\monster\data")


def test_by_courier(run_id, sub_process: bool = False):
    context_data = {
        "currency_params": {
            "run_type": "experiment",
            "save_context": True,
            "conf_data": {
                "system": {
                    "sample": "test_chip",
                    "env_name": "D00_1019",
                    "point_label": "person_point",
                    "invoker_addr": "tcp://127.0.0.1:8088",
                    "baseband_freq": 566.667,
                    "qaio_type": 8,
                    "save_type": "local",
                    "local_root": "D:\\test",
                    "log_path": "",
                    "config_path": ""
                },
                "mongo": {
                    "inst_host": "127.0.0.1",
                    "inst_port": 27017
                },
                "minio": {
                    "s3_root": "10.10.24.76:9000",
                    "s3_access_key": "super",
                    "s3_secret_key": "80138013"
                },
                "qdc": {
                    "cpci_1": "192.168.2.100",
                    "cpci_2": "192.168.2.101",
                    "cpci_3": "192.168.2.102",
                    "cpci_4": "192.168.2.103",
                    "cpci_5": "192.168.2.104"
                }
            }
        },
        "base_params": {
            "instrument_type": 8,
            "qubit_names": [
                "q1"
            ],
            "coupler_names": [],
            "compensate_policy": "minimum",
            "environment_elements": [
                "q0",
                "q1",
                "q2",
                "q3",
                "c0",
                "c1",
                "c2"
            ],
            "discriminator": {
                "names": [],
                "union_flag": False,
                "union_names": []
            }
        }, }
    save_context = True
    run_type = "dag"
    kwargs = dict(context=context_data, run_type=run_type, run_id=run_id, update_context=save_context)
    if sub_process:
        run_sub_process(**kwargs)
    else:
        execute(**kwargs)


def register_dag():
    Invoker.load_account()
    db = DataCenter()
    data = db.query_dag_details("SingleQubitCalibrate")
    if data and data.get("code", None) == 200:
        dag = Dag.from_dict(data["data"])
        dag.register()
        dag_history_id = dag.id
        test_by_courier(dag_history_id, False)


if __name__ == '__main__':
    test_by_local_file()
    # register_dag()
