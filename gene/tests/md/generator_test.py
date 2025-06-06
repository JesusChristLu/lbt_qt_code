from pyqcat_visage.md.generator.generator_expriment import ExperimentGenerator
from pyqcat_visage.md.generator.generator_dag import DagGenerator

file_path = r"F:\md\{}"


def test_exp_generator():
    with open(
            r"D:\pyqcat_data\fivago\AmpOptimize\q2\2022-10-12\16.51.14\AmpOptimize(result).png",
            "rb") as f:
        result_img = {"data": f.read()}

    with open(
            r"D:\pyqcat_data\fivago\AmpOptimize\q2\2022-10-12\16.51.14\schedule\index=-1.png",
            "rb") as sf:
        schedule_img = {"data": sf.read()}

    exp_generator = ExperimentGenerator()
    exp_generator.option.language = "cn"
    exp_generator.option.is_time_schedule = True
    exp_generator.option.separation_img = True
    exp_generator.result_plot = result_img
    exp_generator.schedule_plot = schedule_img

    exp_generator.exp_params = {
        'show_result': True,
        'crosstalk_dict': None,
        'username': None,
        'sample': None,
        'env_name': None,
        'simulator_data_path': None,
        'simulator': False,
        'bind_dc': True,
        'bind_drive': True,
        'bind_probe': True,
        'file_flag': 0,
        'multi_readout_channels': None,
        'repeat': 1000,
        'data_type': 'amp_phase',
        'enable_one_sweep': False,
        'register_pulse_save': False,
        'schedule_flag': True,
        'schedule_save': True,
        'schedule_measure': True,
        'schedule_type': 'envelop',
        'schedule_show_measure': 150,
        'schedule_show_real': True,
        'schedule_index': -1,
        'measure_bits': None,
        'save_label': None,
        'is_dynamic': 1,
        'fidelity_matrix': None,
        'loop_num': 1,
        'iq_flag': False,
        'ac_prepare_time': 0,
        'readout_point_model': 'FlatTopGaussian',
        'idle_qubits': [],
        'freq_list': None,
        'drive_power': None,
        'z_amp': None,
        'use_square': True,
        'band_width': 50,
        'fine_flag': False,
        'rough_window_length': 7,
        'rough_freq_distance': 70,
        'fine_window_length': 11,
        'fine_freq_distance': 80,
        'xpulse_params': {
            'time': 5000,
            'offset': 15,
            'amp': 1.0,
            'detune': 0,
            'freq': 466.667
        },
        'zpulse_params': {
            'time': 5100,
            'amp': 0.0,
            'sigma': 3,
            'fast_m': True
        }
    }
    exp_generator.execute()

    file_name = f"exp_generator({exp_generator.option.language}).md"

    with open(file_path.format(file_name), "w", encoding="utf-8") as f:
        f.write(exp_generator.markdown)


def test_dag_generator():
    dag = DagGenerator()
    dag.execute()
    with open(f"dag_generator({dag.option.language}).md",
              "w",
              encoding="utf-8") as f:
        f.write(dag.markdown)


if __name__ == '__main__':
    test_exp_generator()
    # test_dag_generator()