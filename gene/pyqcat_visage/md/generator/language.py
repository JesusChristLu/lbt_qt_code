# -*- coding: utf-8 -*-

# This code is part of pyqcat-visage
#
# Copyright (c) 2017-2030 Origin Quantum Computing. All Right Reserved.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

# __date:         2022/10/18
# __author:       Lang Zhu
# __corporation:  OriginQuantum
"""
Base Generator.
_template = {"cn": "", "en": ""}

"""
instruction = {"cn": "简介", "en": "instruction"}
result = {"cn": "结果", "en": "result"}
unknown = {"cn": "未知", "en": "unknown"}
report = {"cn": "报告", "en": "report"}
experiment = {"cn": "实验", "en": "experiment"}
reason = {"cn": "原因", "en": "reason"}
execute_params = {"cn": "执行参数", "en": "execute params"}
id = {"cn": "ID", "en": "ID"}
executor = {"cn": "执行人", "en": "executor"}
runtime_start = {"cn": "开始时间", "en": "runtime start"}
runtime_end = {"cn": "结束时间", "en": "runtime end"}
sample = {"cn": "芯片编号", "en": "sample"}
chiller = {"cn": "制冷机编号", "en": "chiller ID"}
version = {"cn": "版本", "en": "version"}
file_path = {"cn": "文件路径", "en": "file path"}
name = {"cn": "名称", "en": "name"}
description = {"cn": "描述", "en": "description"}
official = {"cn": "标准流程", "en": "official"}
bit = {"cn": "比特", "en": "bit"}
volt_type = {"cn": "电压类型", "en": "volt type"}
params = {"cn": "参数", "en": "params"}

value = {"cn": "值", "en": "value"}
status = {"cn": "执行状态", "en": "status"}
success = {"cn": "成功", "en": "success"}
true = {"cn": "是", "en": "true"}
false = {"cn": "否", "en": "false"}
none = {"cn": "空", "en": "none"}

backtrack = {"cn": "回溯", "en": "backtrack"}

environment = {"cn": "执行环境", "en": "environment"}
bit_params = {"cn": "比特参数", "en": "bit params"}
bit_params_before = {"cn": "执行前比特参数", "en": "bit params before execute"}
bit_params_after = {"cn": "执行后比特参数", "en": "bit params after execute"}
straight_back_length = {"cn": "单次回溯节点数", "en": "single straight back length"}
results_plot = {"cn": "结果图", "en": "results plot"}
# Experiment
experimental_status = {"cn": "实验状态", "en": "experimental status"}
experiment_id = {"cn": "实验ID", "en": "experiment id"}
experiment_name = {"cn": "实验名称", "en": "experiment name"}
schedule_plot = {"cn": "时序图", "en": "schedule plot"}
execution_sequence_number = {"cn": "执行序号", "en": "execution sequence number"}
experiment_type = {"cn": "实验类型", "en": "experiment type"}
single = {"cn": "单层实验", "en": "single experiment"}
composite = {"cn": "复合实验", "en": "composite experiment"}
# Dag
dag_report = {"cn": "DAG报告", "en": "dag report"}
is_traceback = {"cn": "是否支持回溯", "en": "is support traceback"}
is_report = {"cn": "是否需要打印报告", "en": "is report"}
start_node = {"cn": "起始节点", "en": "start node"}
search_type = {"cn": "DAG遍历模式", "en": "search type"}
dag_execute_img = {"cn": "DAG执行流程图", "en": "DAG execution flowchart"}
node_map = {"cn": "节点映射表", "en": "node map"}

node_list = {"cn": "节点详情", "en": "node details list"}
working_dc = {"cn": "工作电压", "en": "working volt"}
dag_execute_path = {"cn": "DAG执行路径", "en": "dag execute path"}
happen_backtrack = {"cn": "期间发生回溯", "en": "happen backtrack"}
no_backtrack = {"cn": "未发生回溯", "en": "no backtrack"}
backtrack_child_details = {"cn": "本次回溯期间, 在结束逆向回溯后,正向执行节点的过程中,再次发生{}次子回溯过程.",
                           "en": "During this backtracking, {} second child backtracking occurred again during the forward node execution after the reverse backtracking was completed."}
backtrack_stats = {"cn": "回溯状态", "en": "backtrack stats"}
straight_back_length = {"cn": "回溯节点长度", "en": "straight back length"}
exist_backtrack_inside = {"cn": "内部是否发生子回溯", "en": "exist backtrack inside"}
parent_backtrack = {"cn": "上层回溯事件", "en": "parent backtrack"}
