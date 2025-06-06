import matplotlib.pyplot as plt
import qpandalite
import qpandalite.task.origin_qcloud as originq

finished, task_count = originq.query_all_task()

if finished != task_count:
    print(f'Unfinished / All = {task_count - finished} / {task_count}')
    exit(0)

taskid = originq.get_last_taskid()
results = originq.query_by_taskid_sync(taskid)

results = qpandalite.convert_originq_result(
    results, 
    style='list', 
    prob_or_shots='shots',
    key_style='dec'
)

figs, axs = plt.subplots(1, 5)

titles = ['only left', 
          'only right', 
          'left+right meas-left', 
          'left+right meas-right',
          'left+right meas-all']

for i, result in enumerate(results):
    axs[i].plot(result)
    axs[i].set_title(titles[i])

plt.show()
    

