import os
import subprocess
import requests
import json

os.system('cd /home/mmadala/tmp/scripts/logs')
# run_keys=str(subprocess.check_output('ls -R /home/mmadala/tmp/scripts/logs/ | grep plugins/profile/.*',shell=True))
run_keys="b'./index_020200729/train/plugins/profile/2020_07_29_23_21_22:\\n./index_120200729/train/plugins/profile/2020_07_29_23_24_04:\\n./index_220200729/train/plugins/profile/2020_07_29_23_26_45:\\n./index_320200729/train/plugins/profile/2020_07_29_23_29_28:\\n'"
run_keys=[ entry[:-3].replace("/plugins/profile/","/") for entry in run_keys[2:-1].split("./")[1:]]
print(run_keys)
print("########################################")
result=[]
for key in run_keys:
    url='http://localhost:6006/data/plugin/profile/data?run='+key+'&tag=overview_page@&host=exxact-u1'
    response=requests.get(url=url)
    data=json.loads(response.text)
    performanceSummary= data[1]["p"]
    result.append((key.split("/")[0][:-8],performanceSummary["steptime_ms_average"]))
result.sort(key=lambda x:x[1])
print(result)


