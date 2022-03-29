docker run -d -h phd-0 --name phd --rm --gpus=all -v /data/phd:/data -w /data -p 9001:8888 phdgpu:1 bash /opt/codin/start_jupyter.sh
