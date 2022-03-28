sudo apt-get install nginx uwsgi uwsgi-plugin-python3





sudo nano /etc/nginx/sites-available/default
找到server，修改成下面这个样子
server {
        listen 80;
        server_name raspberry;
        location / {
                include uwsgi_params;
                uwsgi_pass  127.0.0.1:5000;
        }
}
sudo /etc/init.d/nginx start


创建一个叫config.ini的文件

[uwsgi]
module = myproject:app
socket = 127.0.0.1:5000
processes = 4
threads = 2


uwsgi --ini config.ini

pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html
