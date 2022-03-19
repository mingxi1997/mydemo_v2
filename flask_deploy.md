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
#uwsgi启动时，所使用的地址和端口（这个是http协议的）
socket=127.0.0.1:5000
#指向网站目录
chdir=/home/xu/flask_deploy
#python 启动程序文件
wsgi-file=run.py
#python 程序内用以启动的application 变量名
callable=app
#处理器数
processes=4
#线程数
threads=2


uwsgi --ini config.ini

pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html
