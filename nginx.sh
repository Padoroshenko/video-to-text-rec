sudo apt-get install build-essential libpcre3 libpcre3-dev libssl-dev git zlib1g-dev -y
sudo mkdir ~/build && cd ~/build
sudo git clone git://github.com/arut/nginx-rtmp-module.git
sudo wget http://nginx.org/download/nginx-1.14.1.tar.gz
sudo tar xzf nginx-1.14.1.tar.gz
cd nginx-1.14.1
sudo ./configure --with-http_ssl_module --add-module=../nginx-rtmp-module
sudo make
sudo make install

