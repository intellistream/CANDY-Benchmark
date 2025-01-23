### Simple benchmark tool under python

Test basic outcomes of delete, insert and search vectors, the result looks like the following
```shell
key,value,type
95%latency(Del),9426.000000,Double
95%latency(Insert),139137.000000,Double
QPS,644.554138,Double
recall,0.001000,Double
```
Advanced features like congestion and drop in C++ is not supported yet. Please make sure PyCandy is compliled and installed

#### Command line
```shell
python3 main.py <your config file>
```
### GUI (experimental)
You should install PyQT first, by 
```shell
pip PyQt5
```
Then, run the following
```shell
python3 mainWindow.py
```
This GUI allows you to load config, modify some commom stream and batch settings, and run. If you don't load extra config, you can still use the basic field to assign and run.