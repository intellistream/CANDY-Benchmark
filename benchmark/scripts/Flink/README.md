# Example Apps to Integrate CANDY into Apache Flink
## Compatablities and Limitations
 - Current only python
 - No shared mutable state
 - Please initilize candy index inside open() rather than __init__, as it is currently unable to serilize
 - Please follow PyTorch to see how tensor is handled by flink, and candy is totally the same
## Set up on ubuntu 22.04
After build candy and installed PyCANDY package, run the following forinstalling flink
```shell
sudo apt install python-is-python3
pip install apache-flink
```
## basic_search.ipynb

Just to show how CANDY works under python, no other dependencies

