
all:
	pip3 install --user scipy
	pip3 install --user numpy
	pip3 install --user numba
	pip3 install --user matplotlib

run:clean
	python3 ./online_main.py

clean:
	@rm -rf traces-*