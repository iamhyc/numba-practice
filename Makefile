
all:
	pip3 install --user scipy
	pip3 install --user numpy
	pip3 install --user numba
	pip3 install --user matplotlib

run:
	mkdir -p logs
	python3 ./online_main.py

clean:
	rm -f logs/*.npz
	rm -f traces/*.npz