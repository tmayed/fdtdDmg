.PHONY: clean

fdtd:
	h5c++ -std=c++17 -fopenmp -o fdtd_S24_moire_dnt fdtd_S24_moire_dnt.cpp ../hdf5c.cpp
	rm *.o

clean :
	rm *.o
