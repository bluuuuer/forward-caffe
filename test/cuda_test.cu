#include <cuda.h>
#include <iostream>

__global__ void mykernel(void) {}

int main(void) {
	mykernel<<<1,1>>>();
	cout << "Hello World!" << endl;
	return 0;	
}
