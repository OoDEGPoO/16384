/* Autores:
 *	Daniel López Moreno
 *	Diego-Edgar Gracia Peña
 * Enunciado:
 *	Juego de 16384
 *		Versión en CUDA del Juego 2048
 *
 * Sin Bloques ni Memoria Compartida
 */

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

const int WS = 6;
int TILE_WIDTH = 0;

//	Ejemplo de como quedaría la matriz

//		-	-	N	-	-
//	|	00	01	02	03	04
//	M	10	11	12	13	14
//	|	20	21	22	23	24

//M es el eje Y
//N es el eje X

//------------------------------------------- Host ------------------------------------------------

void imprimeMatriz(int *v, int m, int n) {//( m * n )
	int i, j, x;
	int ws;//numero de espacios de caracteres por casilla
	printf("\n");
	for (i = 0; i < m; i++) {//recorremos eje m
		for (j = 0; j < n; j++) {//recorremos eje n
			ws = WS;
			x = v[i*n + j];

			//No se consideran numeros negativos, y el límite son 6 dígitos (que no se alcanzan)

			do {//Se ocupa un hueco por digito del numero
				ws--;
				x = x / 10;
			} while (x > 0);

			printf("%d", v[i*n + j]);//imprimimos el numero
			while (ws > 0) {//y ocupamos el resto de huecos con espacios en blanco
				printf(" ");
				ws--;
			}
		}
		printf("\n");
	}
}

//	Introduce en la matriz de juego un nuevo numero
//	-	*m matriz de Juego, WidthM y WidthN dimensiones de columna y fila
//	-	x e y, coordenadas donde se intenta introducir el elemento "set", si ya hay un elemento (!= 0), no se introduce y devuelve false
bool introNum(int *m, int WidthM, int WidthN, int x, int y, int set) {
	//comprobación de que esté dentro
	if (x < WidthN && y < WidthM) {

		if (m[y*WidthN + x] == 0) {
			m[y*WidthN + x] = set;
			return true;
		}

	}
	
	return false;
}

bool checkMatrizBool(bool *b, int m, int n) {
	bool out = false;
	int i, j;

	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			out = out || b[i*n + j];
		}
	}

	return out;
}

//-------------------------------------------------------------------------------------------------

//------------------------------------------ Device -----------------------------------------------

//	Inicializador de la matriz de juego
//	-	*m Matriz en forma vectorial con la que se trabaja, WidthM y WidthN su tamaño de columna y fila
//	-	x e y, las coordenadas del elemento que se iniciará al menor entero positivo válido
__global__ void Inicializador(int *m, int WidthM, int WidthN, int x, int y) {
	//obtención id del hilo
	int idBx = blockIdx.x;	int idBy = blockIdx.y;
	int idTx = threadIdx.x;	int idTy = threadIdx.y;

	int id_fil = idBy * TILE_WIDTH + idTy;//coordenada y
	int id_col = idBx * TILE_WIDTH + idTx;//coordenada x

	if (id_fil < WidthM && id_col < WidthN) {//Comprobación de que el hilo esté dentro de los límites
		if (x == id_col && y == id_fil) m[id_fil*WidthN + id_col] = 2;
		else m[id_fil*WidthN + id_col] = 0;
	}
}

//	Inicializador de matrices booleanas
//	-	*b Matriz vectorial de booleanos, WidthM y WidthN dimensiones de columna y fila
//	-	set el valor booleano a introducir
__global__ void iniBool(bool *b, int WidthM, int WidthN, bool set) {
	//obtención id del hilo
	int idBx = blockIdx.x;	int idBy = blockIdx.y;
	int idTx = threadIdx.x;	int idTy = threadIdx.y;

	int id_fil = idBy * TILE_WIDTH + idTy;//coordenada y
	int id_col = idBx * TILE_WIDTH + idTx;//coordenada x

	if (id_fil < WidthM && id_col < WidthN) {//Comprobación de que el hilo esté dentro de los límites

		b[id_fil*WidthN + id_col] = set;//damos al elemento correspondiente el valor indicado

	}
}

//	Ejecución de Movimiento
__global__ void ExMov(int *m, bool *b, int WidthM, int WidthN) {
	//obtención id del hilo
	int idBx = blockIdx.x;	int idBy = blockIdx.y;
	int idTx = threadIdx.x;	int idTy = threadIdx.y;

	int id_fil = idBy * TILE_WIDTH + idTy;//coordenada y
	int id_col = idBx * TILE_WIDTH + idTx;//coordenada x
}

//-------------------------------------------------------------------------------------------------

//--------------------------------- Inicialización Ejemplo de VS2015 --------------------------------------------

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int main()
{
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
