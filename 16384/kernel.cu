
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
#include <conio.h>
#include <stdlib.h>
#include <Windows.h>

#include <fstream>
#include <iostream>
#include <string>
	using namespace std;

const int WS = 6;
int BAJO[] = { 2, 4, 8 };
int ALTO[] = { 2, 4 };
int VIDAS = 5;
char MODO[] = "-m";
char FICHERO[] = "16384.sav";

__device__ int TILE_WIDTH_M = 0;
__device__ int TILE_WIDTH_N = 0;


//	Ejemplo de como quedaría la matriz

//		-	-	N	-	-
//	|	00	01	02	03	04
//	M	10	11	12	13	14
//	|	20	21	22	23	24

//M es el eje Y
//N es el eje X

//------------------------------------------ Device -----------------------------------------------

//	Inicializador de la matriz de juego
//	-	*m Matriz en forma vectorial con la que se trabaja, WidthM y WidthN su tamaño de columna y fila
__global__ void Inicializador(int *m, int WidthM, int WidthN) {
	//obtención id del hilo
	/*int idBx = blockIdx.x;	int idBy = blockIdx.y;
	int idTx = threadIdx.x;	int idTy = threadIdx.y;

	int id_fil = idBy * TILE_WIDTH + idTy;//coordenada y
	int id_col = idBx * TILE_WIDTH + idTx;//coordenada x
	*/

	int id_fil = threadIdx.y, id_col = threadIdx.x;

	if (id_fil < WidthM && id_col < WidthN) {//Comprobación de que el hilo esté dentro de los límites
		m[id_fil*WidthN + id_col] = 0;
	}
}

//	Inicializador de matrices booleanas
//	-	*b Matriz vectorial de booleanos, WidthM y WidthN dimensiones de columna y fila
//	-	set el valor booleano a introducir
__global__ void iniBool(bool *b, int WidthM, int WidthN, bool set) {
	//obtención id del hilo
	/*int idBx = blockIdx.x;	int idBy = blockIdx.y;
	int idTx = threadIdx.x;	int idTy = threadIdx.y;

	int id_fil = idBy * TILE_WIDTH + idTy;//coordenada y
	int id_col = idBx * TILE_WIDTH + idTx;//coordenada x
	*/

	int id_fil = threadIdx.y, id_col = threadIdx.x;

	if (id_fil < WidthM && id_col < WidthN) {//Comprobación de que el hilo esté dentro de los límites

		b[id_fil*WidthN + id_col] = set;//damos al elemento correspondiente el valor indicado

	}
}

//---------------------- Dch -------------------------

//	Cada hilo busca una pareja para su elemento correspondiente, y si es viable, realiza la suma
//	-	Cada hilo recorre la matriz hacia la derecha buscando fichas como la suya viables para la suma
//	-	para ello, cuenta cuantas coincidencias hay, si el numero es congruente con 0 mod 2,
//	-	no se realizará ninguna acción por parte del hilo, si es congruente con 1 mod 2,
//	-	se multiplica por 2 el primer coincidente y se borra la ficha del hilo.
//	-	Las coincidencias deben de ser inmediatas, solo permitiendose el 0 entre las fichas (0 == vacio)
//	-	-	La puntuación se recoge en la matriz p
__global__ void SumaDch(int *m, int *p, int WidthM, int WidthN) {
	//obtención id del hilo
	/*int idBx = blockIdx.x;	int idBy = blockIdx.y;
	int idTx = threadIdx.x;	int idTy = threadIdx.y;

	int id_fil = idBy * TILE_WIDTH + idTy;//coordenada y
	int id_col = idBx * TILE_WIDTH + idTx;//coordenada x
	*/

	int id_fil = threadIdx.y, id_col = threadIdx.x;

	int ficha, c = 0, aux, i;

	//filtro de hilos
	if (id_fil < WidthM && id_col < WidthN) {
		ficha = m[id_fil*WidthN + id_col];

		//si la ficha está vacia, el hilo no buscará
		if (ficha != 0) {
			//Se realiza la busqueda hacia la dch
			for (i = id_col + 1; i < WidthN; i++) {
				aux = m[id_fil*WidthN + i];

				if (aux == ficha) c++;//contamos las coincidencias
				else if (aux != 0) i = WidthN;//No podemos emparejar saltandonos fichas
			}

			//	Si el numero de coincidencias es congruente con 1 mod 2
			//	se busca la primera coincidencia, se multiplica por 2 y se borra la ficha 
			//	Si fuese congruente con 0 mod 2, no debe acceder al for
			if ((c % 2) == 0) p[id_fil*WidthN + id_col] = 0;//	Si no opera, puntuación 0
			for (i = id_col + 1; i < WidthN && (c % 2) == 1; i++) {
				aux = m[id_fil*WidthN + i];
				if (aux == ficha) {
					m[id_fil*WidthN + i] = ficha * 2;
					m[id_fil*WidthN + id_col] = 0;
					p[id_fil*WidthN + id_col] = ficha * 2;//	Grabamos la puntuación obtenido con la suma
					c--;//Para que el bucle for termine
				}

				//	(Aclaración) Si estamos entrando en este bucle for,
				//		significa que se ha encontrado una pareja viable anteriormente
				//		por lo que no se filtra si se opera con una ficha no válida
			}
		}
		else p[id_fil*WidthN + id_col] = 0;
	}
}

//	Ejecución de Movimiento a la Derecha de las piezas
//	-	Cada hilo toma su ficha (si es distinta de 0) y busca espacios en blanco a su derecha
//	-	Cuando no encuentra más huecos en la matriz, intercambia su ficha con la del último hueco hallado
//	-	al ser 0, intercambia con una vacía, si no hubiese huecos a su derecha, la intercambia consigo mismo
//	-	-	Esta función debe ser llamada hasta que no devuelva ningún cambio en la Matriz de Juego
__global__ void exMovDch(int *m, bool *b, int WidthM, int WidthN) {
	//obtención id del hilo
	/*int idBx = blockIdx.x;	int idBy = blockIdx.y;
	int idTx = threadIdx.x;	int idTy = threadIdx.y;

	int id_fil = idBy * TILE_WIDTH + idTy;//coordenada y
	int id_col = idBx * TILE_WIDTH + idTx;//coordenada x
	*/

	int id_fil = threadIdx.y, id_col = threadIdx.x;

	int ficha, id_aux = id_col;

	//filtro de hilos
	if (id_fil < WidthM && id_col < WidthN) {
		ficha = m[id_fil*WidthN + id_col];

		if (ficha != 0) {//si es 0, no hay que hacer ningún movimiento
			for (int i = id_col + 1; i < WidthN; i++) {
				if (m[id_fil*WidthN + i] == 0) id_aux = i;//se va buscando huecos vacios
				else i = WidthN;//hasta toparse con otra ficha, entonces paramos la búsqueda
			}

			//Intercambiamos las fichas, aunque no se haya encontrado ningún hueco
			m[id_fil*WidthN + id_col] = m[id_fil*WidthN + id_aux];
			m[id_fil*WidthN + id_aux] = ficha;
		}

		//	Si no hay ningún movimiento de ficha en el hilo, será false
		//	de haberlo, será true
		b[id_fil*WidthN + id_col] = id_col != id_aux;
	}

	//	El resultado de m deberá ser la matriz con las fichas que se pudieran mover a la derecha, movidas,
	//	Y el de b todos los elementos a false, excepto los coincidentes con las fichas que se han podido mover
}

//---------------------- Izq -------------------------

//	Cada hilo busca una pareja para su elemento correspondiente, y si es viable, realiza la suma
//	-	Cada hilo recorre la matriz hacia la izquierda buscando fichas como la suya viables para la suma
//	-	para ello, cuenta cuantas coincidencias hay, si el numero es congruente con 0 mod 2,
//	-	no se realizará ninguna acción por parte del hilo, si es congruente con 1 mod 2,
//	-	se multiplica por 2 el primer coincidente y se borra la ficha del hilo.
//	-	Las coincidencias deben de ser inmediatas, solo permitiendose el 0 entre las fichas (0 == vacio)
//	-	-	La puntuación se recoge en la matriz p
__global__ void SumaIzq(int *m, int *p, int WidthM, int WidthN) {
	//obtención id del hilo
	/*int idBx = blockIdx.x;	int idBy = blockIdx.y;
	int idTx = threadIdx.x;	int idTy = threadIdx.y;

	int id_fil = idBy * TILE_WIDTH + idTy;//coordenada y
	int id_col = idBx * TILE_WIDTH + idTx;//coordenada x
	*/

	int id_fil = threadIdx.y, id_col = threadIdx.x;

	int ficha, c = 0, aux, i;

	//filtro de hilos
	if (id_fil < WidthM && id_col < WidthN) {
		ficha = m[id_fil*WidthN + id_col];
		//si la ficha está vacia, el hilo no buscará
		if (ficha != 0) {
			//Se realiza la busqueda hacia la izq
			for (i = id_col - 1; i >= 0; i--) {
				aux = m[id_fil*WidthN + i];

				if (aux == ficha) c++;//contamos las coincidencias
				else if (aux != 0) i = -1;//No podemos emparejar saltandonos fichas
			}

			//	Si el numero de coincidencias es congruente con 1 mod 2
			//	se busca la primera coincidencia, se multiplica por 2 y se borra la ficha 
			//	Si fuese congruente con 0 mod 2, no debe acceder al for
			if ((c % 2) == 0) p[id_fil*WidthN + id_col] = 0;
			for (i = id_col - 1; i >= 0 && (c % 2) == 1; i--) {
				aux = m[id_fil*WidthN + i];
				if (aux == ficha) {
					m[id_fil*WidthN + i] = ficha * 2;
					m[id_fil*WidthN + id_col] = 0;
					p[id_fil*WidthN + id_col] = ficha * 2;//	Grabamos la puntuación obtenido con la suma
					c--;//Para que el bucle for termine
				}

				//	(Aclaración) Si estamos entrando en este bucle for,
				//		significa que se ha encontrado una pareja viable anteriormente
				//		por lo que no se filtra si se opera con una ficha no válida
			}
		}
		else p[id_fil*WidthN + id_col] = 0;
	}
}

//	Ejecución de Movimiento a la Izquierda de las piezas
//	-	Cada hilo toma su ficha (si es distinta de 0) y busca espacios en blanco a su izquierda
//	-	Cuando no encuentra más huecos en la matriz, intercambia su ficha con la del último hueco hallado
//	-	al ser 0, intercambia con una vacía, si no hubiese huecos a su izquierda, la intercambia consigo mismo
//	-	-	Esta función debe ser llamada hasta que no devuelva ningún cambio en la Matriz de Juego
__global__ void exMovIzq(int *m, bool *b, int WidthM, int WidthN) {
	//obtención id del hilo
	/*int idBx = blockIdx.x;	int idBy = blockIdx.y;
	int idTx = threadIdx.x;	int idTy = threadIdx.y;

	int id_fil = idBy * TILE_WIDTH + idTy;//coordenada y
	int id_col = idBx * TILE_WIDTH + idTx;//coordenada x
	*/

	int id_fil = threadIdx.y, id_col = threadIdx.x;

	int ficha, id_aux = id_col;

	//filtro de hilos
	if (id_fil < WidthM && id_col < WidthN) {
		ficha = m[id_fil*WidthN + id_col];

		if (ficha != 0) {//si es 0, no hay que hacer ningún movimiento
			for (int i = id_col - 1; i >= 0; i--) {
				if (m[id_fil*WidthN + i] == 0) id_aux = i;//se va buscando huecos vacios
				else i = -1;//hasta toparse con otra ficha, entonces paramos la búsqueda
			}

			//Intercambiamos las fichas, aunque no se haya encontrado ningún hueco
			m[id_fil*WidthN + id_col] = m[id_fil*WidthN + id_aux];
			m[id_fil*WidthN + id_aux] = ficha;
		}
		

		//	Si no hay ningún movimiento de ficha en el hilo, será false
		//	de haberlo, será true
		b[id_fil*WidthN + id_col] = id_col != id_aux;
	}

	//	El resultado de m deberá ser la matriz con las fichas que se pudieran mover a la izquierda, movidas,
	//	Y el de b todos los elementos a false, excepto los coincidentes con las fichas que se han podido mover
}

//---------------------- Arb -------------------------

//	Cada hilo busca una pareja para su elemento correspondiente, y si es viable, realiza la suma
//	-	Cada hilo recorre la matriz hacia arriba buscando fichas como la suya viables para la suma
//	-	para ello, cuenta cuantas coincidencias hay, si el numero es congruente con 0 mod 2,
//	-	no se realizará ninguna acción por parte del hilo, si es congruente con 1 mod 2,
//	-	se multiplica por 2 el primer coincidente y se borra la ficha del hilo.
//	-	Las coincidencias deben de ser inmediatas, solo permitiendose el 0 entre las fichas (0 == vacio)
//	-	-	La puntuación se recoge en la matriz p
__global__ void SumaArb(int *m, int *p, int WidthM, int WidthN) {
	//obtención id del hilo
	/*int idBx = blockIdx.x;	int idBy = blockIdx.y;
	int idTx = threadIdx.x;	int idTy = threadIdx.y;

	int id_fil = idBy * TILE_WIDTH + idTy;//coordenada y
	int id_col = idBx * TILE_WIDTH + idTx;//coordenada x
	*/

	int id_fil = threadIdx.y, id_col = threadIdx.x;

	int ficha, c = 0, aux, i;

	//filtro de hilos
	if (id_fil < WidthM && id_col < WidthN) {
		ficha = m[id_fil*WidthN + id_col];

		//si la ficha está vacia, el hilo no buscará
		if (ficha != 0) {
			//Se realiza la busqueda hacia arriba
			for (i = id_fil - 1; i >= 0; i--) {
				aux = m[i*WidthN + id_col];

				if (aux == ficha) c++;//contamos las coincidencias
				else if (aux != 0) i = -1;//No podemos emparejar saltandonos fichas
			}

			//	Si el numero de coincidencias es congruente con 1 mod 2
			//	se busca la primera coincidencia, se multiplica por 2 y se borra la ficha 
			//	Si fuese congruente con 0 mod 2, no debe acceder al for
			if ((c % 2) == 0) p[id_fil*WidthN + id_col] = 0;
			for (i = id_fil - 1; i >= 0 && (c % 2) == 1; i--) {
				aux = m[i*WidthN + id_col];
				if (aux == ficha) {
					m[i*WidthN + id_col] = ficha * 2;
					m[id_fil*WidthN + id_col] = 0;
					p[id_fil*WidthN + id_col] = ficha * 2;//	Grabamos la puntuación obtenido con la suma
					c--;//Para que el bucle for termine
				}

				//	(Aclaración) Si estamos entrando en este bucle for,
				//		significa que se ha encontrado una pareja viable anteriormente
				//		por lo que no se filtra si se opera con una ficha no válida
			}
		}
		else p[id_fil*WidthN + id_col] = 0;
	}
}

//	Ejecución de Movimiento hacia Arriba de las piezas
//	-	Cada hilo toma su ficha (si es distinta de 0) y busca espacios en blanco por encima
//	-	Cuando no encuentra más huecos en la matriz, intercambia su ficha con la del último hueco hallado
//	-	al ser 0, intercambia con una vacía, si no hubiese huecos por encima, la intercambia consigo mismo
//	-	-	Esta función debe ser llamada hasta que no devuelva ningún cambio en la Matriz de Juego
__global__ void exMovArb(int *m, bool *b, int WidthM, int WidthN) {
	//obtención id del hilo
	/*int idBx = blockIdx.x;	int idBy = blockIdx.y;
	int idTx = threadIdx.x;	int idTy = threadIdx.y;

	int id_fil = idBy * TILE_WIDTH + idTy;//coordenada y
	int id_col = idBx * TILE_WIDTH + idTx;//coordenada x
	*/

	int id_fil = threadIdx.y, id_col = threadIdx.x;

	int ficha, id_aux = id_fil;

	//filtro de hilos
	if (id_fil < WidthM && id_col < WidthN) {
		ficha = m[id_fil*WidthN + id_col];

		if (ficha != 0) {//si es 0, no hay que hacer ningún movimiento
			for (int i = id_fil - 1; i >= 0; i--) {
				if (m[i*WidthN + id_col] == 0) id_aux = i;//se va buscando huecos vacios
				else i = -1;//hasta toparse con otra ficha, entonces paramos la búsqueda
			}

			//Intercambiamos las fichas, aunque no se haya encontrado ningún hueco
			m[id_fil*WidthN + id_col] = m[id_aux*WidthN + id_col];
			m[id_aux*WidthN + id_col] = ficha;
		}

		//	Si no hay ningún movimiento de ficha en el hilo, será false
		//	de haberlo, será true
		b[id_fil*WidthN + id_col] = id_fil != id_aux;
	}

	//	El resultado de m deberá ser la matriz con las fichas que se pudieran mover hacia arriba, movidas,
	//	Y el de b todos los elementos a false, excepto los coincidentes con las fichas que se han podido mover
}

//---------------------- Abj -------------------------

//	Cada hilo busca una pareja para su elemento correspondiente, y si es viable, realiza la suma
//	-	Cada hilo recorre la matriz hacia abajo buscando fichas como la suya viables para la suma
//	-	para ello, cuenta cuantas coincidencias hay, si el numero es congruente con 0 mod 2,
//	-	no se realizará ninguna acción por parte del hilo, si es congruente con 1 mod 2,
//	-	se multiplica por 2 el primer coincidente y se borra la ficha del hilo.
//	-	Las coincidencias deben de ser inmediatas, solo permitiendose el 0 entre las fichas (0 == vacio)
//	-	-	La puntuación se recoge en la matriz p
__global__ void SumaAbj(int *m, int *p, int WidthM, int WidthN) {
	//obtención id del hilo
	/*int idBx = blockIdx.x;	int idBy = blockIdx.y;
	int idTx = threadIdx.x;	int idTy = threadIdx.y;

	int id_fil = idBy * TILE_WIDTH + idTy;//coordenada y
	int id_col = idBx * TILE_WIDTH + idTx;//coordenada x
	*/

	int id_fil = threadIdx.y, id_col = threadIdx.x;

	int ficha, c = 0, aux, i;

	//filtro de hilos
	if (id_fil < WidthM && id_col < WidthN) {
		ficha = m[id_fil*WidthN + id_col];

		//si la ficha está vacia, el hilo no buscará
		if (ficha != 0) {
			//Se realiza la busqueda hacia abj
			for (i = id_fil + 1; i < WidthM; i++) {
				aux = m[i*WidthN + id_col];

				if (aux == ficha) c++;//contamos las coincidencias
				else if (aux != 0) i = WidthM;//No podemos emparejar saltandonos fichas
			}

			//	Si el numero de coincidencias es congruente con 1 mod 2
			//	se busca la primera coincidencia, se multiplica por 2 y se borra la ficha 
			//	Si fuese congruente con 0 mod 2, no debe acceder al for
			if ((c % 2) == 0)p[id_fil*WidthN + id_col] = 0;
			for (i = id_fil + 1; i < WidthM && (c % 2) == 1; i++) {
				aux = m[i*WidthN + id_col];
				if (aux == ficha) {
					m[i*WidthN + id_col] = ficha * 2;
					m[id_fil*WidthN + id_col] = 0;
					p[id_fil*WidthN + id_col] = ficha * 2;//	Grabamos la puntuación obtenido con la suma
					c--;//Para que el bucle for termine
				}

				//	(Aclaración) Si estamos entrando en este bucle for,
				//		significa que se ha encontrado una pareja viable anteriormente
				//		por lo que no se filtra si se opera con una ficha no válida
			}
		}
		else p[id_fil*WidthN + id_col] = 0;
	}
}

//	Ejecución de Movimiento hacia Abajo de las piezas
//	-	Cada hilo toma su ficha (si es distinta de 0) y busca espacios en blanco por debajo de ella
//	-	Cuando no encuentra más huecos en la matriz, intercambia su ficha con la del último hueco hallado
//	-	al ser 0, intercambia con una vacía, si no hubiese huecos por debajo, la intercambia consigo mismo
//	-	-	Esta función debe ser llamada hasta que no devuelva ningún cambio en la Matriz de Juego
__global__ void exMovAbj(int *m, bool *b, int WidthM, int WidthN) {
	//obtención id del hilo
	/*int idBx = blockIdx.x;	int idBy = blockIdx.y;
	int idTx = threadIdx.x;	int idTy = threadIdx.y;

	int id_fil = idBy * TILE_WIDTH + idTy;//coordenada y
	int id_col = idBx * TILE_WIDTH + idTx;//coordenada x
	*/

	int id_fil = threadIdx.y, id_col = threadIdx.x;

	int ficha, id_aux = id_fil;

	//filtro de hilos
	if (id_fil < WidthM && id_col < WidthN) {
		ficha = m[id_fil*WidthN + id_col];

		if (ficha != 0) {//si es 0, no hay que hacer ningún movimiento
			for (int i = id_fil + 1; i < WidthM; i++) {
				if (m[i*WidthN + id_col] == 0) id_aux = i;//se va buscando huecos vacios
				else i = WidthM;//hasta toparse con otra ficha, entonces paramos la búsqueda
			}

			//Intercambiamos las fichas, aunque no se haya encontrado ningún hueco
			m[id_fil*WidthN + id_col] = m[id_aux*WidthN + id_col];
			m[id_aux*WidthN + id_col] = ficha;
		}

		//	Si no hay ningún movimiento de ficha en el hilo, será false
		//	de haberlo, será true
		b[id_fil*WidthN + id_col] = id_fil != id_aux;
	}

	//	El resultado de m deberá ser la matriz con las fichas que se pudieran mover hacia abajo, movidas,
	//	Y el de b todos los elementos a false, excepto los coincidentes con las fichas que se han podido mover
}

//-------------------------------------------------------------------------------------------------

//------------------------------------------- Host ------------------------------------------------

enum Colores {	//Colores para el fondo y la fuente de la consola
	BLACK = 0,
	BLUE = 1,
	GREEN = 2,
	CYAN = 3,
	RED = 4,
	MAGENTA = 5,
	BROWN = 6,
	LGREY = 7,
	DGREY = 8,
	LBLUE = 9,
	LGREEN = 10,
	LCYAN = 11,
	LRED = 12,
	LMAGENTA = 13,
	YELLOW = 14,
	WHITE = 15
};

//	Cambia el color de fondo y de fuente de la consola
void Color(int fondo, int fuente) {

	HANDLE Consola = GetStdHandle(STD_OUTPUT_HANDLE);
	//Cálculo para convertir los colores al valor necesario
	int color_nuevo = fuente + (fondo * 16);
	//Aplicamos el color a la consola
	SetConsoleTextAttribute(Consola, color_nuevo);

}

//	Inicializador de la matriz de juego
//	-	*m Matriz en forma vectorial con la que se trabaja, WidthM y WidthN su tamaño de columna y fila
//	-	x e y, las coordenadas del elemento que se introducira con el valor indicado
bool IntroCasilla(int *m, int WidthN, int x, int y, int valor) {
	bool out = m[y*WidthN + x] == 0;

	if (out) m[y*WidthN + x] = valor;

	return out;
}

void obtenerCaracteristicas(int n_columnas, int n_filas) {
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);

	printf("Características de la tarjeta: \n");
	printf("Nombre: %s \n", prop.name);
	printf("Capabilities: %d.%d \n", prop.major, prop.minor);
	printf("Maximo de hilos por bloque: %d \n", prop.maxThreadsPerBlock);
	printf("Maximo de hilos por SM: %d \n", prop.maxThreadsPerMultiProcessor);
	printf("Maximo de memoria global: %zd \n", prop.totalGlobalMem);
	printf("Maximo de memoria compartida: %zd \n", prop.sharedMemPerBlock);
	printf("Maximo de registros: %d \n", prop.regsPerBlock);
	printf("Numero de multiprocesadores: %d \n", prop.multiProcessorCount);

	//Tamaño de la matriz en hilos y memoria
	printf("Numero de hilos de la matriz: %d \n", n_columnas*n_filas);
	printf("Cantidad de memoria utilizada por la matriz: %zd \n", n_columnas*n_filas * sizeof(int));

	if (prop.maxThreadsPerBlock < (n_columnas*n_filas)) {
		printf("No hay suficientes hilos disponibles para calcular la matriz\n");
		exit(-1);
	}
	if (prop.totalGlobalMem < (n_columnas*n_filas * sizeof(int))) {
		printf("El tamaño de memoria global es insuficiente para calcular la matriz \n");
		exit(-1);
	}
}

//	Carga los datos del Juego Guardados anteriormente
//	-	v Matriz de Juego, WidthM y WidthN el tamaó de la matriz de juego,
//	-	puntuacion de juego acumulada y s el nombre del archivo de guardado
bool cargaDatos(int *v, int *WidthM, int *WidthN, int *puntuacion, int *vidas, int *dificultad, char *s) {
	string linea; //buffer de entrada
	char *c; //cadena de char para transformación
	char *token; //tokens de la matriz
	int k; //contador de acceso a la matriz
	ifstream entrada(s, ios::binary); //Fichero de entrada
	bool out = entrada.is_open(); //Si ha logrado abrir el archivo

	if (out) {
		//	Carga de la matriz de juego
		if (getline(entrada, linea, ';')) {//	Toma hasta el ';'
			c = (char*)malloc(linea.size() * sizeof(char));
			strcpy(c, linea.c_str());//	Pasamos a char para trabajar con ello
			token = strtok(c, " ,");//	Extraemos los numeros con Tokens
			k = 0;	//	Contador a 0

			while (token != NULL) {
				v[k] = atoi(token);//	Introducimos el cada numero en la matriz obteniendo Tokens
				k++;
				token = strtok(NULL, " ,");
			}

			free((void *)c);//	liberamos la memoria reservada para el char *
		}
		//	Carga del numero de filas de la matriz de juego
		if (getline(entrada, linea, ';')) {
			c = (char*)malloc(linea.size() * sizeof(char));
			strcpy(c, linea.c_str());
			*WidthM = atoi(c);
			free((void *)c);
		}
		//	Carga del numero de columnas de la matriz de juego
		if (getline(entrada, linea, ';')) {
			c = (char*)malloc(linea.size() * sizeof(char));
			strcpy(c, linea.c_str());
			*WidthN = atoi(c);
			free((void *)c);
		}
		//	Carga de la puntuación
		if (getline(entrada, linea, ';')) {
			c = (char*)malloc(linea.size() * sizeof(char));
			strcpy(c, linea.c_str());
			*puntuacion = atoi(c);
			free((void *)c);
		}
		//	Carga el numero de vidas
		if (getline(entrada, linea, ';')) {
			c = (char*)malloc(linea.size() * sizeof(char));
			strcpy(c, linea.c_str());
			*vidas = atoi(c);
			free((void *)c);
		}
		//	Carga la dificultad
		if (getline(entrada, linea, ';')) {
			c = (char*)malloc(linea.size() * sizeof(char));
			strcpy(c, linea.c_str());
			*dificultad = atoi(c);
			free((void *)c);
		}
		//	Carga el modo
		if (getline(entrada, linea, ';')) {
			c = (char*)malloc(linea.size() * sizeof(char));
			strcpy(c, linea.c_str());
			MODO[0] = c[0];
			MODO[1] = c[1];
			free((void *)c);
		}

		entrada.close();//Cerramos el archivo
	}
	else fprintf(stderr, "Fallo al intentar abrir el archivo de guardado\n");

	return out;
}

//	Guarda los datos de juego en el archivo de guardado indicado
//	-	v matriz de juego, WidthM y WidthN dimensiones de la matriz de juego,
//	-	puntuación de la partida y nombre del archivo destino
void guardaDatos(int *v, int WidthM, int WidthN, int puntuacion, int vidas, int dificultad, char *s) {
	ofstream salida;//	Fichero de salida
	int Width = WidthM * WidthN;//	Numero de elementos de la matriz de juego
	salida.open(s);//	Abrimos el fichero
	if (salida.is_open()) {
		for (int i = 0; i < Width; i++) {	//	Recorremos la matriz y
			salida << v[i];					//	vamos grabando los elementos en el archivo
			if (i < Width - 1) salida << ",";//	separando con ','
		}
		//	El resto de los datos los grabamos separandolos con ';'
		salida << ";" << WidthM << ";" << WidthN << ";" << puntuacion << ";" << vidas << ";" << dificultad << ";" << MODO << ";";

		salida.close();//	Cerramos el archivo
	}
	else fprintf(stderr, "Fallo al intentar abrir el archivo de guardado\n");
}

//	Leemos de teclado y devolvemos un numero de salida en funcion de la tecla pulsada
int reconocerTeclado() {
	char tecla;
	int salida;
	bool flag = true;

	do {
		tecla = getch();

		if (tecla == 'p' || tecla == 'P') { salida = 0; flag = false;}
		if (tecla == 'w' || tecla == 'W') { salida = 1; flag = false;}
		if (tecla == 'a' || tecla == 'A') { salida = 2; flag = false;}
		if (tecla == 'd' || tecla == 'D') {salida = 3; flag = false;}
		if (tecla == 's' || tecla == 'S') {salida = 4; flag = false;}

		if (tecla == 'r' || tecla == 'R') {salida = 5; flag = false;}
		if (tecla == 'g' || tecla == 'G') {salida = 6; flag = false;}

		if (tecla == -32) {
			tecla = getch();
			if (tecla == 72) {salida = 1; flag = false;}
			if (tecla == 75) {salida = 2; flag = false;}
			if (tecla == 77) {salida = 3; flag = false;}
			if (tecla == 80) {salida = 4; flag = false;}
		}

		if (tecla == 13) {salida = 7; flag = false;}
	} while (flag);

	return salida;
}

//	Mostramos una introduccion
void mostrarMenuInicial() {

	printf(".----------------.  .----------------.  .----------------.  .----------------.  .----------------.\n");
	printf("| .--------------. || .--------------. || .--------------. || .--------------. || .--------------. |\n");
	printf("| |     __       | || |    ______    | || |    ______    | || |     ____     | || |   _    _     | |\n");
	printf("| |    /  |      | || |  .' ____ \\   | || |   / ____ `.  | || |   .' __ '.   | || |  | |  | |    | |\n");
	printf("| |    `| |      | || |  | |____\\_|  | || |   `'  __) |  | || |   | (__) |   | || |  | |__| |_   | |\n");
	printf("| |     | |      | || |  | '____`'.  | || |   _ | __ '.  | || |   .`____'.   | || |  |____   _|  | |\n");
	printf("| |    _| |_     | || |  | (____) |  | || |  | \\____) |  | || |  | (____) |  | || |      _| |_   | |\n");
	printf("| |   |_____|    | || |  '.______.'  | || |   \\______.'  | || |  `.______.'  | || |     |_____|  | |\n");
	printf("| |              | || |              | || |              | || |              | || |              | |\n");
	printf("| '--------------' || '--------------' || '--------------' || '--------------' || '--------------' |\n");
	printf("'----------------'  '----------------'  '----------------'  '----------------'  '----------------' \n\n");
	printf("                       Created by: Diego-Edgar Gracia & Daniel Lopez                                \n\n");
	printf("                                                                                                      ");

}

//	Mostramos menu de carga de datos
int menuCargaDatos() {
	bool flag = true;
	int sel = 2;//	0 Cargar	/	1 Nueva	/	2 Salir
	int tecla;

	do {
		system("cls");
		Color(WHITE, BLACK);
		printf("\tSe ha encontrado una partida anterior\n¿Deseas usar cargar esa partida?\n\n");
		switch (sel)
		{
		case 0: Color(LCYAN, BLACK); printf("- Cargar Partida\n");
			Color(BLACK, WHITE); printf("- Nueva Partida\n- Salir\n");
			break;
		case 1: Color(BLACK, WHITE); printf("- Cargar Partida\n");
			Color(LCYAN, BLACK); printf("- Nueva Partida\n");
			Color(BLACK, WHITE); printf("- Salir\n");
			break;
		case 2: Color(BLACK, WHITE); printf("- Cargar Partida\n- Nueva Partida\n");
			Color(LCYAN, BLACK); printf("- Salir\n");
			break;
		default:
			break;
		}

		switch (reconocerTeclado())
		{
		case 1: sel--;
			if (sel < 0) sel = 2;
			break;
		case 4: sel++;
			if (sel > 2) sel = 0;
			break;
		case 7: flag = false;
			break;
		default:
			break;
		}
	} while (flag);

	return sel;
}

//	Mostramos las opciones de pausa
void mostrarMenuPausa() {
	system("cls");
	Color(WHITE, BLACK);
	printf("                        PAUSA             ");
	Color(BLACK, WHITE);
	printf("\n\n");
	printf("Selecciona una opcion:\n");
	printf("\t R - Reanudar \n");
	printf("\t G - Guardar progreso y salir \n");
	printf("\t S - Salir sin guardar \n");
}

//	Imprime la matriz de juego
//	-	Recorre las filas de la matriz de juego
void imprimeMatriz(int *p, int *v, int m, int n) {//( m * n )
	int i, j, x;
	int ws;//numero de espacios de caracteres por casilla
	printf("\n");
	system("cls");
	Color(WHITE, BLACK);
	printf("\t-WASD y Flechas del Teclado para mover las fichas\n\t\-P para Pausa");
	Color(BLACK, WHITE);
	printf("\nPuntuacion: %d \n", *p);
	for (i = 0; i < m; i++) {//recorremos eje m
		for (j = 0; j < n; j++) {//recorremos eje n
			ws = WS;
			x = v[i*n + j];

			//No se consideran numeros negativos, y el límite son 6 dígitos (que no se alcanzan)

			do {//Se ocupa un hueco por digito del numero
				ws--;
				x = x / 10;
			} while (x > 0);

			switch (v[i*n + j]) {//	Modifica el color en el que se mostrarán los elementos
			case 0:
				Color(BLACK, RED);
				break;
			case 2:
				Color(WHITE, BLACK);
				break;
			case 4:
				Color(YELLOW, BLACK);
				break;
			case 8:
				Color(LMAGENTA, BLACK);
				break;
			case 16:
				Color(MAGENTA, BLACK);
				break;
			case 32:
				Color(BROWN, BLACK);
				break;
			case 64:
				Color(RED, BLACK);
				break;
			case 128:
				Color(LBLUE, BLACK);
				break;
			case 256:
				Color(BLUE, BLACK);
				break;
			case 512:
				Color(LGREEN, BLACK);
				break;
			case 1024:
				Color(GREEN, BLACK);
				break;
			case 2048:
				Color(LGREY, BLACK);
				break;
			case 4096:
				Color(DGREY, BLACK);
				break;
			case 8192:
				Color(CYAN, BLACK);
				break;
			case 16384:
				Color(WHITE, BLACK);
				break;
			default:
				Color(BLACK, WHITE);
				break;
			}

			printf("%d", v[i*n + j]);//imprimimos el numero
			while (ws > 0) {//y ocupamos el resto de huecos con espacios en blanco
				if (ws == 1) {
					Color(BLACK, WHITE);
				}
				printf(" ");
				ws--;
			}
		}
		printf("\n");
		
	}
	printf("\n");
	Color(WHITE, BLACK);
	printf("VIDAS:              ");
	Color(BLACK, WHITE);
	printf("\n");
	Color(WHITE, RED);
	for (int i = 0; i < VIDAS; i++) {
		printf(" <3 ");
	}
	Color(WHITE, BLACK);
	printf("\n");

	Color(BLACK, WHITE);
}

//Solo para pruebas
/*
void imprimeBooleanos(bool *v, int m, int n) {//( m * n )
int i, j;
bool x;
int ws;//numero de espacios de caracteres por casilla
printf("\n");
for (i = 0; i < m; i++) {//recorremos eje m
for (j = 0; j < n; j++) {//recorremos eje n
ws = WS;
x = v[i*n + j];
if (v[i*n + j]) { printf("True"); ws = ws - 4; }
else { printf("False"); ws = ws - 5; }
while (ws > 0) {//y ocupamos el resto de huecos con espacios en blanco
printf(" ");
ws--;
}
}
printf("\n");
}
}*/

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

//	Comprueba si hay al menos un elemento verdadero en la matriz
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

//	Comprueba si hay al menos una casilla vacia
bool checkLleno(int *v, int m, int n) {
	bool out = false;
	int i, j;

	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			out = out || (v[i*n + j] == 0);
		}
	}

	return out;
}

//	Suma la puntuación total de la matriz
//	-	Recorremos toda la matriz buscando la puntuación total que guarda
int sumaPuntuacion(int *p, int m, int n) {
	int out = 0;
	int i, j;

	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			out = out + p[i*n + j];
		}
	}

	return out;
}

void introSemilla(int *v, int WidthM, int WidthN, int dificultad) {
	int x, y, valor;

	//Metemos nuevas semillas despues de realizar cada movimiento
	if (dificultad == 1) {
		for (int i = 0; i < 15; i++) {
			do {
				x = rand() % WidthN;
				y = rand() % WidthM;
				valor = BAJO[rand() % 3];
			} while (!IntroCasilla(v, WidthN, x, y, valor) && checkLleno(v, WidthM, WidthN));
		}
	}
	else if (dificultad == 2) {
		for (int i = 0; i < 8; i++) {
			do {
				x = rand() % WidthN;
				y = rand() % WidthM;
				valor = BAJO[rand() % 2];
			} while (!IntroCasilla(v, WidthN, x, y, valor) && checkLleno(v, WidthM, WidthN));
		}
	}
}

//	Realizamos las sumas y los movimientos hacia arriba
//	-	v Matriz de juego, p UN SOLO ENTERO CON LA PUNTUACIÓN,
//	-	la propia función se encargará de obtenerlo
cudaError_t accionArriba(int *v, int *p, int WidthM, int WidthN) {
	//printf("accion arriba");
	cudaError_t cudaStatus;
	int *dev_v = 0, *dev_p = 0;
	bool *dev_b = 0;
	dim3 dimGrid(1, 1);
	dim3 dimBlock(WidthM, WidthN);

	int *h_p = (int*) malloc(WidthM * WidthN * sizeof(int));
	bool *b = (bool*) malloc(WidthM * WidthN * sizeof(bool));

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto FreeArb;
	}

	// Allocate GPU buffers
	cudaStatus = cudaMalloc((void**)&dev_v, WidthM * WidthN * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto FreeArb;
	}

	cudaStatus = cudaMalloc((void**)&dev_p, WidthM * WidthN * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto FreeArb;
	}

	cudaStatus = cudaMemcpy(dev_v, v, WidthM * WidthN * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto FreeArb;
	}

	//	Sumamos las fichas que se puedan juntar

	//printf("Sumamos");
	SumaArb << <dimGrid, dimBlock >> >(dev_v, dev_p, WidthM, WidthN);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "SumaArb launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto FreeArb;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching SumaArb!\n", cudaStatus);
		goto FreeArb;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(v, dev_v, WidthM * WidthN * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto FreeArb;
	}

	cudaStatus = cudaMemcpy(h_p, dev_p, WidthM * WidthN * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto FreeArb;
	}

	//printf(" sumando puntuacion ");
	*p = *p + sumaPuntuacion(h_p, WidthM, WidthN);

	cudaStatus = cudaMalloc((void**)&dev_b, WidthM * WidthN * sizeof(bool));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto FreeArb;
	}

	do {
		//printf("\nMovemos casillas ");
		//	Inicializamos la matriz de bool

		//	Rellena de False la matriz de booleanos
		//printf("bools");
		iniBool << <dimGrid, dimBlock >> > (dev_b, WidthM, WidthN, false);

		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "iniBool launch failed: %s\n", cudaGetErrorString(cudaStatus));
			goto FreeArb;
		}

		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching iniBool!\n", cudaStatus);
			goto FreeArb;
		}

		//	Movemos

		exMovArb << <dimGrid, dimBlock >> > (dev_v, dev_b, WidthM, WidthN);

		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "exMovArb launch failed: %s\n", cudaGetErrorString(cudaStatus));
			goto FreeArb;
		}

		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching esMovArb!\n", cudaStatus);
			goto FreeArb;
		}

		cudaStatus = cudaMemcpy(v, dev_v, WidthM * WidthN * sizeof(int), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			goto FreeArb;
		}

		cudaStatus = cudaMemcpy(b, dev_b, WidthM * WidthN * sizeof(bool), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			goto FreeArb;
		}

		//Mientras se haya movido una ficha se vuelve a ejecutar el movimiento
	} while (checkMatrizBool(b, WidthM, WidthN));

	//printf("\nTodo Movido");

FreeArb:
	cudaFree(dev_v);
	cudaFree(dev_p);
	cudaFree(dev_b);
	free(h_p);
	free(b);

	return cudaStatus;
}

//Realizamos las sumas y los movimientos hacia la izquierda
cudaError_t accionIzquierda(int *v, int *p, int WidthM, int WidthN) {
	//printf("accion izquierda");
	cudaError_t cudaStatus;
	int *dev_v = 0, *dev_p = 0;
	bool *dev_b = 0;
	dim3 dimGrid(1, 1);
	dim3 dimBlock(WidthM, WidthN);

	int *h_p = (int*)malloc(WidthM * WidthN * sizeof(int));
	bool *b = (bool*)malloc(WidthM * WidthN * sizeof(bool));

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto FreeIzq;
	}

	// Allocate GPU buffers
	cudaStatus = cudaMalloc((void**)&dev_v, WidthM * WidthN * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto FreeIzq;
	}

	cudaStatus = cudaMalloc((void**)&dev_p, WidthM * WidthN * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto FreeIzq;
	}

	cudaStatus = cudaMemcpy(dev_v, v, WidthM * WidthN * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto FreeIzq;
	}

	//	Sumamos las fichas que se puedan juntar

	//printf("Sumamos");
	SumaIzq << <dimGrid, dimBlock >> > (dev_v, dev_p, WidthM, WidthN);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "SumaIzq launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto FreeIzq;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching SumaIzq!\n", cudaStatus);
		goto FreeIzq;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(v, dev_v, WidthM * WidthN * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto FreeIzq;
	}

	cudaStatus = cudaMemcpy(h_p, dev_p, WidthM * WidthN * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto FreeIzq;
	}

	//printf(" sumando puntuacion ");
	*p = *p + sumaPuntuacion(h_p, WidthM, WidthN);

	cudaStatus = cudaMalloc((void**)&dev_b, WidthM * WidthN * sizeof(bool));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto FreeIzq;
	}

	do {
		//printf("\nMovemos casillas ");
		//	Inicializamos la matriz de bool

		//	Rellena de False la matriz de booleanos
		//printf("bools");
		iniBool << <dimGrid, dimBlock >> > (dev_b, WidthM, WidthN, false);

		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "iniBool launch failed: %s\n", cudaGetErrorString(cudaStatus));
			goto FreeIzq;
		}

		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching iniBool!\n", cudaStatus);
			goto FreeIzq;
		}

		//	Movemos

		exMovIzq << <dimGrid, dimBlock >> > (dev_v, dev_b, WidthM, WidthN);

		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "exMovIzq launch failed: %s\n", cudaGetErrorString(cudaStatus));
			goto FreeIzq;
		}

		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching exMovIzq!\n", cudaStatus);
			goto FreeIzq;
		}

		cudaStatus = cudaMemcpy(v, dev_v, WidthM * WidthN * sizeof(int), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			goto FreeIzq;
		}

		cudaStatus = cudaMemcpy(b, dev_b, WidthM * WidthN * sizeof(bool), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			goto FreeIzq;
		}

		//Mientras se haya movido una ficha se vuelve a ejecutar el movimiento
	} while (checkMatrizBool(b, WidthM, WidthN));

	//printf("\nTodo Movido");

FreeIzq:
	cudaFree(dev_v);
	cudaFree(dev_p);
	cudaFree(dev_b);
	free(h_p);
	free(b);

	return cudaStatus;
}

//Realizamos las sumas y los movimientos hacia la derecha
cudaError_t accionDerecha(int *v, int *p, int WidthM, int WidthN) {
	//printf("accion derecha");
	cudaError_t cudaStatus;
	int *dev_v = 0, *dev_p = 0;
	bool *dev_b = 0;
	dim3 dimGrid(1, 1);
	dim3 dimBlock(WidthM, WidthN);

	int *h_p = (int*)malloc(WidthM * WidthN * sizeof(int));
	bool *b = (bool*)malloc(WidthM * WidthN * sizeof(bool));

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto FreeDch;
	}

	// Allocate GPU buffers
	cudaStatus = cudaMalloc((void**)&dev_v, WidthM * WidthN * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto FreeDch;
	}

	cudaStatus = cudaMalloc((void**)&dev_p, WidthM * WidthN * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto FreeDch;
	}

	cudaStatus = cudaMemcpy(dev_v, v, WidthM * WidthN * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto FreeDch;
	}

	//	Sumamos las fichas que se puedan juntar

	//printf("Sumamos");
	SumaDch << <dimGrid, dimBlock >> > (dev_v, dev_p, WidthM, WidthN);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "SumaDch launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto FreeDch;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching SumaDch!\n", cudaStatus);
		goto FreeDch;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(v, dev_v, WidthM * WidthN * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto FreeDch;
	}

	cudaStatus = cudaMemcpy(h_p, dev_p, WidthM * WidthN * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto FreeDch;
	}

	//printf(" sumando puntuacion ");
	*p = *p + sumaPuntuacion(h_p, WidthM, WidthN);

	cudaStatus = cudaMalloc((void**)&dev_b, WidthM * WidthN * sizeof(bool));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto FreeDch;
	}

	do {
		//printf("\nMovemos casillas ");
		//	Inicializamos la matriz de bool

		//	Rellena de False la matriz de booleanos
		//printf("bools");
		iniBool << <dimGrid, dimBlock >> > (dev_b, WidthM, WidthN, false);

		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "iniBool launch failed: %s\n", cudaGetErrorString(cudaStatus));
			goto FreeDch;
		}

		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching iniBool!\n", cudaStatus);
			goto FreeDch;
		}

		//	Movemos

		exMovDch << <dimGrid, dimBlock >> > (dev_v, dev_b, WidthM, WidthN);

		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "exMovDch launch failed: %s\n", cudaGetErrorString(cudaStatus));
			goto FreeDch;
		}

		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching exMovDch!\n", cudaStatus);
			goto FreeDch;
		}

		cudaStatus = cudaMemcpy(v, dev_v, WidthM * WidthN * sizeof(int), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			goto FreeDch;
		}

		cudaStatus = cudaMemcpy(b, dev_b, WidthM * WidthN * sizeof(bool), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			goto FreeDch;
		}

		//Mientras se haya movido una ficha se vuelve a ejecutar el movimiento
	} while (checkMatrizBool(b, WidthM, WidthN));

	//printf("\nTodo Movido");

FreeDch:
	cudaFree(dev_v);
	cudaFree(dev_p);
	cudaFree(dev_b);
	free(h_p);
	free(b);

	return cudaStatus;
}

//Realizamos las sumas y los movimientos hacia abajo
cudaError_t accionAbajo(int *v, int *p, int WidthM, int WidthN) {
	//printf("accion abajo");
	cudaError_t cudaStatus;
	int *dev_v = 0, *dev_p = 0;
	bool *dev_b = 0;
	dim3 dimGrid(1, 1);
	dim3 dimBlock(WidthM, WidthN);

	int *h_p = (int*) malloc(WidthM * WidthN * sizeof(int));
	bool *b = (bool*) malloc(WidthM * WidthN * sizeof(bool));

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto FreeAbj;
	}

	// Allocate GPU buffers
	cudaStatus = cudaMalloc((void**)&dev_v, WidthM * WidthN * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto FreeAbj;
	}

	cudaStatus = cudaMalloc((void**)&dev_p, WidthM * WidthN * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto FreeAbj;
	}

	cudaStatus = cudaMemcpy(dev_v, v, WidthM * WidthN * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto FreeAbj;
	}

	//	Sumamos las fichas que se puedan juntar
	//printf("Sumamos");
	SumaAbj << <dimGrid, dimBlock >> > (dev_v, dev_p, WidthM, WidthN);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "SumaAbj launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto FreeAbj;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching SumaAbj!\n", cudaStatus);
		goto FreeAbj;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(v, dev_v, WidthM * WidthN * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto FreeAbj;
	}

	cudaStatus = cudaMemcpy(h_p, dev_p, WidthM * WidthN * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto FreeAbj;
	}

	//printf(" sumando puntuacion ");
	*p = *p + sumaPuntuacion(h_p, WidthM, WidthN);

	cudaStatus = cudaMalloc((void**)&dev_b, WidthM * WidthN * sizeof(bool));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto FreeAbj;
	}

	do {
		//printf("\nMovemos casillas ");
		//	Inicializamos la matriz de bool

		//	Rellena de False la matriz de booleanos
		//printf("bools");
		iniBool << <dimGrid, dimBlock >> > (dev_b, WidthM, WidthN, false);

		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "iniBool launch failed: %s\n", cudaGetErrorString(cudaStatus));
			goto FreeAbj;
		}

		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching iniBool!\n", cudaStatus);
			goto FreeAbj;
		}

		//	Movemos

		exMovAbj << <dimGrid, dimBlock >> > (dev_v, dev_b, WidthM, WidthN);

		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "exMovAbj launch failed: %s\n", cudaGetErrorString(cudaStatus));
			goto FreeAbj;
		}

		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching exMovAbj!\n", cudaStatus);
			goto FreeAbj;
		}

		cudaStatus = cudaMemcpy(v, dev_v, WidthM * WidthN * sizeof(int), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			goto FreeAbj;
		}

		cudaStatus = cudaMemcpy(b, dev_b, WidthM * WidthN * sizeof(bool), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			goto FreeAbj;
		}

		//Mientras se haya movido una ficha se vuelve a ejecutar el movimiento
	} while (checkMatrizBool(b, WidthM, WidthN));

	//printf("\nTodo Movido");

FreeAbj:
	cudaFree(dev_v);
	cudaFree(dev_p);
	cudaFree(dev_b);
	free(h_p);
	free(b);

	return cudaStatus;
}

//Salimos del juego sin guardar partida
void accionSalir() {
	printf("Salir \n");
	exit(-1);
}

//Volvemos al juego
void accionReanudar() {
	printf("Reanudar \n");
}

//Guardamos el progreso y salimos
void accionGuardarSalir(int *v, int WidthM, int WidthN, int puntuacion, int dificultad) {
	printf("Guardar y salir \n");
	guardaDatos(v, WidthM, WidthN, puntuacion, VIDAS, dificultad, FICHERO);
	exit(-1);
}

//	Ejecutamos una accion en funcion de la tecla pulsada
void accionPausa(int *v, int WidthM, int WidthN, int puntuacion, int dificultad) {
	int tecla;
	bool flag = true;

	do {
		tecla = reconocerTeclado();

		switch (tecla) {
			//Salir sin guardar
			case 4:
				accionSalir();
				break;
			//Reanudar
			case 5:
				accionReanudar();
				flag = false;
				break;
			//Guardar y salir
			case 6:
				accionGuardarSalir(v, WidthM, WidthN, puntuacion, dificultad);
				break;
		}
	} while (flag);
}

void modoAutomatico() {
	printf("Modo automatico");
}

void modoManual(int *v, int dificultad, int WidthM, int WidthN) {
	int teclaPulsada;
	int *p = (int*) malloc(sizeof(int));
	cudaError_t cudaStatus;

	*p = 0;


	//imprimeMatriz(p, v, WidthM, WidthN);
	//printf("Matriz Generada\n");

	introSemilla(v, WidthM, WidthN, dificultad);

	//imprimeMatriz(p, v, WidthM, WidthN);
	//printf("Semilla metida\n");

	do {

		imprimeMatriz(p, v, WidthM, WidthN);

		teclaPulsada = reconocerTeclado();
		//printf("%d", teclaPulsada);
		switch (teclaPulsada) {
			//Menu de Pausa
		case 0:
			mostrarMenuPausa();
			accionPausa(v, WidthM, WidthN, *p, dificultad);
			break;

			//Arriba
		case 1:
			cudaStatus= accionArriba(v, p, WidthM, WidthN);
			introSemilla(v, WidthM, WidthN, dificultad);
			break;

			//Izquierda
		case 2:
			cudaStatus = accionIzquierda(v, p, WidthM, WidthN);
			introSemilla(v, WidthM, WidthN, dificultad);
			break;

			//Derecha
		case 3:
			cudaStatus = accionDerecha(v, p, WidthM, WidthN);
			introSemilla(v, WidthM, WidthN, dificultad);
			break;

			//Abajo
		case 4:
			cudaStatus = accionAbajo(v, p, WidthM, WidthN);
			introSemilla(v, WidthM, WidthN, dificultad);
			break;

			//Defecto
		default:
			break;
		}

		//imprimeMatriz(p, v, WidthM, WidthN);

		//printf("\n - fin loop - ");
	} while (checkLleno(v, WidthM, WidthN));

	//printf("\nMatriz llena - fin de partida");

	if (!checkLleno(v, WidthM, WidthN)) {
		VIDAS--;
	}
	

//Morgan
FreeMan:
	free(v);
	free(p);
}

cudaError_t iniciaMatriz(int *v, int WidthM, int WidthN) {
	int *dev_v = 0;

	dim3 dimGrid(1, 1);
	dim3 dimBlock(WidthM, WidthN);

	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto FreeIni;
	}

	// Allocate GPU buffers
	cudaStatus = cudaMalloc((void**)&dev_v, WidthM * WidthN * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc iniciaMatriz failed!");
		goto FreeIni;
	}

	cudaStatus = cudaMemcpy(dev_v, v, WidthM * WidthN * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto FreeIni;
	}

	Inicializador << <dimGrid, dimBlock >> > (dev_v, WidthM, WidthN);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Inicializador launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto FreeIni;
	}
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching Inicializador!\n", cudaStatus);
		goto FreeIni;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(v, dev_v, WidthM * WidthN * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto FreeIni;
	}

FreeIni:
	cudaFree(dev_v);

	return cudaStatus;
}

int main(int argc, char** argv) {
	//Mostramos el menú inicial y procedemos a jugar
	mostrarMenuInicial();

	int *v = 0, WidthM, WidthN, punt, vidas, dificultad;

	if (argc < 5) {
		if (cargaDatos(v, &WidthM, &WidthN, &punt, &vidas, &dificultad, FICHERO)) printf("Datos anteriores Cargados\n");
	} else {
		printf("antonio");
		if (cargaDatos(v, &WidthM, &WidthN, &punt, &vidas, &dificultad, FICHERO)) {
			printf("hay datos\n");
			switch (menuCargaDatos()) {
			case 0:
				printf("cargo");
				break;
			case 1:
				printf("nueva");
				WidthM = atoi(argv[3]);
				WidthN = atoi(argv[4]);
				VIDAS = 5;
				dificultad = atoi(argv[2]);
				strcpy(MODO, argv[1]);
				v = (int*)malloc(WidthM*WidthN * sizeof(int));

				iniciaMatriz(v, WidthM, WidthN);
				break;
			case 2:
				exit(0);
				break;
			default:
				exit(-1);
				break;
			}
		}
		else {
			printf("nueva");
			WidthM = atoi(argv[3]);
			WidthN = atoi(argv[4]);
			VIDAS = 5;
			dificultad = atoi(argv[2]);
			strcpy(MODO, argv[1]);
			v = (int*)malloc(WidthM*WidthN * sizeof(int));

			iniciaMatriz(v, WidthM, WidthN);
		}
	}

		printf("\n\n%d, %d, %d, %d, %s\n", WidthM, WidthN, dificultad, VIDAS, MODO);
		getch();

		

	//Modo Automatico
	if (strcmp(MODO, "-a") == 0) {

		modoAutomatico();
	}

	//Modo Manual
	else if (strcmp(MODO, "-m") == 0) {
		do {
			modoManual(v, dificultad, WidthM, WidthN);
		} while (VIDAS>0);
	}

	free(v);
}


//-------------------------------------------------------------------------------------------------

//--------------------------------- Inicialización Ejemplo de VS2015 --------------------------------------------

/*cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
}

int main()
{
	int Width = 5;
	int M[5 * 5];

	for (int i = 0; i < 5; i++) {
		for (int j = 0; j < 5; j++) {
			M[i * 5 + j] = rand() % 10;
		}
	}
	for (int i = 0; i < 5; i++) {
		for (int j = 0; j < 5; j++) {
			printf("%d", M[i * 5 + j]);
			printf("\t");
		}
		printf("\n");
	}
	printf("colorines");

	imprimeMatriz(M, Width, Width);

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

	printf("{1,2,3,4,5} + {10,20,30,40,50} ={%d, %d, %d, %d, %d}\n",
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
	addKernel << <1, size >> > (dev_c, dev_a, dev_b);

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
}*/

