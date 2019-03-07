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
#include <Windows.h>

const int WS = 6;
const int TILE_WIDTH = 0;

//	Ejemplo de como quedaría la matriz

//		-	-	N	-	-
//	|	00	01	02	03	04
//	M	10	11	12	13	14
//	|	20	21	22	23	24

//M es el eje Y
//N es el eje X

//------------------------------------------- Host ------------------------------------------------

/*int main() {
	switch () {

		//Menu de Pausa
		case 0:

			break;

		//Arriba
		case 1:
			break;

		//Izquierda
		case 2:
			break;

		//Derecha
		case 3:
			break;

		//Abajo
		case 4:
			break;
		default:
			break;
	}
}*/

void mostrarMenu() {

	printf(".----------------.  .----------------.  .----------------.  .----------------.  .----------------.\n");
	printf("| .--------------. || .--------------. || .--------------. || .--------------. || .--------------. |\n");
	printf("| |     __       | || |    ______    | || |    ______    | || |     ____     | || |   _    _     | |\n");
	printf("| |    /  |      | || |  .' ____ \   | || |   / ____ `.  | || |   .' __ '.   | || |  | |  | |    | |\n");
	printf("| |    `| |      | || |  | |____\_|  | || |   `'  __) |  | || |   | (__) |   | || |  | |__| |_   | |\n");
	printf("| |     | |      | || |  | '____`'.  | || |   _ | __ '.  | || |   .`____'.   | || |  |____   _|  | |\n");
	printf("| |    _| |_     | || |  | (____) |  | || |  | \____) |  | || |  | (____) |  | || |      _| |_   | |\n");
	printf("| |   |_____|    | || |  '.______.'  | || |   \______.'  | || |  `.______.'  | || |     |_____|  | |\n");
	printf("| |              | || |              | || |              | || |              | || |              | |\n");
	printf("| '--------------' || '--------------' || '--------------' || '--------------' || '--------------' |\n");
	printf("'----------------'  '----------------'  '----------------'  '----------------'  '----------------' \n\n");
	printf("                       Created by: Diego-Edgar Gracia & Daniel Lopez                                \n");

}

enum Colores {
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

void Color(int Background, int Text) {

	HANDLE Console = GetStdHandle(STD_OUTPUT_HANDLE);
	//Cálculo para convertir los colores al valor necesario
	int New_Color = Text + (Background * 16);
	//Aplicamos el color a la consola
	SetConsoleTextAttribute(Console, New_Color);

}

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
			
			switch (v[i*n+j]) {
				case 0:
					Color(BLACK,BLACK);
					break;
				case 2:
					Color(WHITE,BLACK);
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
					Color(BLACK,WHITE);
					break;
			}

			printf("%d", v[i*n + j]);//imprimimos el numero
			while (ws > 0) {//y ocupamos el resto de huecos con espacios en blanco
				printf(" ");
				ws--;
			}
		}
		printf("\n");
		Color(BLACK,WHITE);
	}
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

//---------------------- Dch -------------------------

//	Cada hilo busca una pareja para su elemento correspondiente, y si es viable, realiza la suma
//	-	Cada hilo recorre la matriz hacia la derecha buscando fichas como la suya viables para la suma
//	-	para ello, cuenta cuantas coincidencias hay, si el numero es congruente con 0 mod 2,
//	-	no se realizará ninguna acción por parte del hilo, si es congruente con 1 mod 2,
//	-	se multiplica por 2 el primer coincidente y se borra la ficha del hilo.
//	-	Las coincidencias deben de ser inmediatas, solo permitiendose el 0 entre las fichas (0 == vacio)
__global__ void SumaDch(int *m, int WidthM, int WidthN) {
	//obtención id del hilo
	int idBx = blockIdx.x;	int idBy = blockIdx.y;
	int idTx = threadIdx.x;	int idTy = threadIdx.y;

	int id_fil = idBy * TILE_WIDTH + idTy;//coordenada y
	int id_col = idBx * TILE_WIDTH + idTx;//coordenada x

	int ficha, c = 0, aux, i;

	//filtro de hilos
	if (id_fil < WidthM && id_col < WidthN) {
		ficha = m[id_fil*WidthN + id_col];

		//si la ficha está vacia, el hilo no buscará
		if (ficha != 0) {
			//Se realiza la busqueda hacia la dch
			for (i = id_col+1; i < WidthN; i++) {
				aux = m[id_fil*WidthN + i];

				if (aux == ficha) c++;//contamos las coincidencias
				else if (aux != 0) i = WidthN;//No podemos emparejar saltandonos fichas
			}

			//	Si el numero de coincidencias es congruente con 1 mod 2
			//	se busca la primera coincidencia, se multiplica por 2 y se borra la ficha 
			//	Si fuese congruente con 0 mod 2, no debe acceder al for
			for (i = id_col + 1; i < WidthN && (c % 2) == 1; i++) {
				aux = m[id_fil*WidthN + i];
				if (aux == ficha) {
					m[id_fil*WidthN + i] = ficha * 2;
					m[id_fil*WidthN + id_col] = 0;
					c--;//Para que el bucle for termine
				}

				//	(Aclaración) Si estamos entrando en este bucle for,
				//		significa que se ha encontrado una pareja viable anteriormente
				//		por lo que no se filtra si se opera con una ficha no válida
			}
		}
	}
}

//	Ejecución de Movimiento a la Derecha de las piezas
//	-	Cada hilo toma su ficha (si es distinta de 0) y busca espacios en blanco a su derecha
//	-	Cuando no encuentra más huecos en la matriz, intercambia su ficha con la del último hueco hallado
//	-	al ser 0, intercambia con una vacía, si no hubiese huecos a su derecha, la intercambia consigo mismo
//	-	-	Esta función debe ser llamada hasta que no devuelva ningún cambio en la Matriz de Juego
__global__ void exMovDch(int *m, bool *b, int WidthM, int WidthN) {
	//obtención id del hilo
	int idBx = blockIdx.x;	int idBy = blockIdx.y;
	int idTx = threadIdx.x;	int idTy = threadIdx.y;

	int id_fil = idBy * TILE_WIDTH + idTy;//coordenada y
	int id_col = idBx * TILE_WIDTH + idTx;//coordenada x

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
__global__ void SumaIzq(int *m, int WidthM, int WidthN) {
	//obtención id del hilo
	int idBx = blockIdx.x;	int idBy = blockIdx.y;
	int idTx = threadIdx.x;	int idTy = threadIdx.y;

	int id_fil = idBy * TILE_WIDTH + idTy;//coordenada y
	int id_col = idBx * TILE_WIDTH + idTx;//coordenada x

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
			for (i = id_col - 1; i <= 0 && (c % 2) == 1; i--) {
				aux = m[id_fil*WidthN + i];
				if (aux == ficha) {
					m[id_fil*WidthN + i] = ficha * 2;
					m[id_fil*WidthN + id_col] = 0;
					c--;//Para que el bucle for termine
				}

				//	(Aclaración) Si estamos entrando en este bucle for,
				//		significa que se ha encontrado una pareja viable anteriormente
				//		por lo que no se filtra si se opera con una ficha no válida
			}
		}
	}
}

//	Ejecución de Movimiento a la Izquierda de las piezas
//	-	Cada hilo toma su ficha (si es distinta de 0) y busca espacios en blanco a su izquierda
//	-	Cuando no encuentra más huecos en la matriz, intercambia su ficha con la del último hueco hallado
//	-	al ser 0, intercambia con una vacía, si no hubiese huecos a su izquierda, la intercambia consigo mismo
//	-	-	Esta función debe ser llamada hasta que no devuelva ningún cambio en la Matriz de Juego
__global__ void exMovIzq(int *m, bool *b, int WidthM, int WidthN) {
	//obtención id del hilo
	int idBx = blockIdx.x;	int idBy = blockIdx.y;
	int idTx = threadIdx.x;	int idTy = threadIdx.y;

	int id_fil = idBy * TILE_WIDTH + idTy;//coordenada y
	int id_col = idBx * TILE_WIDTH + idTx;//coordenada x

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
__global__ void SumaArb(int *m, int WidthM, int WidthN) {
	//obtención id del hilo
	int idBx = blockIdx.x;	int idBy = blockIdx.y;
	int idTx = threadIdx.x;	int idTy = threadIdx.y;

	int id_fil = idBy * TILE_WIDTH + idTy;//coordenada y
	int id_col = idBx * TILE_WIDTH + idTx;//coordenada x

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
			for (i = id_fil - 1; i <= 0 && (c % 2) == 1; i--) {
				aux = m[i*WidthN + id_col];
				if (aux == ficha) {
					m[i*WidthN + id_col] = ficha * 2;
					m[id_fil*WidthN + id_col] = 0;
					c--;//Para que el bucle for termine
				}

				//	(Aclaración) Si estamos entrando en este bucle for,
				//		significa que se ha encontrado una pareja viable anteriormente
				//		por lo que no se filtra si se opera con una ficha no válida
			}
		}
	}
}

//	Ejecución de Movimiento hacia Arriba de las piezas
//	-	Cada hilo toma su ficha (si es distinta de 0) y busca espacios en blanco por encima
//	-	Cuando no encuentra más huecos en la matriz, intercambia su ficha con la del último hueco hallado
//	-	al ser 0, intercambia con una vacía, si no hubiese huecos por encima, la intercambia consigo mismo
//	-	-	Esta función debe ser llamada hasta que no devuelva ningún cambio en la Matriz de Juego
__global__ void exMovArb(int *m, bool *b, int WidthM, int WidthN) {
	//obtención id del hilo
	int idBx = blockIdx.x;	int idBy = blockIdx.y;
	int idTx = threadIdx.x;	int idTy = threadIdx.y;

	int id_fil = idBy * TILE_WIDTH + idTy;//coordenada y
	int id_col = idBx * TILE_WIDTH + idTx;//coordenada x

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
__global__ void SumaAbj(int *m, int WidthM, int WidthN) {
	//obtención id del hilo
	int idBx = blockIdx.x;	int idBy = blockIdx.y;
	int idTx = threadIdx.x;	int idTy = threadIdx.y;

	int id_fil = idBy * TILE_WIDTH + idTy;//coordenada y
	int id_col = idBx * TILE_WIDTH + idTx;//coordenada x

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
			for (i = id_fil + 1; i < WidthM && (c % 2) == 1; i++) {
				aux = m[i*WidthN + id_col];
				if (aux == ficha) {
					m[i*WidthN + id_col] = ficha * 2;
					m[id_fil*WidthN + id_col] = 0;
					c--;//Para que el bucle for termine
				}

				//	(Aclaración) Si estamos entrando en este bucle for,
				//		significa que se ha encontrado una pareja viable anteriormente
				//		por lo que no se filtra si se opera con una ficha no válida
			}
		}
	}
}

//	Ejecución de Movimiento hacia Abajo de las piezas
//	-	Cada hilo toma su ficha (si es distinta de 0) y busca espacios en blanco por debajo de ella
//	-	Cuando no encuentra más huecos en la matriz, intercambia su ficha con la del último hueco hallado
//	-	al ser 0, intercambia con una vacía, si no hubiese huecos por debajo, la intercambia consigo mismo
//	-	-	Esta función debe ser llamada hasta que no devuelva ningún cambio en la Matriz de Juego
__global__ void exMovAbj(int *m, bool *b, int WidthM, int WidthN) {
	//obtención id del hilo
	int idBx = blockIdx.x;	int idBy = blockIdx.y;
	int idTx = threadIdx.x;	int idTy = threadIdx.y;

	int id_fil = idBy * TILE_WIDTH + idTy;//coordenada y
	int id_col = idBx * TILE_WIDTH + idTx;//coordenada x

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

//--------------------------------- Inicialización Ejemplo de VS2015 --------------------------------------------

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int main()
{
	int Width = 5;
	int M[5*5];

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

	imprimeMatriz(M,Width,Width);

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
