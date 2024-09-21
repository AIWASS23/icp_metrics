#include <stdio.h>
#include <stdlib.h>
#include <string.h> 

unsigned short int width;	
unsigned short int height;	

void laplacian(unsigned char m[width][height], unsigned char temp[width][height]){  

	//Novo arquivo de saída com valores dos pixels para o pgm
	FILE *arquivo;
	arquivo = fopen("out.txt", "w");

	unsigned short int i;
	unsigned short int j;
	int aux = 0;
	
	fprintf(arquivo, "%hu %hu\n", width, height);	//Escreve na primeira linha as dimensões da matriz
	//Laplacian kernel 5x5
	for(i = 0; i < width; i++){
		for(j = 0; j < height; j++){
			if((i > 1) && (i < width-2) && (j > 1) && (j < height-2)){ //Se não é uma borda da imagem
				
				aux = temp[i-2][j-2] + temp[i-1][j-2] + temp[i][j-2]    + temp[i+1][j-2] + temp[i+2][j-2]
				   	+ temp[i-2][j-1] + temp[i-1][j-1] + temp[i][j-1]    + temp[i+1][j-1] + temp[i+2][j-1]
				    + temp[i-2][j+0] + temp[i-1][j+0] - temp[i][j+0]*24 + temp[i+1][j+0] + temp[i+2][j+0]   
					+ temp[i-2][j+1] + temp[i-1][j+1] + temp[i][j+1]    + temp[i+1][j+1] + temp[i+2][j+1]
					+ temp[i-2][j+2] + temp[i-1][j+2] + temp[i][j+2]    + temp[i+1][j+2] + temp[i+2][j+2];
				
				if(aux < 0)
					m[i][j] = 0;

				else if(aux > 255)
					m[i][j] = 255;

				else
					m[i][j] = aux;

			}
			fprintf(arquivo, "%hhu ", m[i][j]);
		}
		fprintf(arquivo, "\n");
	}

	fclose(arquivo);
}//laplacian

int main(){
  
	FILE *arquivo = NULL;
	char caminhoDoArquivo[12];
	int i;		//variável auxiliar	//DESTROCAR PARA UNSIGNED SHORT INT
	int j;		//variável auxiliar //DESTROCAR PARA UNSIGNED SHORT INT

	//Abre o arquivo (matriz.txt)
	while(arquivo == NULL){
		printf("Insira o caminho do arquivo e pressione Enter: \n");
		fgets(caminhoDoArquivo, sizeof(caminhoDoArquivo), stdin);
		//Remove o último caractere do caminho, pois o fgets armazena a quebra de linha "\n"
		char *p_chr = strchr(caminhoDoArquivo, '\n'); 
		if(p_chr != NULL) 
			*p_chr = '\0';
		arquivo = fopen(caminhoDoArquivo, "r");
	}

	fscanf(arquivo, "%hu %hu", &width, &height);	//Armazena as dimensões da matriz 
	unsigned char matriz[width][height];			//Armazena a matriz de entrada			
	unsigned char temp[width][height];				//Cópia da matriz de entrada			
	//Armazena os valores lidos do arquivo na matriz
	for(i = 0; i < width; i++){
		for(j = 0; j < height; j++){
			fscanf(arquivo, "%hhu", &matriz[i][j]);
			temp[i][j] = matriz[i][j];
		}
	}

	laplacian(matriz, temp);
	fclose(arquivo);

	return 0;
}