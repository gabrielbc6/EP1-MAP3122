import numpy as np
import matplotlib.pyplot as plt

arqEntrada = input("Entre com o nome ou caminho do arquivo .npy:") #Nome do arquivo.npy
p_vetor = np.load(arqEntrada)
arqEntrada = input("Entre com o nome ou caminho do arquivo .png:") #Nome da imagem
img = plt.imread(arqEntrada) #Transforma a imagem em uma matriz
n=p_vetor.size//2 #Deduz n a partir do vetor p
#Logica de formacao da matriz A, os arrays a seguir sao auxiliares para usar no metodo kron e gerar A
B=np.ones(n)
I=np.identity(n)
identConcat=np.kron(B,I)
identGrande=np.kron(I,B)

A = np.concatenate((identConcat,identGrande)) #Concatena as duas partes geradas na logica anterior, gerando A
AT = np.transpose(A) #Transposta de A
matrizGeral = np.dot(AT,A) #Produto matricial entre A transposta e A, para gerar a matriz que multiplica f no sistema
matrizResultado = np.dot(AT,p_vetor) #Produto matricial entre A transposta e p, para gerar a matriz que e resultado do sistema
delta=1
#Faz uma iteracao para cada delta
for k in range(3):
    delta = delta/10 #A cada iteracao diminui o delta para resolver o sistema com um delta menor
    #Identidade do sistema que e somada a matriz geral
    identidade = np.identity((n**2)) 
    identidade = delta*identidade
    matrizGeral = matrizGeral + identidade
    #Sera utilizado o metodo de Gauss Siedel para encontrar f
    f = np.zeros(n**2,dtype=float) #inicializacao de f
    #O loop abaixo define o numero de iteracoes do metodo(100)
    for g in range(100):
        #O loop abaixo percorrera as linhas da matriz A e as linhas do vetor p
        for i in range(n**2):
            soma=0 #a cada linha percorrida a soma e re-inicializada
            #O loop abaixo percorrera as colunas da matriz A e as linhas do vetor f
            for j in range(n**2):
                #Se nao e um elemento da diagonal multiplica-se o elemento da matriz pelo vetor f e subtrai da soma
                if i != j:
                    soma = soma - matrizGeral[i][j]*f[j]
            #Soma o elemento correspondentedo vetor p e divide pelo elemento da diagonal
            soma = soma + matrizResultado[i]
            f[i] = soma/matrizGeral[i][i]
    #Transforma o vetor em matriz e transpoe para que a matriz fique na orientacao certa
    f=np.reshape(f,(n,n))
    f=np.transpose(f)
    #Gera a figura das duas imagens lado a lado com o titulo respectivo
    titulo = "Imagem gerada da solucao com delta = "+str(delta)
    x=plt.figure()
    x.add_subplot(1,2,1)
    plt.imshow(f,cmap='binary')
    x.add_subplot(1,2, 2)
    plt.imshow(img,cmap='binary')
    plt.suptitle(titulo,fontsize=16, y=1)
    plt.show(block=True)
    matrizGeral = matrizGeral - identidade