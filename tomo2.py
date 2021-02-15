import numpy as np
import matplotlib.pyplot as plt

arqEntrada = input("Entre com o nome ou caminho do arquivo .npy:") #Nome do arquivo.npy
p_vetor = np.load(arqEntrada)
arqEntrada = arqEntrada.replace('2.npy', '1.npy') #Nome do arquivo.npy correspondente do ex 1
p1_vetor = np.load(arqEntrada)
arqEntrada = input("Entre com o nome ou caminho do arquivo .png:") #Nome da imagem
img = plt.imread(arqEntrada) #Transforma a imagem em uma matriz
n=(p_vetor.size+2)//6 #Deduz n a partir do vetor p
#Logica de formacao da matriz A, os arrays a seguir sao auxiliares para usar no metodo kron e gerar a parte superior de A
B=np.ones(n)
I=np.identity(n)
identConcat=np.kron(B,I)
identGrande=np.kron(I,B)
A = np.concatenate((identConcat,identGrande))
#Gera a matriz A do ex1 e sua tranposta
ex1A = A
ex1AT = np.transpose(A)
ex1MatrizGeral = np.dot(ex1AT,ex1A) #Produto matricial entre A transposta e A(ex1), para gerar a matriz que multiplica f no sistema
ex1MatrizResultado = np.dot(ex1AT,p1_vetor) #Produto matricial entre A transposta e p do ex1, para gerar a matriz que e resultado do sistema
#Gera uma identidade espelhada para montar o proximo padrao da matriz A
I=np.flip(I,1)
matrizPadrao = I
#Gera as primeiras n colunas do proximo padrao de matrizes de A
for i in range(n-1):
    Z=np.zeros((1,n))
    matrizPadrao=np.concatenate((matrizPadrao,Z))
#A partir do primeiro modelo, gera as colunas restantes 
for i in range(1,n):
    Z=np.zeros((1,n))
    cont=1
    matrizSoma=Z
    while cont < i:
        matrizSoma=np.concatenate((matrizSoma,Z))
        cont+=1
    matrizSoma=np.concatenate((matrizSoma,I))
    cont=0
    while cont < (n-i-1):
        matrizSoma=np.concatenate((matrizSoma,Z))
        cont+=1
    matrizPadrao=np.hstack([matrizPadrao,matrizSoma])
A = np.concatenate((A,matrizPadrao))
#O padrao abaixo segue a mesma logica acima, porem a repeticao e uma identidade agora
I=np.identity(n)
matrizPadrao = I
for i in range(n-1):
    Z=np.zeros((1,n))
    matrizPadrao=np.concatenate((matrizPadrao,Z))
for i in range(1,n):
    Z=np.zeros((1,n))
    cont=1
    matrizSoma=Z
    while cont < i:
        matrizSoma=np.concatenate((matrizSoma,Z))
        cont+=1
    matrizSoma=np.concatenate((matrizSoma,I))
    cont=0
    while cont < (n-i-1):
        matrizSoma=np.concatenate((matrizSoma,Z))
        cont+=1
    matrizPadrao=np.hstack([matrizPadrao,matrizSoma])
A = np.concatenate((A,matrizPadrao))
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
    ex1MatrizGeral = ex1MatrizGeral + identidade
    #Sera utilizado o metodo de Gauss Siedel para encontrar f
    ex1F = np.zeros(n**2,dtype=float) #inicializacao de f(ex1)
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
                    soma = soma - ex1MatrizGeral[i][j]*ex1F[j]
            #Soma o elemento correspondentedo vetor p e divide pelo elemento da diagonal
            soma = soma + ex1MatrizResultado[i]
            ex1F[i] = soma/ex1MatrizGeral[i][i]
    #Transforma o vetor em matriz e transpoe para que a matriz fique na orientacao certa
    ex1F=np.reshape(ex1F,(n,n))
    ex1F=np.transpose(ex1F)
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
    #Usa o numpy para fazer as normas dos vetores
    erroF=np.linalg.norm(f)
    aux=f-img
    erroDif=np.linalg.norm(aux)
    #Calculo do erro
    erro=100*(erroDif/erroF)
    print("erro=",erro,"%")
    #Gera a figura das duas imagens lado a lado com o titulo respectivo
    titulo = "Imagens gerada da solucao com delta = "+str(delta)
    x=plt.figure()
    x.add_subplot(1,3,1)
    plt.imshow(ex1F,cmap='binary')
    x.add_subplot(1,3, 2)
    plt.imshow(f,cmap='binary')
    x.add_subplot(1,3, 3)
    plt.imshow(img,cmap='binary')
    plt.suptitle(titulo,fontsize=16, y=1)
    plt.show(block=True)
    matrizGeral = matrizGeral - identidade
    ex1MatrizGeral = ex1MatrizGeral - identidade