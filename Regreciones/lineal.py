from functools import reduce
# NOTA : importamos de la libreria functools la funcion reduce para usarla mas adelante 

def lineal(x,y):
    """
    funcion que recibe dos vectores con valores numericos 
    de los cuales se retorna un modelo matematico que representa 
    la recta
            Y = a0 + a1X 
    """
    n=len(x)
    a1 = (n*smtra(x,y) - smtra(x)*smtra(y)) / (n*smtra(x, [], 2) - (smtra(x))**2)
    a0 = prom(y) - a1*prom(x)
    print("y=",a1,"+",a0,"X")

    def model(x):
        return a0 + (a1*x)

    return model

# funciones auxiliares
def smtra(l1,l2=[],elv=1):
    if len(l2)==0:
        return reduce(lambda a,b:(a+b)**elv,l1)
    else:
        l1_l2=[l1[i]*l2[i] for i in range(len(l1))]
        return reduce(lambda a,b:a+b,l1_l2)

prom = lambda l1: reduce(lambda a,b:(a+b),l1) / len(l1)
