"""
Pandas y Numpy 
nos ayuda a leer los archivos .csv 
y asigana los datos de estos archivos en una 
estructura de datos que podamos manipular 
"""
import pandas as pd 
import numpy as np
"""
matplotlib
la usamos para generar las graficas necesarias 
para el analisis de los datos 
"""
import matplotlib.pyplot as plt
"""
sklearn
la usamos parea generar y entrenar modelos 
de regrecion lienal para entender el comportamiendo de 
los datos y poder generar predicciones 
"""
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn import datasets, ensemble


# leemos y guaradamos los datos de los archivos .csv en Dataframes
data = pd.read_csv("data.csv", sep=";")
pib_per_regiones = pd.read_csv("pib-per-regiones.csv", sep=";")
xes = pd.read_csv("pib regiones.csv", sep=";")

regiones=["Caribe","Oriental","Central","Pacífica","Bogotá","Antioquia","Valle del Cauca"]


# regrecion años y pib toatl
def LR_anios_PIB(reg):
    # ontenemos las columnas correspondientes de la tabla 
    x=xes["Anios"].values.reshape(-1, 1)
    y=xes[reg].values.reshape(-1, 1)

    # estandarizamos los datos ibtenidos del Dataframe
    sc_x = StandardScaler()
    sc_y = StandardScaler()

    X_std = sc_x.fit_transform(x)
    y_std = sc_y.fit_transform(y)

    # declaramos el modelo de regrecion lineal
    slr = LinearRegression()
    slr.fit(X_std, y_std) # entrenamos el modelo 
    
    result={
        "model":slr,
        "sc_x":sc_x,
        "sc_y":sc_y
    }
    return result

# regrecion años y pib percapita
def LR_anios_PIB_PER(reg):
    # ontenemos las columnas correspondientes de la tabla 
    x=xes["Anios"].values.reshape(-1, 1)
    y=pib_per_regiones[reg].values.reshape(-1, 1)

    # estandarizamos los datos ibtenidos del Dataframe
    sc_x = StandardScaler()
    sc_y = StandardScaler()

    X_std = sc_x.fit_transform(x)
    y_std = sc_y.fit_transform(y)

    # declaramos el modelo de regrecion lineal
    slr = LinearRegression()
    slr.fit(X_std, y_std) # entrenamos el modelo 
    
    result={
        "model":slr,
        "sc_x":sc_x,
        "sc_y":sc_y
    }
    return result

# regrecion porcentaje de PM y PIB total 
def LR_PIB_PB_MLT(reg):
    pb_reg="PB-MLT-"+reg
    pib_reg="PIB-"+reg

    # ontenemos las columnas correspondientes de la tabla 
    x = data[pib_reg].values.reshape(-1, 1)
    y = data[pb_reg].values.reshape(-1, 1)
    
    # estandarizamos los datos ibtenidos del Dataframe
    sc_x = StandardScaler()
    sc_y = StandardScaler()
    
    x_std = sc_x.fit_transform(x)
    y_std = sc_y.fit_transform(y)

    # declaramos el modelo de regrecion
    slr = LinearRegression()  
    slr.fit(x_std, y_std ) # entrenamos el modelo 
    
    result={
        "model":slr,
        "sc_x":sc_x,
        "sc_y":sc_y
    }
    return result

def regression_evaluator(reg, slr1, slr2, slr3, year_start, year_end):
    print(" analisis en la region ",reg)
    print(" anios de ",year_start, " a ",year_end)
    
    while year_start<=year_end:
        print("-"*50)
        # vamos etandarizando el año de prediccion
        year_start_std =  slr1["sc_x"].transform(np.array([year_start]).reshape(-1,1))

        # realizamos la predicciones 
        pib_predict= slr1["sc_y"].inverse_transform(slr1["model"].predict(year_start_std))
        pib_per_predict= slr2["sc_y"].inverse_transform(slr2["model"].predict(year_start_std))

        # estandarizamos el PIB obtenido en la prediccion 
        pib_predict_std = slr3["sc_x"].transform(np.array([pib_predict]).reshape(-1,1))
        # relaizamos la prediccion del indice de pobreza multidimencional
        PB_MLT_predict = slr3["sc_y"].inverse_transform(slr3["model"].predict(pib_predict_std))
        
        # mostramos resultados 
        print("\n prediciones en ele anio ",year_start)
        print("{:^25}{:^20,}{:^20}".format("PIB",round(pib_predict[0][0],2),"Miles de millones de pesos"))
        print("{:^25}{:^20,}{:^5}".format("PIB Per-Capita",round(pib_per_predict[0][0],2),"Pesos"))
        ingr_mes = pib_per_predict[0][0]/12
        print("\n Se deduce que los ingresos mensuales por persona seran aproximandamente de ")
        print("{:^20,}{:^5}".format(round(ingr_mes,2),"Pesos"))

        print("\n Aprocimandamente el porcentaje de poblacion en pobreza multidimencional seria del ",round(PB_MLT_predict[0][0],2)," %")

        year_start+=1
        


def run():
    # solicitamos y validamos la region 
    print(" Analisis de indicadores de pobreza multidimencional a mediano plazo ")
    print(" Regiones")
    i=1
    for r in regiones:
        print("#",i,":",r)
        i+=1
    indr=int(input("\n elija la region a predecir: "))
    reg=regiones[indr-1]

    print("\n ingresa un intervalo de anios a mediano plazo para evaluar")
    year_start=int(input("anio inicial:"))
    year_end=int(input("anio final:"))
    while year_end<=year_start:
        print(" ingresa un anio superior al inicial !!!")
        year_end=int(input("anio final:"))

    # obtenemos los modelos de regrecion 
    slr_x_x_amio = LR_anios_PIB(reg)
    slr_pib_per_x_amio = LR_anios_PIB_PER(reg)
    slr_PB_MLT_PIB = LR_PIB_PB_MLT(reg)
    # evaluamos los modelos de regrecion en el rango de años seleccionado por el usuario
    regression_evaluator(reg, slr_x_x_amio, slr_pib_per_x_amio, slr_PB_MLT_PIB, year_start, year_end)


if __name__=="__main__":
    run()